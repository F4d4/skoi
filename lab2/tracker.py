"""Трекинг плоского объекта с первого кадра видео."""

import argparse
from pathlib import Path

import cv2
import numpy as np


class PlanarTracker:
    def __init__(self, first_frame, roi=None):
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        x, y, w, h = roi or (0, 0, width, height)
        if min(x, y) < 0 or min(w, h) < 2 or x + w > width or y + h > height:
            raise ValueError(f"ROI должен находиться внутри кадра {width}x{height}")

        self.template = gray[y:y + h, x:x + w]
        self.corners = np.float32([[x, y], [x + w - 1, y],
                                   [x + w - 1, y + h - 1], [x, y + h - 1]])
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.015)
        self.ref_keys, self.ref_desc = self.sift.detectAndCompute(self.template, None)
        if self.ref_desc is None or len(self.ref_keys) < 8:
            raise ValueError("На первом кадре слишком мало деталей для поиска объекта")
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.polygon = self.corners.copy()
        self.previous = gray
        self.points = self._new_points(gray, self.polygon)
        self.frame_number = 0

    @staticmethod
    def _valid(polygon, shape):
        height, width = shape
        if polygon is None or not np.isfinite(polygon).all():
            return False
        contour = np.float32(polygon).reshape(-1, 1, 2)
        area = cv2.contourArea(contour)
        sides = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1)
        left, top = polygon.min(axis=0)
        right, bottom = polygon.max(axis=0)
        return (cv2.isContourConvex(contour) and 80 < area < 4 * width * height
                and min(sides) > 8 and left < width and right > 0
                and top < height and bottom > 0
                and np.abs(polygon).max() < 3 * max(width, height))

    @staticmethod
    def _new_points(gray, polygon, existing=None):
        count = 0 if existing is None else len(existing)
        if count >= 300:
            return existing
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        if existing is not None:
            for px, py in existing.reshape(-1, 2):
                cv2.circle(mask, (round(float(px)), round(float(py))), 5, 0, -1)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=300 - count, qualityLevel=0.01,
                                         minDistance=4, mask=mask)
        if points is None:
            return existing
        return points if existing is None else np.concatenate((existing, points))

    def _detect(self, gray):
        keys, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(desc) < 2:
            return None
        pairs = self.matcher.knnMatch(self.ref_desc, desc, k=2)
        good = sorted((a for pair in pairs if len(pair) == 2
                       for a, b in [pair] if a.distance < 0.75 * b.distance),
                      key=lambda match: match.distance)
        # Одна точка кадра не должна соответствовать нескольким точкам шаблона.
        unique, used = [], set()
        for match in good:
            if match.trainIdx not in used:
                unique.append(match)
                used.add(match.trainIdx)
        if len(unique) < 8:
            return None

        source = np.float32([self.ref_keys[m.queryIdx].pt for m in unique]).reshape(-1, 1, 2)
        target = np.float32([keys[m.trainIdx].pt for m in unique]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if matrix is None or inliers is None or inliers.sum() < 8 or inliers.mean() < 0.4:
            return None
        local_corners = self.corners - self.corners[0]
        polygon = cv2.perspectiveTransform(local_corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        return polygon if self._valid(polygon, gray.shape) else None

    def _flow(self, gray):
        if self.polygon is None or self.points is None or len(self.points) < 8:
            return None, None
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **options)
        if moved is None:
            return None, None
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, **options)
        if back is None:
            return None, None
        keep = ((forward.ravel() == 1) & (backward.ravel() == 1)
                & (np.linalg.norm(back - self.points, axis=2).ravel() < 1.5))
        if keep.sum() < 8:
            return None, None
        old, new = self.points[keep], moved[keep]
        matrix, inliers = cv2.findHomography(old, new, cv2.RANSAC, 3.0)
        if matrix is None or inliers is None or inliers.sum() < 8 or inliers.mean() < 0.5:
            return None, None
        polygon = cv2.perspectiveTransform(self.polygon.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        if not self._valid(polygon, gray.shape):
            return None, None
        return polygon, new[inliers.ravel() == 1]

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_number += 1
        flow_polygon, flow_points = self._flow(gray)
        detected = self._detect(gray) if self.frame_number % 5 == 0 or flow_polygon is None else None
        if detected is not None:
            self.polygon = detected
            self.points = self._new_points(gray, detected)
        elif flow_polygon is not None:
            self.polygon = flow_polygon
            self.points = flow_points
            if len(self.points) < 100 or self.frame_number % 15 == 0:
                self.points = self._new_points(gray, self.polygon, self.points)
        else:
            self.polygon = None
            self.points = None
        self.previous = gray
        return self.polygon


def process_video(video, output_dir, label, roi=None):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video}")
    writer = None
    try:
        ok, first = capture.read()
        if not ok:
            raise ValueError(f"В видео нет кадров: {video}")
        tracker = PlanarTracker(first, roi)
        height, width = first.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{video.stem}_tracked.avi"
        if output.resolve() == video.resolve():
            raise ValueError("Исходный и выходной файлы совпадают")
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Не удалось создать выходное видео: {output}")

        total = found = 0
        frame = first
        while True:
            polygon = tracker.polygon if total == 0 else tracker.update(frame)
            if polygon is not None:
                found += 1
                cv2.polylines(frame, [np.int32(polygon)], True, (0, 255, 0), 2, cv2.LINE_AA)
                left, top = polygon.min(axis=0)
                cv2.putText(frame, label, (max(5, int(left)), max(22, int(top) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            writer.write(frame)
            total += 1
            ok, frame = capture.read()
            if not ok:
                break
        print(f"{video.name}: рамка на {found}/{total} кадрах ({100 * found / total:.1f} %); {output}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path,
                        help="Видеофайлы; без аргументов обрабатываются три тестовых видео")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--label", default="Mona Lisa", help="Подпись латиницей")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Область объекта на первом кадре (по умолчанию весь кадр)")
    args = parser.parse_args()
    if not args.label.isascii():
        parser.error("OpenCV выводит подпись только латиницей")
    if not args.videos:
        folder = Path("test-videos/test-videos")
        args.videos = [folder / f"{name}.avi" for name in
                       ("mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit")]
    for video in args.videos:
        if not video.is_file():
            parser.error(f"Видео не найдено: {video}")
        try:
            process_video(video, args.output_dir, args.label, args.roi)
        except (ValueError, cv2.error) as error:
            parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
