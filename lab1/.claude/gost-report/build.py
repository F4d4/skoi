import os
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gost_report import GOST_PROFILE, Report, TitleConfig


root = Path(__file__).resolve().parents[2]
figures = root / "docs" / "figures"
figures.mkdir(parents=True, exist_ok=True)

items = [
    ("Исходное", root / "examples" / "input.png"),
    ("NumPy, 5 x 5", root / "examples" / "gaussian_numpy.png"),
    ("Python, 5 x 5", root / "examples" / "gaussian_python.png"),
    ("NumPy, 7 x 7", root / "examples" / "sigma2" / "gaussian_numpy.png"),
]

panel_width = 480
panel_height = 360
gap = 24
top = 88
canvas = Image.new("RGB", (4 * panel_width + 5 * gap, top + panel_height + gap), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)

for index, (label, path) in enumerate(items):
    x = gap + index * (panel_width + gap)
    with Image.open(path) as image:
        canvas.paste(image.convert("RGB").resize((panel_width, panel_height), Image.Resampling.NEAREST), (x, top))
    draw.text((x + panel_width / 2, 24), label, font=font, fill="black", anchor="mt")
    draw.rectangle((x, top, x + panel_width - 1, top + panel_height - 1), outline="#9a9a9a", width=2)

comparison = figures / "comparison.png"
canvas.save(comparison)

report = Report(
    TitleConfig(
        work_type="Отчёт по лабораторной работе",
        work_number="№1",
        topic="Усредняющий фильтр Гаусса",
        variant="3",
        student_name=os.environ.get("REPORT_STUDENT_NAME", "Фамилия И.О."),
        student_group=os.environ.get("REPORT_GROUP", "________"),
        year="2026",
        university_full=os.environ.get("REPORT_UNIVERSITY", ""),
    ),
    profile=replace(GOST_PROFILE, ministry=""),
    project_root=root,
)
report.doc.sections[1].different_first_page_header_footer = False

report.h1("Описание работы")
report.text("Реализован фильтр Гаусса для цветных изображений с настройкой размера ядра и σ. Алгоритм выполнен с готовой функцией свёртки NumPy и вручную на Python.")
report.h2("Результаты")
report.figure(comparison, "Исходное изображение и результаты фильтрации", width_cm=14)
report.text("При 5 x 5 и σ = 1,2 версии совпали по пикселям. Справа показан результат для 7 x 7 и σ = 2,0.")
report.figure(figures / "benchmark.png", "Медианное время девяти запусков на изображении 160 x 120 пикселей", width_cm=13.5)
report.h2("Вывод")
report.text("На проверенных изображениях версии работают одинаково. NumPy ускорил обработку примерно в 24-40 раз.")

output = report.save(root / "docs" / "report_variant3.docx")
print(output)
