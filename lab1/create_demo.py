import random
from pathlib import Path

from PIL import Image, ImageDraw


width, height = 160, 120
random.seed(3)
image = Image.new("RGB", (width, height))
pixels = image.load()

for y in range(height):
    for x in range(width):
        noise = random.randint(-18, 18)
        pixels[x, y] = (
            max(0, min(255, 35 + x + noise)),
            max(0, min(255, 50 + y + noise)),
            max(0, min(255, 190 - x // 2 + noise)),
        )

draw = ImageDraw.Draw(image)
draw.rectangle((15, 18, 64, 96), fill=(242, 210, 55))
draw.ellipse((78, 20, 145, 88), fill=(220, 48, 60))
draw.line((0, 110, 159, 5), fill=(255, 255, 255), width=3)

path = Path("examples/input.png")
path.parent.mkdir(parents=True, exist_ok=True)
image.save(path)
print(path)
