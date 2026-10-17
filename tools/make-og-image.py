#!/usr/bin/env python3
"""Generate a 1200x630 OG image for CS231n (cs231n.github.io).

Pillow-only, byte-reproducible. Clean academic card: light ground,
course-blue (#2a7ae2 / #205caa chain from css/main.css) + dark text,
title + subtitle + topic pills + domain.
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (247, 246, 241)          # #f7f6f1 site background
BLUE = (42, 122, 226)         # #2a7ae2 link
DARK = (51, 51, 51)           # #333
GREY = (129, 129, 129)        # #818181
WHITE = (255, 255, 255)

FONT_DIR = os.environ.get("FONT_DIR", "/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = os.path.join(FONT_DIR, name)
    if not os.path.exists(p):
        p = os.path.join(FONT_DIR, "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")
    return ImageFont.truetype(p, size)


def rounded(d, box, r, fill):
    d.rounded_rectangle(box, radius=r, fill=fill)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # top accent bar
    d.rectangle([0, 0, W, 10], fill=BLUE)

    # Title
    f_title = font(72, bold=True)
    f_sub = font(34)
    f_pill = font(30, bold=True)
    f_domain = font(30, bold=True)
    f_foot = font(26)

    d.text((70, 74), "CS231n", font=f_title, fill=DARK)
    d.text((70, 170), "Deep Learning for Computer Vision", font=f_sub, fill=GREY)

    # separator line
    d.line([70, 260, W - 70, 260], fill=(210, 208, 200), width=2)

    # topic pills
    pills = ["Convolutional Networks", "RNNs & Transformers", "Object Detection", "Neural Nets 1-3"]
    x = 70
    y = 300
    for p_ in pills:
        w = d.textlength(p_, font=f_pill) + 44
        rounded(d, [x, y, x + w, y + 66], 33, BLUE)
        d.text((x + 22, y + 14), p_, font=f_pill, fill=WHITE)
        x += w + 18
        if x > 940:  # wrap
            x = 70
            y += 86

    # footer bar with domain
    d.rounded_rectangle([70, H - 92, 430, H - 44], 24, fill=DARK)
    d.text((108, H - 84), "cs231n.github.io", font=f_domain, fill=WHITE)
    # subtle line-art: 3x3 grid motif hint (computer vision)
    gx = W - 260
    gy = H - 150
    for i in range(3):
        for j in range(3):
            d.rectangle([gx + i * 46, gy + j * 46, gx + i * 46 + 30, gy + j * 46 + 30],
                        outline=BLUE, width=3)

    out = "og-image.png"
    img.save(out, "PNG")
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
