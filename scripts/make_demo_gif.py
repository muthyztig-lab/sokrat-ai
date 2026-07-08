from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "sokrat-demo.gif"

BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (201, 209, 217)
GREEN = (63, 185, 80)
BLUE = (88, 166, 255)
DIM = (110, 118, 129)
YELLOW = (210, 153, 34)
DOT_R = (255, 95, 86)
DOT_Y = (255, 189, 46)
DOT_G = (39, 201, 63)

WIDTH = 860
PAD = 22
BAR_H = 34
LINE_H = 28
FONT_SIZE = 19

LINES: list[tuple[str, tuple[int, int, int]]] = [
    ("$ sokrat chat --course \"Фінансова грамотність\"", FG),
    ("", FG),
    ("СТУДЕНТ  > поясни складні відсотки", GREEN),
    ("   ... шукає відповідь у матеріалах курсу", DIM),
    ("SOKRAT   > Спершу подумай: чому вклад росте усе швидше,", BLUE),
    ("           а не рівномірно?", BLUE),
    ("", FG),
    ("СТУДЕНТ  > бо відсотки нараховуються і на самі відсотки", GREEN),
    ("   ... запам'ятав, що тему засвоєно", DIM),
    ("SOKRAT   > Саме так. Це і є ефект складного відсотка.", BLUE),
    ("", FG),
    ("--- звіт викладачу --------------------------------------", YELLOW),
    ("Олена: складні відсотки — засвоєно. Людина не потрібна.", FG),
]


def _font(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    raise SystemExit(
        "No monospace font with Cyrillic found. Set a .ttf path in _font()."
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font = _font(
        [
            "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "DejaVuSansMono.ttf",
        ],
        FONT_SIZE,
    )
    height = BAR_H + PAD * 2 + LINE_H * len(LINES)

    def render(n_visible: int, cursor: bool) -> Image.Image:
        img = Image.new("RGB", (WIDTH, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, WIDTH, BAR_H], fill=BAR)
        for i, color in enumerate((DOT_R, DOT_Y, DOT_G)):
            cx = 20 + i * 22
            d.ellipse([cx - 6, BAR_H // 2 - 6, cx + 6, BAR_H // 2 + 6], fill=color)
        d.text((WIDTH // 2, BAR_H // 2), "sokrat — demo", font=font, fill=DIM, anchor="mm")

        y = BAR_H + PAD
        for idx in range(n_visible):
            text, color = LINES[idx]
            d.text((PAD, y), text, font=font, fill=color)
            y += LINE_H
        if cursor and n_visible < len(LINES) + 1:
            cy = BAR_H + PAD + max(0, n_visible - 1) * LINE_H
            tx = PAD
            if n_visible >= 1:
                tx = PAD + int(font.getlength(LINES[min(n_visible, len(LINES)) - 1][0])) + 4
            d.rectangle([tx, cy + 3, tx + 10, cy + LINE_H - 6], fill=FG)
        return img

    frames: list[Image.Image] = []
    durations: list[int] = []
    for n in range(1, len(LINES) + 1):
        frames.append(render(n, cursor=True))
        durations.append(90 if LINES[n - 1][0] == "" else 620)
    frames.append(render(len(LINES), cursor=False))
    durations.append(3500)

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
