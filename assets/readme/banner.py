"""HelloOperator repo banner — indigo seam/glitch family style.
Motif: abstracted switchboard; one bright route patched through. No words."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

random.seed(7); np.random.seed(7)
W, H = 1920, 1080

# ---- field: violet, subtle vertical gradient ----
base_top = np.array([96, 34, 152]); base_bot = np.array([62, 20, 110])
grad = np.linspace(0, 1, H)[:, None, None]
img = (base_top * (1 - grad) + base_bot * grad).astype(np.uint8)
img = np.broadcast_to(img, (H, W, 3)).copy()
im = Image.fromarray(img)
d = ImageDraw.Draw(im, "RGBA")

# ---- socket grid (jack field) ----
cols, rows = 9, 4
x0, y0, dx, dy = 300, 330, 165, 150
sockets = {}
for r in range(rows):
    for c in range(cols):
        x = x0 + c * dx + random.randint(-6, 6)
        y = y0 + r * dy + random.randint(-5, 5)
        sockets[(c, r)] = (x, y)
        rad = 26
        d.ellipse([x-rad-6, y-rad-6, x+rad+6, y+rad+6], outline=(124, 92, 255, 70), width=3)
        d.ellipse([x-rad, y-rad, x+rad, y+rad], fill=(14, 11, 24, 255))
        d.ellipse([x-rad+7, y-rad+7, x+rad-9, y+rad-9], outline=(60, 30, 90, 160), width=2)

def bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**3*np.array(p0) + 3*(1-t)**2*t*np.array(p1)
            + 3*(1-t)*t**2*np.array(p2) + t**3*np.array(p3))

def cable(d, a, b, sag, col, width, split=6):
    (ax, ay), (bx, by) = a, b
    c1 = (ax + (bx-ax)*0.25, ay + sag); c2 = (ax + (bx-ax)*0.75, by + sag)
    pts = [tuple(p) for p in bezier((ax, ay), c1, c2, (bx, by))]
    # channel-split fringes then body
    d.line([(x-split, y) for x, y in pts], fill=(255, 46, 99, 150), width=width)
    d.line([(x+split, y) for x, y in pts], fill=(25, 227, 255, 130), width=width)
    d.line(pts, fill=col, width=width)

# dim cables: routes not taken
dim = (34, 18, 66, 255)
cable(d, sockets[(1, 0)], sockets[(4, 2)], 260, dim, 16)
cable(d, sockets[(6, 0)], sockets[(2, 3)], 300, dim, 16)
cable(d, sockets[(7, 1)], sockets[(5, 3)], 240, dim, 14)
cable(d, sockets[(3, 1)], sockets[(0, 2)], 200, dim, 14)
# the routed call: bright, glowing
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
cable(gd, sockets[(0, 1)], sockets[(8, 2)], 320, (201, 184, 255, 255), 12, split=0)
glow = glow.filter(ImageFilter.GaussianBlur(14))
im.paste(glow, (0, 0), glow)
cable(d, sockets[(0, 1)], sockets[(8, 2)], 320, (236, 226, 255, 255), 9, split=7)
# lit plugs at its two ends
for k in ((0, 1), (8, 2)):
    x, y = sockets[k]
    d.ellipse([x-13, y-13, x+13, y+13], fill=(236, 226, 255, 255))
    d.ellipse([x-22, y-22, x+22, y+22], outline=(201, 184, 255, 160), width=4)

img = np.array(im)

# ---- seam: horizontal fracture with row displacement ----
seam_y = 560
for band, shift in ((slice(seam_y-38, seam_y-12), 22), (slice(seam_y-12, seam_y+6), -48),
                    (slice(seam_y+6, seam_y+22), 12)):
    img[band] = np.roll(img[band], shift, axis=1)
for x0_, x1_ in ((260, 520), (900, 1060), (1420, 1660)):
    img[seam_y-1:seam_y+1, x0_:x1_] = np.minimum(
        img[seam_y-1:seam_y+1, x0_:x1_] + 90, 255)

# ---- global chromatic aberration: split channels in horizontal bands ----
for _ in range(4):
    y = random.randint(0, H-60); h = random.randint(14, 60)
    s = random.choice([-5, -4, 4, 5, 6])
    img[y:y+h, :, 0] = np.roll(img[y:y+h, :, 0], s, axis=1)
    img[y:y+h, :, 2] = np.roll(img[y:y+h, :, 2], -s, axis=1)
im = Image.fromarray(img)
d = ImageDraw.Draw(im, "RGBA")

# ---- datamosh rects ----
for _ in range(16):
    x, y = random.randint(40, W-220), random.randint(30, H-120)
    w, h = random.randint(40, 190), random.randint(16, 85)
    kind = random.random()
    if kind < 0.45: col = (196, 120, 220, random.randint(40, 80))     # lighter
    elif kind < 0.8: col = (30, 8, 40, random.randint(70, 130))       # darker
    else: col = (124, 92, 255, random.randint(35, 60))                # indigo
    d.rectangle([x, y, x+w, y+h], fill=col)

# ---- thin white vertical hairlines ----
for _ in range(15):
    x = random.randint(80, W-80)
    y1 = random.randint(40, 500); ln = random.randint(180, 780)
    d.line([(x, y1), (x + random.randint(-3, 3), min(y1+ln, H-40))],
           fill=(255, 255, 255, random.randint(150, 220)), width=1)

# ---- scanlines + grain + banding ----
img = np.array(im).astype(np.int16)
scan = (np.arange(H) % 4 < 1)[:, None]
img[scan.repeat(W, 1)] -= 10
noise = np.random.normal(0, 5.5, (H, W, 1)).astype(np.int16)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

Image.fromarray(img).save("OUT/hero.jpg", quality=82)
print("written")
