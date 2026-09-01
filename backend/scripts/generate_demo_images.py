"""
Follica Realistic Clinical Scalp Sample Generator.

Design Intent:
Produces realistic photographic scalp assets with natural dermatoscopic textures,
follicular units, hair fiber trajectories, stratum corneum micro-features, and diffuse lighting
for the 1-Click Interactive Demo and Before/After Hair Diary slider.
"""

import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def generate_realistic_scalp(filename: str, has_erythema: bool = True, flake_density: float = 0.08, seed: int = 42):
    w, h = 512, 512
    np.random.seed(seed)

    # 1. Base skin tone with realistic epidermal chrominance & dermatoscopic radial vignette
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1.0 - 0.22 * (dist_from_center / max_dist)**1.5

    # Natural scalp skin tone (Fitzpatrick Type III-IV baseline)
    r_chan = (195 + 15 * np.sin(x / 40.0) + 10 * np.cos(y / 50.0)) * vignette
    g_chan = (150 + 12 * np.sin(x / 45.0) + 8 * np.cos(y / 35.0)) * vignette
    b_chan = (130 + 10 * np.cos(x / 30.0) + 6 * np.sin(y / 40.0)) * vignette

    # Dermal vascular erythema flush if present
    if has_erythema:
        erythema_dist = np.exp(-((x - 240)**2 + (y - 260)**2) / (2 * 95**2))
        r_chan += 28 * erythema_dist
        g_chan -= 8 * erythema_dist
        b_chan -= 10 * erythema_dist

    # Fine stratum corneum micro-relief texture noise
    skin_noise = np.random.normal(0, 4.0, (h, w))
    r_chan += skin_noise
    g_chan += skin_noise
    b_chan += skin_noise

    arr = np.stack([
        np.clip(r_chan, 0, 255),
        np.clip(g_chan, 0, 255),
        np.clip(b_chan, 0, 255)
    ], axis=2).astype(np.uint8)

    img = Image.fromarray(arr).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    # 2. Natural Anatomical Scalp Parting Line
    parting_x = w // 2
    for py in range(0, h, 2):
        jitter = int(np.sin(py / 35.0) * 14 + np.cos(py / 18.0) * 6)
        px = parting_x + jitter
        draw.ellipse([px - 14, py - 4, px + 14, py + 4], fill=(235, 205, 185, 45))
        
        # Micro-flakes along stratum corneum
        if np.random.rand() < flake_density:
            fw = np.random.randint(2, 6)
            fx_offset = np.random.randint(-16, 16)
            draw.ellipse([px + fx_offset, py, px + fx_offset + fw, py + fw], fill=(248, 248, 242, 190))

    # 3. Follicular Ostia and Natural Hair Shaft Trajectories
    num_follicular_units = 190
    for _ in range(num_follicular_units):
        fx = np.random.randint(15, w - 15)
        fy = np.random.randint(15, h - 15)
        
        dist_to_parting = abs(fx - (parting_x + int(np.sin(fy / 35.0) * 14)))
        if dist_to_parting < 14 and np.random.rand() > 0.15:
            continue
            
        # Peripilar ostium
        draw.ellipse([fx - 3, fy - 3, fx + 3, fy + 3], fill=(130, 90, 75, 130))
        draw.ellipse([fx - 1, fy - 1, fx + 1, fy + 1], fill=(50, 30, 25, 210))
        
        num_hairs = np.random.choice([1, 2, 3], p=[0.25, 0.55, 0.20])
        for h_idx in range(num_hairs):
            hair_angle = math.radians(np.random.uniform(70, 110) if fx > parting_x else np.random.uniform(250, 290))
            hair_len = np.random.randint(140, 300)
            hair_width = np.random.choice([2, 3])
            hair_color = (np.random.randint(25, 42), np.random.randint(18, 28), np.random.randint(14, 22), 235)
            
            pts = [(fx + h_idx * 2, fy)]
            curr_x, curr_y = float(fx + h_idx * 2), float(fy)
            steps = 20
            for s in range(steps):
                curr_x += (hair_len / steps) * math.cos(hair_angle) + np.sin(s * 0.35) * 2.2
                curr_y += (hair_len / steps) * math.sin(hair_angle) + 2.0
                pts.append((int(curr_x), int(curr_y)))
                
            draw.line(pts, fill=hair_color, width=hair_width)

    # Blend and smooth
    img_rgb = img.convert("RGB").filter(ImageFilter.SMOOTH_MORE)
    img_rgb.save(filename, "JPEG", quality=95)
    print(f"Generated {filename}")


if __name__ == "__main__":
    generate_realistic_scalp("app/static/sample_scalp_demo.jpg", has_erythema=True, flake_density=0.10, seed=42)
    generate_realistic_scalp("app/static/sample_baseline.jpg", has_erythema=True, flake_density=0.12, seed=101)
    generate_realistic_scalp("app/static/sample_followup.jpg", has_erythema=False, flake_density=0.02, seed=202)
