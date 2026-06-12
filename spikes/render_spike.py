"""Step 0b 렌더 스파이크: map occupancy + (x,y,theta) pose -> 224x224 ego-centric top-down 래스터.

목적: pyglet 없이 (numpy/PIL만, headless-safe) LeWM 입력 이미지를 결정적으로 합성할 수
있는지 검증. pose는 npz에 없으므로(state=[vel_x,vel_y,ang_z,prev_steer,prev_speed])
centerline csv의 (x,y) + 접선 벡터(tx,ty)로 합성한다.

실행: f1tenth_RL_project venv(py3.8, PIL/numpy/yaml 보유)로 실행.
"""
import csv
import math
import os

import numpy as np
import yaml
from PIL import Image, ImageDraw

RL = "/home/dlacksdn/f1tenth_RL_project"
MAP_YAML = f"{RL}/pkg/src/pkg/maps/map_easy3.yaml"
CENTERLINE = f"{RL}/maps/map_easy3_centerline.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

VIEW_M = 22.4   # ego view 한 변(미터) -> 0.1 m/px @ 224
OUT_PX = 224
CAR_LEN_M, CAR_WID_M = 0.58, 0.31  # f1tenth 차체 (length, width)


def load_map():
    with open(MAP_YAML) as f:
        meta = yaml.safe_load(f)
    png = os.path.join(os.path.dirname(MAP_YAML), meta["image"])
    im = Image.open(png).convert("L")
    return im, float(meta["resolution"]), np.array(meta["origin"][:2], float)


def world_to_px(xy, res, origin, img_h):
    """ROS map convention: origin=좌하단, y축 위로 증가. 이미지 좌표는 y 아래로 증가."""
    px = (xy[0] - origin[0]) / res
    py = img_h - (xy[1] - origin[1]) / res
    return px, py


def render_ego(im, res, origin, x, y, theta):
    """pose 중심·heading-up 고정 224x224 crop."""
    px, py = world_to_px((x, y), res, origin, im.height)
    view_px = VIEW_M / res                      # 1120 px
    margin = int(view_px * 0.75)                # 회전 여유 (>= view/sqrt2)

    # 1) pose 주변 큰 사각형 crop (경계 밖은 occupied=0 으로 채움)
    box = (int(px) - margin, int(py) - margin, int(px) + margin, int(py) + margin)
    patch = Image.new("L", (2 * margin, 2 * margin), 0)
    src = im.crop((max(box[0], 0), max(box[1], 0),
                   min(box[2], im.width), min(box[3], im.height)))
    patch.paste(src, (max(-box[0], 0), max(-box[1], 0)))

    # 2) heading이 항상 위를 향하도록 회전 (PIL rotate=반시계, 이미지 y축 반전 감안)
    deg = 90.0 - math.degrees(theta)
    patch = patch.rotate(-deg, resample=Image.BILINEAR,
                         center=(margin, margin), fillcolor=0)

    # 3) 중앙 view_px 크기로 crop -> 224 리사이즈
    half = view_px / 2
    patch = patch.crop((margin - half, margin - half, margin + half, margin + half))
    patch = patch.resize((OUT_PX, OUT_PX), Image.BILINEAR)

    # 4) RGB 변환 + 차량 마커 (중앙, heading-up 고정이므로 항상 위쪽 삼각형)
    rgb = Image.merge("RGB", (patch, patch, patch))
    draw = ImageDraw.Draw(rgb)
    sc = OUT_PX / VIEW_M  # px per meter
    l, w = CAR_LEN_M * sc, CAR_WID_M * sc
    cx = cy = OUT_PX / 2
    draw.polygon([(cx, cy - l / 2), (cx - w / 2, cy + l / 2),
                  (cx + w / 2, cy + l / 2)], fill=(255, 60, 60))
    return rgb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    im, res, origin = load_map()
    rows = list(csv.DictReader(open(CENTERLINE)))
    n = len(rows)
    idxs = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 2]

    for k, i in enumerate(idxs):
        r = rows[i]
        x, y = float(r["x"]), float(r["y"])
        theta = math.atan2(float(r["ty"]), float(r["tx"]))
        img = render_ego(im, res, origin, x, y, theta)
        # output no-overwrite: 기존 파일 있으면 번호 증분
        v = 1
        while os.path.exists(f"{OUT_DIR}/ego_s{float(r['s']):.0f}m-{v}.png"):
            v += 1
        path = f"{OUT_DIR}/ego_s{float(r['s']):.0f}m-{v}.png"
        img.save(path)
        arr = np.asarray(img)
        print(f"[{k}] s={float(r['s']):7.2f}m pose=({x:6.2f},{y:6.2f},{math.degrees(theta):6.1f}deg)"
              f" -> {path} shape={arr.shape} dtype={arr.dtype}"
              f" uniq_gray={len(np.unique(arr[..., 0]))}")

    print("SPIKE_OK")


if __name__ == "__main__":
    main()
