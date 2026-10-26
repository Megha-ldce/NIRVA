#!/usr/bin/env python3
"""nirva_hsv_probe.py - what colours does the camera really see?

  ros2 topic pub --once /nirva/snapshot std_msgs/String "data: go"   # saves /tmp/nirva_frame.png
  python3 nirva_hsv_probe.py [/tmp/nirva_frame.png]

Prints every red / green / blue-ish blob (loose thresholds) with its bounding box and
median / min / max HSV, so a box can be told apart from a wall of the same hue by
brightness (V) or saturation (S).
"""
import sys

import cv2
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/nirva_frame.png'
img = cv2.imread(path)
if img is None:
    sys.exit(f'cannot read {path}')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
LOOSE = {
    'red':   [((0, 80, 40), (10, 255, 255)), ((170, 80, 40), (179, 255, 255))],
    'green': [((35, 60, 30), (90, 255, 255))],
    'blue':  [((95, 60, 30), (135, 255, 255))],
}
print(f'{path}: {img.shape[1]}x{img.shape[0]}')
for name, ranges in LOOSE.items():
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows = sorted(((int(st[i, 4]), i) for i in range(1, n) if st[i, 4] >= 40), reverse=True)
    print(f'--- {name}: {len(rows)} blob(s)')
    for area, i in rows[:6]:
        x, y, w, h = (int(v) for v in st[i, :4])
        px = hsv[lab == i]
        med = np.median(px, axis=0).astype(int)
        print(f'  bbox x={x} y={y} w={w} h={h} area={area}  medianHSV={tuple(med)}  '
              f'V range {int(px[:, 2].min())}-{int(px[:, 2].max())}')