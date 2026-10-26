#!/usr/bin/env python3
"""
perception_node.py - NIRVA perception.

Camera frame -> HSV colour segmentation -> bearing from pixel column ->
lidar range at that bearing -> map-frame (x, y) -> small world model.

Publishes
  /nirva/world_model   std_msgs/String  JSON, 2 Hz:
        {"objects": [{"label","x","y","frame","conf","n_recent","n_total",
                      "spread_m","age_s"}]}
  /nirva/debug_image   sensor_msgs/Image  frame with boxes + ranges drawn

Tuning helper
  ros2 topic pub --once /nirva/snapshot std_msgs/String "data: go"
  -> saves the current frame to /tmp/nirva_frame.png (sample HSV from it with
     python3 -c "import cv2;i=cv2.cvtColor(cv2.imread('/tmp/nirva_frame.png'),cv2.COLOR_BGR2HSV);print(i[y,x])")

Design notes
  * Position = median of the last N observations, so a box that gets PUSHED
    is tracked: the estimate follows it within a second or two.
  * A detection is only used if the lidar range agrees with the range implied
    by the box's apparent size (rejects lidar hits on a wall behind the box).
  * Detections whose bounding box touches the left/right image border are
    skipped (truncated box -> unreliable bearing and size).
  * The lidar is assumed to be at the base origin and the camera near the
    centre line (a few cm of offset -> a few cm of error). The boxes must be
    taller than the lidar scan plane (0.25 m box vs ~0.17 m plane).
"""

import json
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

# OpenCV HSV: H 0-179, S/V 0-255. Red wraps around hue 0, so it has 2 ranges.
# Tune these with /nirva/snapshot if lighting changes the saturation.
COLORS = {
    'red_box':   [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    'green_box': [((40, 100, 50), (85, 255, 255))],
    'blue_box':  [((100, 120, 50), (130, 255, 255))],
}
DRAW = {'red_box': (0, 0, 255), 'green_box': (0, 255, 0), 'blue_box': (255, 0, 0)}


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Track:
    """Rolling window of map-frame observations for one label."""

    def __init__(self, history: int):
        self.obs = deque(maxlen=history)
        self.total = 0
        self.last_seen = 0.0
        self.frame = 'map'

    def add(self, x: float, y: float, frame: str):
        self.obs.append((x, y))
        self.total += 1
        self.last_seen = time.time()
        self.frame = frame

    def summary(self):
        xs = np.array([o[0] for o in self.obs])
        ys = np.array([o[1] for o in self.obs])
        mx, my = float(np.median(xs)), float(np.median(ys))
        # MAD-based spread: robust to a couple of outliers (sightings taken while the robot was moving)
        spread = (float(max(np.median(np.abs(xs - mx)), np.median(np.abs(ys - my))) * 1.4826)
                  if len(xs) > 1 else 0.0)
        n = len(self.obs)
        conf = min(1.0, n / 5.0) * (1.0 if spread < 0.15 else 0.5)
        return mx, my, spread, n, conf


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('nirva_perception_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('box_width_m', 0.20)
        self.declare_parameter('center_offset_m', 0.10)   # lidar sees the near FACE; centre is ~half a box deeper
        self.declare_parameter('min_area_px', 60)
        self.declare_parameter('process_hz', 5.0)
        self.declare_parameter('history', 7)
        self.declare_parameter('fallback_hfov_rad', 1.085)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('edge_gap_m', 0.10)   # a real box has nothing CLOSER than itself right beside it
        self.declare_parameter('mode', 'lidar')              # 'lidar' = lidar-first (robust to same-coloured walls); 'contour' = old colour-first
        self.declare_parameter('box_height_m', 0.25)
        self.declare_parameter('cluster_min_width_m', 0.10)
        self.declare_parameter('cluster_max_width_m', 0.40)

        gp = self.get_parameter
        self.target_frame = gp('target_frame').value
        self.base_frame = gp('base_frame').value
        self.box_w = float(gp('box_width_m').value)
        self.center_off = float(gp('center_offset_m').value)
        self.min_area = int(gp('min_area_px').value)
        self.period = 1.0 / float(gp('process_hz').value)
        self.history = int(gp('history').value)
        self.fallback_hfov = float(gp('fallback_hfov_rad').value)
        self.publish_debug = bool(gp('publish_debug').value)
        self.mode = str(gp('mode').value)
        self.edge_gap = float(gp('edge_gap_m').value)
        self.box_h = float(gp('box_height_m').value)
        self.cl_min_w = float(gp('cluster_min_width_m').value)
        self.cl_max_w = float(gp('cluster_max_width_m').value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tracks = {}
        self.scan = None
        self._scan_wall = 0.0
        self.fx = None
        self.cx = None
        self._last_proc = 0.0
        self._latest_bgr = None
        self._frames = 0
        self._warned_no_scan = False
        self._offsets = {}

        self.create_subscription(Image, gp('image_topic').value, self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, gp('camera_info_topic').value, self._on_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, gp('scan_topic').value, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/nirva/snapshot', self._on_snapshot, 10)

        self.model_pub = self.create_publisher(String, '/nirva/world_model', 5)
        self.debug_pub = self.create_publisher(Image, '/nirva/debug_image', 2)
        self.create_timer(0.5, self._publish_model)
        self.create_timer(8.0, self._watchdog)
        self.create_timer(5.0, self._status)

        self.get_logger().info(
            f"NIRVA perception up. image={gp('image_topic').value} scan={gp('scan_topic').value} "
            f'frame={self.target_frame} box_width={self.box_w} m')

    # ------------------------------------------------------------- inputs

    def _on_info(self, msg: CameraInfo):
        if msg.k[0] > 1.0:
            self.fx, self.cx = float(msg.k[0]), float(msg.k[2])

    def _on_scan(self, msg: LaserScan):
        self.scan = msg
        self._scan_wall = time.time()

    def _diagnose(self, bgr):
        """Per colour: how many pixels match, the biggest blobs and their median HSV. Also saves the masks."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        for label, ranges in COLORS.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(f'/tmp/nirva_mask_{label}.png', mask)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parts = []
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
                a = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                cm = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(cm, [c], -1, 255, -1)
                med = np.median(hsv[cm > 0], axis=0).astype(int).tolist()
                parts.append(f'bbox=({x},{y},{w}x{h}) area={a:.0f} fill={a / max(1, w * h):.2f} medHSV={med}')
            self.get_logger().info(
                f'diag {label}: mask_px={int(np.count_nonzero(mask))} blobs={len(cnts)} | ' + ' | '.join(parts))

    def _on_snapshot(self, _msg: String):
        if self._latest_bgr is not None:
            cv2.imwrite('/tmp/nirva_frame.png', self._latest_bgr)
            self.get_logger().info('saved /tmp/nirva_frame.png')
            self._diagnose(self._latest_bgr)

    def _status(self):
        age = time.time() - self._scan_wall if self.scan is not None else float('inf')
        self.get_logger().info(
            f'status: frames={self._frames} scan_age={age:.1f}s tracks={list(self.tracks.keys())}')

    def _watchdog(self):
        if self._frames == 0:
            self.get_logger().warn('No camera frames yet - check the image topic and that the robot has a camera.')
        if self.scan is None:
            self.get_logger().warn('No lidar scan yet.')

    # ----------------------------------------------------------- geometry

    def _intrinsics(self, img_w: int):
        if self.fx is not None:
            return self.fx, self.cx
        fx = (img_w / 2.0) / math.tan(self.fallback_hfov / 2.0)
        return fx, img_w / 2.0

    def _scan_range(self, bearing: float, half_width: float):
        """Median valid lidar range within +/-half_width of bearing (rad, base frame)."""
        scan = self.scan
        if scan is None or time.time() - self._scan_wall > 3.0:
            return None
        n = len(scan.ranges)
        inc = scan.angle_increment
        if n == 0 or inc == 0.0:
            return None
        span = scan.angle_max - scan.angle_min

        def idx(a):
            a = (a - scan.angle_min) % (2.0 * math.pi)
            if a > span + inc:
                return None
            return int(round(a / inc)) % n

        i0, i1 = idx(bearing - half_width), idx(bearing + half_width)
        if i0 is None or i1 is None:
            return None
        count = (i1 - i0) % n + 1
        if count < 3:
            i0, count = (i0 - 1) % n, 3
        vals = []
        for k in range(count):
            r = scan.ranges[(i0 + k) % n]
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                vals.append(r)
        return float(np.median(vals)) if vals else None

    def _to_map(self, x_b: float, y_b: float, stamp):
        """Base-frame point -> map frame (falls back to odom if map is not available)."""
        for frame in (self.target_frame, 'odom'):
            t = None
            for when in (Time.from_msg(stamp), Time()):
                try:
                    t = self.tf_buffer.lookup_transform(frame, self.base_frame, when)
                    break
                except TransformException:
                    continue
            if t is None:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            c, s = math.cos(yaw), math.sin(yaw)
            return (t.transform.translation.x + c * x_b - s * y_b,
                    t.transform.translation.y + s * x_b + c * y_b, frame)
        return None

    # ---------------------------------------------------------- detection

    def _offset_in_base(self, frame: str):
        """(x, y) of a sensor frame's origin in the base frame (static, cached). Assumes sensor yaw == base yaw."""
        if not frame:
            return (0.0, 0.0)
        if frame in self._offsets:
            return self._offsets[frame]
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
        except TransformException:
            return None
        o = (t.transform.translation.x, t.transform.translation.y)
        self._offsets[frame] = o
        return o

    def _locate(self, u: float, w_px: float, fx: float, cx: float, cam_frame: str):
        """
        Camera pixel + lidar -> box CENTRE in the base frame.
        The camera ray starts at the camera, the lidar range is measured from the lidar
        (a different point), so solve |C + d*u - L| = r for the distance d along the ray,
        then step half a box deeper to get from the visible face to the centre.
        Returns (x_b, y_b, d) or None.
        """
        C = self._offset_in_base(cam_frame)
        scan_frame = self.scan.header.frame_id if self.scan is not None else ''
        L = self._offset_in_base(scan_frame)
        if C is None or L is None:
            return None
        bearing = -math.atan2(u - cx, fx)              # +left, base-frame direction of the camera ray
        ux, uy = math.cos(bearing), math.sin(bearing)
        half = math.atan2(w_px * 0.25, fx)             # central half of the box width
        wx, wy = C[0] - L[0], C[1] - L[1]
        th_l = bearing
        d = None
        for _ in range(2):                             # refine the lidar bearing once
            r = self._scan_range(th_l, half)
            if r is None:
                return None
            b = wx * ux + wy * uy
            disc = b * b - (wx * wx + wy * wy) + r * r
            if disc < 0.0:
                return None
            d = -b + math.sqrt(disc)
            px, py = C[0] + d * ux, C[1] + d * uy
            th_l = math.atan2(py - L[1], px - L[0])
        d_c = d + self.center_off
        return C[0] + d_c * ux, C[1] + d_c * uy, d

    def _cam_height(self, frame: str) -> float:
        """Height of the camera origin above the ground (base_footprint is at ground level)."""
        key = ('h', frame)
        if key in self._offsets:
            return self._offsets[key]
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
        except TransformException:
            return 0.10
        self._offsets[key] = t.transform.translation.z
        return t.transform.translation.z

    def _lidar_clusters(self, half_fov: float):
        """Segment the lidar scan (inside the camera's horizontal FOV) into objects: [{'theta','r','width'}]."""
        scan = self.scan
        if scan is None or time.time() - self._scan_wall > 3.0:
            return []
        inc = scan.angle_increment
        pts = []
        for i, r in enumerate(scan.ranges):
            a = wrap_pi(scan.angle_min + i * inc)
            if abs(a) > half_fov or not math.isfinite(r) or r < max(scan.range_min, 0.05) or r > scan.range_max:
                continue
            pts.append((a, r, i))
        pts.sort()
        groups, cur = [], []
        for a, r, i in pts:
            if cur and (abs(r - cur[-1][1]) > 0.15 or a - cur[-1][0] > 3.0 * abs(inc)):
                groups.append(cur)
                cur = []
            cur.append((a, r, i))
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            if len(g) < 3:
                continue
            (a0, r0, i0), (a1, r1, i1) = g[0], g[-1]
            width = math.sqrt(max(0.0, r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * math.cos(a1 - a0)))

            def nb(i):      # range of the beam just outside the object (inf = no return = far away)
                v = scan.ranges[i % len(scan.ranges)]
                return v if (math.isfinite(v) and v >= max(scan.range_min, 0.05)) else float('inf')
            rl, rr = (nb(i0 - 1), nb(i1 + 1)) if inc > 0 else (float('inf'), float('inf'))
            out.append({'theta': 0.5 * (a0 + a1), 'r': float(np.median([r for _a, r, _i in g])),
                        'width': width, 'rl': rl, 'rr': rr})
        return out

    def _classify_roi(self, roi_hsv):
        """Colour label whose HSV mask covers >= 45 percent of the region, plus all fractions."""
        total = float(roi_hsv.shape[0] * roi_hsv.shape[1])
        best, best_f, fr = None, 0.0, {}
        for label, ranges in COLORS.items():
            m = None
            for lo, hi in ranges:
                mm = cv2.inRange(roi_hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                m = mm if m is None else cv2.bitwise_or(m, mm)
            f = float(np.count_nonzero(m)) / total
            fr[label[0]] = round(f, 2)
            if f > best_f:
                best, best_f = label, f
        return (best if best_f >= 0.45 else None), fr

    def _process_lidar_first(self, bgr, msg, debug, accepted):
        """
        Lidar finds box-sized objects (width 0.10-0.40 m); the camera only COLOURS them by looking at
        the rows/columns where that object must appear. A box in front of a wall of the same colour
        therefore still works, because the wall is never a candidate.
        """
        if self.scan is None:
            return
        img_h, img_w = bgr.shape[:2]
        fx, cx = self._intrinsics(img_w)
        cy = img_h / 2.0
        half_fov = math.atan2(img_w / 2.0, fx)
        C = self._offset_in_base(msg.header.frame_id)
        L = self._offset_in_base(self.scan.header.frame_id)
        if C is None or L is None:
            return
        zc = self._cam_height(msg.header.frame_id)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        unlabeled = []
        occluded = []
        for cl in sorted(self._lidar_clusters(half_fov + 0.1), key=lambda c: c['r']):
            th, r, wd = cl['theta'], cl['r'], cl['width']
            if not (self.cl_min_w <= wd <= self.cl_max_w) or r < 0.25:
                continue
            if min(cl['rl'], cl['rr']) < r - self.edge_gap:
                # something nearer sits right beside it (e.g. a strip of far wall seen between two
                # pillars): it is occluded background, not an object standing in the open
                occluded.append((round(r, 2), round(wd, 2)))
                continue
            bx, by = L[0] + r * math.cos(th), L[1] + r * math.sin(th)     # visible face, base frame
            d_c = bx - C[0]                                                 # forward distance from the camera
            if d_c < 0.2:
                continue
            bearing = math.atan2(by - C[1], d_c)
            if abs(bearing) > half_fov:
                continue
            u = cx - fx * math.tan(bearing)
            wpx = fx * wd / d_c
            v_top = cy + fx * (zc - self.box_h) / d_c
            v_bot = cy + fx * zc / d_c
            v0, v1 = v_top + 0.25 * (v_bot - v_top), v_bot - 0.25 * (v_bot - v_top)
            x0, x1 = int(max(0, u - 0.3 * wpx)), int(min(img_w - 1, u + 0.3 * wpx))
            y0, y1 = int(max(0, v0)), int(min(img_h - 1, v1))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            label, fr = self._classify_roi(hsv[y0:y1 + 1, x0:x1 + 1])
            note = f'{r:.2f}m w{wd:.2f}'
            if label is None:
                unlabeled.append((round(r, 2), round(wd, 2), fr))
                if debug is not None:
                    cv2.rectangle(debug, (x0, y0), (x1, y1), (160, 160, 160), 1)
                    cv2.putText(debug, note, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
                continue
            if label in accepted:
                continue
            # box CENTRE = visible face + half a box deeper, along the lidar ray
            cxb = L[0] + (r + self.center_off) * math.cos(th)
            cyb = L[1] + (r + self.center_off) * math.sin(th)
            p = self._to_map(cxb, cyb, msg.header.stamp)
            if p is None:
                note += ' no-tf'
            else:
                tr = self.tracks.setdefault(label, Track(self.history))
                tr.add(p[0], p[1], p[2])
                accepted.add(label)
                self.get_logger().info(f'{label}: centre ({p[0]:.2f}, {p[1]:.2f}) {p[2]} range {r:.2f} m',
                                       throttle_duration_sec=2.0)
            if debug is not None:
                col = DRAW.get(label, (255, 255, 255))
                cv2.rectangle(debug, (x0, y0), (x1, y1), col, 2)
                cv2.putText(debug, f'{label} {note}', (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if occluded:
            self.get_logger().info(f'box-sized lidar strips ignored (nearer object right beside them) (range, width): {occluded}',
                                   throttle_duration_sec=4.0)
        if unlabeled:
            self.get_logger().info(f'box-sized lidar objects with no colour match (range, width, r/g/b share): {unlabeled}',
                                   throttle_duration_sec=4.0)

    def _detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        found = []
        for label, ranges in COLORS.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            # Consider several blobs per colour (a big same-coloured wall must not hide a small box).
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    break
                bx, by, bw, bh = cv2.boundingRect(c)
                if area / float(bw * bh) < 0.55:        # box faces are solid blobs
                    continue
                if not 0.4 <= bw / float(bh) <= 2.5:    # and roughly square-ish
                    continue
                found.append((label, bx, by, bw, bh))
        return found

    def _to_bgr(self, msg: Image):
        """Image msg -> BGR array without cv_bridge (its cv2_to_imgmsg breaks with some OpenCV/NumPy builds)."""
        enc = msg.encoding
        chans = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}.get(enc)
        if chans is None:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # fallback
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, :msg.width * chans].reshape(msg.height, msg.width, chans)
        if enc == 'bgr8':
            return arr.copy()
        if enc == 'rgb8':
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if enc == 'bgra8':
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if enc == 'rgba8':
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    def _on_image(self, msg: Image):
        now = time.time()
        if now - self._last_proc < self.period:
            return
        self._last_proc = now
        try:
            bgr = self._to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f'image decode failed: {e}', throttle_duration_sec=5.0)
            return
        self._frames += 1
        self._latest_bgr = bgr
        img_h, img_w = bgr.shape[:2]
        fx, cx = self._intrinsics(img_w)
        debug = bgr.copy() if self.publish_debug else None

        if self.scan is None and not self._warned_no_scan:
            self.get_logger().warn('No lidar scan yet - cannot range objects.')
            self._warned_no_scan = True

        accepted = set()
        if self.mode == 'lidar':
            self._process_lidar_first(bgr, msg, debug, accepted)
        for label, x, y, w, h in (self._detect(bgr) if self.mode == 'contour' else []):
            if label in accepted:        # one box per colour per frame: first candidate that passes all checks
                continue
            note = ''
            if x <= 1 or x + w >= img_w - 1:
                note = 'edge'
            else:
                u = x + w / 2.0
                loc = self._locate(u, w, fx, cx, msg.header.frame_id)
                if loc is None:
                    note = 'no-range'
                else:
                    xb, yb, d = loc
                    d_lo = fx * self.box_w / w                 # face-on
                    d_hi = fx * self.box_w * math.sqrt(2) / w  # corner-on
                    if d < 0.7 * d_lo or d > 1.3 * d_hi:
                        note = f'range-mismatch r={d:.2f} est={d_lo:.2f}-{d_hi:.2f}'
                    else:
                        p = self._to_map(xb, yb, msg.header.stamp)
                        if p is None:
                            note = 'no-tf'
                        else:
                            tr = self.tracks.setdefault(label, Track(self.history))
                            tr.add(p[0], p[1], p[2])
                            accepted.add(label)
                            note = f'{d:.2f}m'
                            self.get_logger().info(
                                f'{label}: centre ({p[0]:.2f}, {p[1]:.2f}) {p[2]} range {d:.2f} m',
                                throttle_duration_sec=2.0)
            if label not in accepted and note:
                hm = np.median(cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)
                self.get_logger().info(
                    f'{label} candidate rejected: {note} (blob {w}x{h}px at x={x},y={y}, '
                    f'median HSV {hm[0]:.0f},{hm[1]:.0f},{hm[2]:.0f})', throttle_duration_sec=3.0)
            if debug is not None:
                col = DRAW.get(label, (255, 255, 255))
                cv2.rectangle(debug, (x, y), (x + w, y + h), col, 2)
                cv2.putText(debug, f'{label} {note}', (x, max(10, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if debug is not None:
            out = Image()
            out.header = msg.header
            out.height, out.width = debug.shape[:2]
            out.encoding = 'bgr8'
            out.is_bigendian = 0
            out.step = out.width * 3
            out.data = np.ascontiguousarray(debug).tobytes()
            self.debug_pub.publish(out)

    # -------------------------------------------------------------- output

    def _publish_model(self):
        objs = []
        now = time.time()
        for label, tr in self.tracks.items():
            if not tr.obs:
                continue
            x, y, spread, n, conf = tr.summary()
            objs.append({'label': label, 'x': round(x, 3), 'y': round(y, 3), 'frame': tr.frame,
                         'conf': round(conf, 2), 'n_recent': n, 'n_total': tr.total,
                         'spread_m': round(spread, 3), 'age_s': round(now - tr.last_seen, 1)})
        msg = String()
        msg.data = json.dumps({'objects': objs})
        self.model_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()