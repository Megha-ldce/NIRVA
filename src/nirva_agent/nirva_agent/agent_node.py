#!/usr/bin/env python3
"""
agent_node.py - the robot's "brain". Accepts ANY natural language prompt,
asks Nemotron to pick a tool, executes it, observes, and loops.

Tools: navigate (named places/coords via Nav2), drive (closed-loop rotate /
straight / circle driven by real TF pose, not timers), describe_surroundings
+ detect_object (real camera + vision model), pick_place (simulated, no arm
on this platform), replan, give_up, task_complete.

Key properties of this version
  * Pose comes from TF (map -> base_footprint, falling back to odom), NOT
    dead reckoning. Works unchanged once slam_toolbox builds the map live.
  * Every drive skill is closed-loop: it runs until the MEASURED yaw/distance
    reaches the target, and reports what was actually achieved. A skill can
    no longer say "success" for a motion that was cut short.
  * task_complete is verified in code: it is rejected if the last physical
    action failed.
  * Sensor subscriptions use sensor-data QoS (best-effort) so they actually
    match Gazebo/bridge publishers.
  * Lidar front-clearance uses real beam angles (+/-30 deg) and fails CLOSED
    when there is no fresh scan (toggle REQUIRE_LIDAR below).
  * /nirva/stop aborts the current motion immediately.
"""

import base64
import json
import math
import re
import time

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # older rclpy
    class ExternalShutdownException(Exception):
        pass
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from nirva_agent.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL,
    POSITIONS, SAFE_BOUNDS, MAX_ITERATIONS, CAMERA_TOPIC,
)
from nirva_agent.model_client import ModelClient


# ---------------------------------------------------------------------------
# Motion constants
# ---------------------------------------------------------------------------
DT = 0.05                       # control period (20 Hz)
ROT_MAX = 0.5                   # rad/s
ROT_MIN = 0.12                  # rad/s, so the final approach never stalls
LIN_SPEED = 0.15                # m/s
CIRCLE_LIN_SPEED = 0.1          # m/s
YAW_TOL = math.radians(1.0)
DIST_TOL = 0.01                 # m
SAFETY_STOP_DISTANCE_M = 0.25   # stop if something is closer than this, dead ahead
FRONT_ARC_HALF_DEG = 30.0       # +/- this many degrees counts as "ahead"
REQUIRE_LIDAR = True            # True: refuse to drive forward without a fresh /scan
MAX_STRAIGHT_M = 5.0
MAX_ROTATE_DEG = 1080.0
MAX_ARC_DEG = 720.0
NAV_TIMEOUT_S = 60.0
NAV_ARRIVAL_TOLERANCE_M = 0.6   # Nav2 said "success" but we are further than this -> report error
NOOP_ROTATE_DEG = 1.5           # rotations smaller than this are reported as "already there"
STALL_LIMIT = 4                 # consecutive navigate/drive calls with no pose change -> give up
STALL_MOVE_M = 0.02
STALL_TURN_DEG = 1.5

# Tools that physically change the world; used to verify task_complete.
ACTION_TOOLS = ('navigate', 'drive', 'pick_place', 'find_object', 'explore', 'push_object_to', 'verify_relation')

# Perception / exploration
KNOWN_LABELS = ('red_box', 'green_box', 'blue_box')
FRONTIER_MIN_CELLS = 6            # smallest frontier worth visiting (cells)
FRONTIER_CLEARANCE_M = 0.25       # viewpoints must be this far from occupied cells (robot radius ~0.22)
FRONTIER_VIEWPOINT_RADIUS_M = 1.0 # a viewpoint may be up to this far from its frontier
FRONTIER_BLACKLIST_RADIUS_M = 0.6
MAX_EXPLORE_GOALS = 25
COVERAGE_SPACING_M = 0.5          # fallback search: candidate stops on a grid this fine
COVERAGE_MIN_SEP_M = 0.9          # ...but never closer than this to a place already scanned
COVERAGE_CLEARANCE_M = 0.30
LOCATE_CANDIDATE_CONF = 0.3       # perception confidence that makes the robot STOP and look closer
LOCATE_CONFIRM_CONF = 0.6         # confidence required (while standing still) to accept a location
SEEN_CONF = 0.3                   # min confidence for a box already located to be used by push / verify
# Pushing a box next to another box
PUSH_SPEED = 0.10                 # m/s while in contact
STAGE_DIST_M = 0.55               # robot centre this far behind the box before the push
CONTACT_DIST_M = 0.26             # robot centre to box centre while pushing (box half 0.10 + waffle_pi front ~0.15; measured: contact at ~0.25)
BACKOFF_M = 0.45
GAP_TARGET_M = 0.40               # "next to" = box centres this far apart ...
GAP_MIN_M = 0.30                  # ... anywhere in this window counts
GAP_MAX_M = 0.50
STANDOFF_M = 0.7                  # distance from a box when approaching it

COMPASS = {
    'east': 0.0, 'northeast': 45.0, 'north': 90.0, 'northwest': 135.0,
    'west': 180.0, 'southwest': -135.0, 'south': -90.0, 'southeast': -45.0,
}

ERR_TEXT = {
    'obstacle': f'obstacle within {SAFETY_STOP_DISTANCE_M} m directly ahead - stopped early for safety',
    'no_lidar': 'no fresh /scan data - refusing to drive forward blind',
    'timeout': 'motion timed out before reaching the target',
    'aborted': 'aborted by a /nirva/stop request',
    'tf_lost': 'robot pose (TF) unavailable - cannot run closed-loop motion',
}


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def num(v, default=None):
    """Tolerant float conversion (LLMs sometimes send numbers as strings)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": (
                "Drive using Nav2 path planning (it plans AROUND obstacles). target is a known "
                "named location, an 'x,y' coordinate, or 'ahead:<metres>' = a point that many "
                "metres in front of the robot's current facing, or 'object:<label>' (e.g. 'object:red_box') = "
                "a safe standoff pose facing a box already in WORLD STATE objects (computed in code)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Named location or 'x,y' string."},
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drive",
            "description": (
                "Move the robot. You give MEANINGFUL quantities (degrees, "
                "metres); control is closed-loop in code using the real robot "
                "pose, and the result reports what was actually achieved. "
                "Pick ONE mode per call:\n"
                "- mode='rotate': spin in place. EITHER 'degrees' (relative "
                "turn) with direction='left' (counterclockwise) or 'right' "
                "(clockwise), OR 'to_heading_degrees' (absolute compass "
                "heading, 0=+x/east, 90=+y/north; the shortest turn is "
                "computed in code).\n"
                "- mode='straight': go forward 'distance_m' (max 5) along "
                "the CURRENT facing. Does NOT need a rotate first.\n"
                "- mode='circle': drive a circle/arc of 'radius_m' for "
                "'degrees' of arc (360=full circle, 180=half). "
                "direction='left' or 'right'. Keep radius_m <=0.6."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["rotate", "straight", "circle"]},
                    "degrees": {"type": "number", "description": "relative turn (rotate) or arc length (circle)"},
                    "to_heading_degrees": {"type": "number", "description": "rotate only: absolute compass heading to face"},
                    "to_compass": {"type": "string", "enum": list(COMPASS.keys()),
                                   "description": "rotate only: face this compass direction (east=0 deg, north=90, west=180, south=-90)"},
                    "distance_m": {"type": "number", "description": "for straight mode"},
                    "radius_m": {"type": "number", "description": "for circle mode, default 0.4 if unspecified"},
                    "direction": {"type": "string", "enum": ["left", "right"], "description": "for rotate (relative) / circle"},
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_surroundings",
            "description": (
                "Ask what the robot currently sees, in general terms, using the "
                "live camera + a vision model. To locate a box use find_object instead."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_object",
            "description": (
                "Locate a coloured box (red_box, green_box or blue_box) anywhere in the unknown "
                "environment. If it is not already in WORLD STATE objects, the robot explores "
                "unexplored space (building the map as it goes) and scans around until the box is "
                "seen. Returns its map x,y. Use this instead of driving around to look."
            ),
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string", "description": "e.g. 'red_box' or 'red box'."}},
                "required": ["label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore",
            "description": (
                "Explore unexplored space (frontier exploration) for up to max_goals goals, "
                "building the map. Use for requests like 'explore the room' or 'map the area'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"max_goals": {"type": "number", "description": "default 3, max 10"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_object_to",
            "description": (
                "Push one box with the robot's body until it is NEXT TO another box (centres "
                "0.30-0.50 m apart). It locates the boxes first if needed. It stages "
                "behind the box, pushes in a straight line, backs off, looks at both boxes from the "
                "side and MEASURES the gap; it retries once if the gap is still too big. Success "
                "means the measured gap is in range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "the box to push, e.g. 'red_box'"},
                    "target_label": {"type": "string", "description": "the box it must end up next to, e.g. 'green_box'"},
                    "gap_m": {"type": "number", "description": "desired centre-to-centre gap, default 0.40"},
                },
                "required": ["label", "target_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_relation",
            "description": (
                "Look at two located boxes from the side and measure how far apart their centres are. "
                "Succeeds if they are next to each other (0.30-0.50 m)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"label_a": {"type": "string"}, "label_b": {"type": "string"}},
                "required": ["label_a", "label_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replan",
            "description": "Signal that the previous step failed and a new approach is needed.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": "Call if the request genuinely cannot be done, INSTEAD of repeating replan with the same reason.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Call when the goal is fully achieved. Ends the loop. The summary "
                "may only state facts that appear in HISTORY. Rejected if the "
                "last physical action failed."
            ),
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are the reasoning module for an autonomous mobile \
robot. You receive an operator's natural language request plus the \
current world-state and action history, and must choose exactly ONE next \
tool to call.

HISTORY contains only actions taken for the CURRENT request (it is cleared \
for every new request). Every entry's result is measured by the robot, so \
trust it: 'turned_degrees' / 'travelled_m' are what really happened.

WORLD STATE fields: robot_pose is the robot's real position in metres. \
compass_heading_degrees is the direction it currently faces (0=+x/east, \
90=+y/north). It is INFORMATION ONLY - it is NOT an amount to rotate by.

Known named locations: {json.dumps(list(POSITIONS.keys()))}.
Use 'navigate' only for a known named location, or an explicit 'x,y' the \
operator gave. For everything else involving motion, use 'drive' - you \
only pick the mode and the quantities; control is computed in code.

- "rotate X degrees" / "spin around" -> drive(mode='rotate', degrees=X,
  direction='left' or 'right' - 'left' if not specified)
- "turn left/right X degrees" -> drive(mode='rotate', degrees=X,
  direction per the request). RELATIVE to current facing.
- "face / point / head to N degrees" or "set heading N" WITHOUT left/right
  = ABSOLUTE compass heading -> drive(mode='rotate', to_heading_degrees=N).
  ("face N degrees left/right" is a RELATIVE turn instead.) 'degrees' is
  always a POSITIVE number; use 'direction' for the side.
- COMPASS: "face / head / point east, north, west, south (or northeast,
  ...)" -> drive(mode='rotate', to_compass='east') (resp. 'north', 'west',
  ...). EAST IS NOT WEST: east=0 degrees, north=90, west=180, south=-90.
  Never work out the turn amount yourself.
- "turn around" / "u-turn" / "about face" -> rotate 180 degrees.
- "go N metres" / "go straight N m" -> ONE call: drive(mode='straight',
  distance_m=N). It already goes along the current facing. Do NOT rotate
  first.
- "go N metres at HEADING degrees": TWO calls across TWO turns - first
  drive(mode='rotate', to_heading_degrees=HEADING), THEN once history shows
  that succeeded, drive(mode='straight', distance_m=N).
- If a rotate result says no rotation was needed, you are ALREADY facing
  that way - move on. NEVER repeat tiny corrective rotations.
- If drive(mode='straight') fails with an obstacle error, the straight
  path is BLOCKED. Do not try to fix heading. Call replan ONCE, then
  navigate(target='ahead:<remaining_m>') using the remaining_m from the
  result - Nav2 plans around obstacles. If that fails too, give_up.
- "circle" / "loop" / "go around" -> drive(mode='circle', radius_m=0.4
  unless specified, degrees=360, direction='left' or 'right'). Keep
  radius_m <=0.6 - the room is small. ARC of X degrees -> same with
  degrees=X. If a circle returns an obstacle error: do NOT navigate back
  and retry the same radius - retry from the CURRENT position with half the
  radius, or give_up if it still fails.
- ANY REGULAR POLYGON (N sides, side length L): repeat, ONE STEP PER TURN,
  alternating (a) drive(mode='rotate', degrees=360/N, direction='left')
  and (b) drive(mode='straight', distance_m=L). 2*N calls in total. Count
  completed sides from HISTORY; call task_complete only after the Nth
  straight segment succeeds.
- "draw / trace / make a square (or any shape)" is a MOTION request: use
  the polygon recipe above. Never call describe_surroundings or
  detect_object for motion-only requests - they are only for requests to
  look, see or find something. If a camera tool errors, it is not needed
  for motion: carry on, or give_up if the request truly needs vision.

SIMPLE BOX REQUESTS:
- "find the X box": find_object(X), then task_complete. Do nothing else.
- "go to / approach the X box": find_object(X) if needed, navigate(target='object:X'),
  then task_complete.
- "put / place / push X next to Y" (X, Y boxes): call push_object_to(X, Y) DIRECTLY,
  then task_complete. It locates missing boxes, drives, pushes and MEASURES the result
  itself - do not call find_object or navigate first. If it fails, read its message and
  follow its 'next' hint (e.g. push the other box instead) or give_up. Never call replan
  twice in a row.
- "is X next to Y?" / "check the gap": verify_relation(X, Y), then task_complete with the
  measured answer in the summary - YES or NO. A NO is a valid answer: never give_up or
  replan after a question was answered. Only change the world if the user asked you to.

PERCEPTION / MAPPING (the map is built live; all coordinates are map-frame metres):
- WORLD STATE 'objects' lists boxes the robot has already located (label, x, y).
  Known labels: red_box, green_box, blue_box.
- To locate a box call find_object(label). It explores unknown space by
  itself until the box is seen. Do NOT hunt for boxes with drive/navigate
  and do not describe the scene to locate boxes.
- "go to / approach the X box": find_object(X) if it is not already in
  objects, then navigate(target='object:X').
- To explore with no target: explore(max_goals=3).
- If find_object reports the search finished or timed out, call give_up.
  Never repeat the same failed search.

COMPLETION RULES:
- Results within 3 degrees or 0.05 m of the request count as achieved.
  Never chase tiny corrections.
- replan only writes a note - it moves nothing. After replan your NEXT
  call must be a DIFFERENT physical action, or give_up. Never call replan
  twice in a row.
- For a SIMPLE single-action request ("turn 90 degrees left", "go 2 \
metres", "rotate 360 degrees"): once HISTORY shows that action with \
status 'success', call task_complete IMMEDIATELY. Do not repeat it.
- If the last physical action has status 'error', you may NOT call \
task_complete. Call replan (once) with a new approach, or give_up.
- The task_complete summary must only restate what HISTORY shows. Never \
claim something that is not in HISTORY.

Before calling navigate, check history: if the immediately preceding entry \
was ALSO a successful navigate to the SAME target, do NOT call it again - \
you are already there, move on.

If a previous action failed, call replan ONCE with a new approach - if \
you're about to give the SAME reason as the last history entry, call \
give_up instead.

Do not deliberate at length - pick the next action in 1-2 short internal \
sentences. For a compound request, just do the FIRST step now; you'll be \
asked again after it completes."""

AUDIT_COMPLETION = True   # extra LLM check that EVERY part of the request is evidenced

VERDICT_TOOL = [{
    "type": "function",
    "function": {
        "name": "verdict",
        "description": "Judge whether the operator's request was fully achieved.",
        "parameters": {
            "type": "object",
            "properties": {
                "fulfilled": {"type": "boolean"},
                "missing": {"type": "string", "description": "If not fulfilled: which part is NOT evidenced."},
            },
            "required": ["fulfilled"],
        },
    },
}]

AUDIT_PROMPT = (
    "You are the completion auditor for a mobile robot. You get the operator's "
    "REQUEST, the robot's measured FINAL STATE and the HISTORY of executed "
    "actions. All results are measured by the robot - trust them, not the "
    "planner's claim. Decide whether EVERY part of the request is evidenced as "
    "done. RULES: (1) Tolerances: a rotation within 3 degrees, or a distance "
    "within 0.05 m (or 5 percent), counts as ACHIEVED - never reject for "
    "smaller errors. (2) Rotations are RELATIVE: judge 'turn/rotate/spin N "
    "degrees' by turned_degrees in HISTORY, NOT by the final compass heading. "
    "Only requests that name an absolute direction ('face east', 'heading 90') "
    "are judged by final compass heading (east=0, north=90, west=180, "
    "south=-90). (3) Judge distances by travelled_m. (4) Reject only when a "
    "part of the request is clearly missing or clearly wrong. (5) 'Next to' means box "
    "centres 0.30-0.50 m apart as MEASURED by push_object_to / verify_relation. (6) If the REQUEST is a QUESTION "
    "('is / are / how far ...?'), the request is to REPORT: a measured answer, YES or NO, "
    "fully satisfies it - fulfilled=true. Judge a command by its end state, a question by "
    "whether it was measured and answered. Call verdict."
)

VISION_SYSTEM_PROMPT = (
    "You are the vision module for a mobile robot. You are given one "
    "image from the robot's forward-facing camera. Answer concisely and "
    "factually about only what is visible in the image - do not guess "
    "about anything outside the frame."
)


class AgentNode(Node):
    def __init__(self):
        super().__init__('nirva_agent_node')

        self.declare_parameter('base_frame', 'base_footprint')
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.model = ModelClient(LLM_API_KEY, LLM_BASE_URL)
        self.vision_model = ModelClient(VLLM_API_KEY, VLLM_BASE_URL)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 ready.')

        # Real pose from TF (map->base, falls back to odom->base).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Abort any running motion the moment rclpy starts shutting down (Ctrl+C).
        try:
            rclpy.get_default_context().on_shutdown(lambda: setattr(self, '_abort_flag', True))
        except Exception:
            pass

        # Per-goal state
        self.history = []
        self._last_effect = None     # (tool_name, result) of last physical action
        self._carrying = None        # simulated gripper contents
        self._abort_flag = False
        self._push_trip = None
        self._goal = ''

        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Sensor data: BEST_EFFORT QoS so we match Gazebo/bridge publishers.
        # (A default RELIABLE subscriber silently receives nothing from a
        # best-effort publisher.)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, CAMERA_TOPIC, self._camera_callback, qos_profile_sensor_data)
        self._cam_timer = self.create_timer(10.0, self._camera_watchdog)

        # Simulation time for timeouts, taken from /scan header stamps (5 Hz). Do NOT subscribe to /clock
        # from Python: it runs at ~1 kHz in Gazebo and costs about one CPU core.
        self._sim_now = None
        self._rtf = None
        self._rtf_ref = None

        self.latest_scan = None
        self._scan_rx_wall = 0.0
        self.create_subscription(LaserScan, '/scan', self._scan_callback, qos_profile_sensor_data)

        # Live map (slam_toolbox publishes /map TRANSIENT_LOCAL) and NIRVA's world model.
        self.latest_map = None
        self._objects = {}
        self._wm_rx = 0.0
        self._frontier_blacklist = []
        self._visited_sweeps = []
        self._tried_candidates = {}
        self._ghosts = []
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, '/map', self._map_callback, map_qos)
        self.create_subscription(String, '/nirva/world_model', self._world_model_callback, 10)

        # Prompt callback is serialised against itself (a second prompt must
        # not run concurrently and corrupt history). The stop subscription
        # lives in the default group so it can fire WHILE a prompt runs.
        prompt_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, '/nirva/prompt', self.prompt_callback, 1,
                                 callback_group=prompt_cb_group)
        self.create_subscription(String, '/nirva/stop', self._stop_callback, 10)
        self.speech_pub = self.create_publisher(String, '/nirva/speech', 1)

        self.get_logger().info(
            f'Ready. Pose source: TF ({self.base_frame} in map, fallback odom).\n'
            'Publish a prompt to /nirva/prompt, e.g.:\n'
            '  ros2 topic pub --once /nirva/prompt std_msgs/String "data: \'go to start and circle\'"\n'
            'Emergency stop:\n'
            '  ros2 topic pub --once /nirva/stop std_msgs/String "data: stop"'
        )

    # -------------------------------------------------------------
    # Sensors / pose
    # -------------------------------------------------------------

    def _image_topics(self):
        # Only topics that have at least one PUBLISHER. Our own subscription
        # makes the topic name show up in the graph even when nothing publishes.
        try:
            return sorted(n for n, types in self.get_topic_names_and_types()
                          if 'sensor_msgs/msg/Image' in types
                          and len(self.get_publishers_info_by_topic(n)) > 0)
        except Exception:
            return []

    def _no_frame_msg(self) -> str:
        return (f'No camera frame on {CAMERA_TOPIC}. Image topics visible right now: '
                f'{self._image_topics() or "none"}. Check the camera plugin/bridge and CAMERA_TOPIC in config.')

    def _camera_watchdog(self):
        if self.latest_frame is None:
            self.get_logger().warn(self._no_frame_msg())
        self._cam_timer.cancel()

    def _camera_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'camera decode failed: {e}')

    def _scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        w = time.time()
        self._scan_rx_wall = w
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t > 0.0:
            self._sim_now = t
            if self._rtf_ref is None:
                self._rtf_ref = (w, t)
            elif w - self._rtf_ref[0] >= 5.0:
                self._rtf = max(0.0, (t - self._rtf_ref[1]) / (w - self._rtf_ref[0]))
                self._rtf_ref = (w, t)

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def _world_model_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        now = time.time()
        objs = {}
        for o in data.get('objects', []):
            if 'label' in o and 'x' in o and 'y' in o:
                o['_rx'] = now
                objs[o['label']] = o
        self._objects = objs
        self._wm_rx = now

    @staticmethod
    def _norm_label(label):
        l = str(label).lower().replace('-', '_').replace(' ', '_')
        for c in ('red', 'green', 'blue'):
            if c in l:
                return f'{c}_box'
        return None

    def _get_object(self, label, min_conf=0.6, max_age_s=None):
        """Located object as {'x','y','conf','age_s'} or None. max_age_s=None -> remembered positions count."""
        o = self._objects.get(label)
        if not o or o.get('conf', 0.0) < min_conf:
            return None
        age = float(o.get('age_s', 0.0)) + (time.time() - o['_rx'])
        if max_age_s is not None and age > max_age_s:
            return None
        return {'x': float(o['x']), 'y': float(o['y']), 'conf': o.get('conf'), 'age_s': round(age, 1)}

    def _objects_for_prompt(self):
        out = []
        for label, o in self._objects.items():
            age = float(o.get('age_s', 0.0)) + (time.time() - o['_rx'])
            out.append({'label': label, 'x': round(float(o['x']), 2), 'y': round(float(o['y']), 2),
                        'conf': o.get('conf'), 'seen_s_ago': round(age, 0)})
        return out

    def _now(self) -> float:
        """Sim time in seconds from the latest /scan stamp (wall time until the first scan)."""
        return self._sim_now if self._sim_now is not None else time.time()

    def _rtf_text(self) -> str:
        return 'rtf=?' if self._rtf is None else f'rtf={self._rtf:.2f}'

    def _stop_callback(self, _msg: String):
        self._abort_flag = True
        self._publish_vel(0.0, 0.0)
        self.get_logger().warn('STOP requested - aborting current motion.')

    def get_pose(self, frame: str = None):
        """
        Latest robot pose from TF as {'x','y','yaw','frame'}, or None.
        frame=None tries 'map' first, then 'odom'. Pass an explicit frame to
        stay consistent inside one motion.
        """
        frames = [frame] if frame else ['map', 'odom']
        for f in frames:
            try:
                t = self.tf_buffer.lookup_transform(f, self.base_frame, rclpy.time.Time())
            except TransformException:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return {'x': t.transform.translation.x,
                    'y': t.transform.translation.y,
                    'yaw': yaw, 'frame': f}
        return None

    def _wait_for_pose(self, timeout_s: float = 3.0, frame: str = None):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            p = self.get_pose(frame)
            if p is not None:
                return p
            time.sleep(0.1)
        return None

    def _pose_summary(self):
        p = self.get_pose()
        if p is None:
            return None, None
        return ({'x': round(p['x'], 3), 'y': round(p['y'], 3), 'frame': p['frame']},
                round(math.degrees(wrap_pi(p['yaw'])), 1))

    def _front_clearance_m(self):
        """
        Minimum lidar range within +/-FRONT_ARC_HALF_DEG of straight ahead,
        using the real beam angles. float('inf') = clear. None = no fresh scan.
        """
        scan = self.latest_scan
        if scan is None or time.time() - self._scan_rx_wall > 1.0:
            return None
        half = math.radians(FRONT_ARC_HALF_DEG)
        best = float('inf')
        for i, r in enumerate(scan.ranges):
            a = wrap_pi(scan.angle_min + i * scan.angle_increment)
            if abs(a) > half:
                continue
            if not math.isfinite(r) or r < max(scan.range_min, 0.01) or r > scan.range_max:
                continue
            if r < best:
                best = r
        return best

    def _forward_check(self):
        """Returns an ERR_TEXT key if driving forward is unsafe, else None."""
        c = self._front_clearance_m()
        if c is None:
            return 'no_lidar' if REQUIRE_LIDAR else None
        if c < SAFETY_STOP_DISTANCE_M:
            return 'obstacle'
        return None

    def _encode_latest_frame(self):
        if self.latest_frame is None:
            return None
        ok, buf = cv2.imencode('.jpg', self.latest_frame)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    # -------------------------------------------------------------
    # Low-level velocity helpers
    # -------------------------------------------------------------

    def _publish_vel(self, lin: float, ang: float):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.twist.linear.x = float(lin)
        m.twist.angular.z = float(ang)
        self.cmd_vel_pub.publish(m)

    def _stop(self):
        for _ in range(3):
            self._publish_vel(0.0, 0.0)
            time.sleep(0.02)

    # -------------------------------------------------------------
    # Nav2 (named places / coordinates)
    # -------------------------------------------------------------

    def _send_nav_goal(self, x: float, y: float, yaw: float = 0.0, timeout_s: float = NAV_TIMEOUT_S,
                       stop_when=None) -> dict:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        # Zero stamp = "use the latest transform"; avoids wall/sim clock mismatch.
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        fb = {}
        def _on_feedback(m):
            f = m.feedback
            fb['dist'] = float(f.distance_remaining)
            fb['rec'] = int(f.number_of_recoveries)
        send_future = self.nav_client.send_goal_async(goal_msg, feedback_callback=_on_feedback)
        t0 = time.time()
        while not send_future.done():
            if time.time() - t0 > 5.0:
                return {'success': False, 'reason': 'goal send timed out (Nav2 not responding)'}
            time.sleep(0.05)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return {'success': False, 'reason': 'Nav2 rejected the goal (invalid pose or costmap issue)'}

        result_future = goal_handle.get_result_async()
        t0 = self._now()
        while not result_future.done():
            if self._abort_flag:
                goal_handle.cancel_goal_async()
                return {'success': False, 'reason': ERR_TEXT['aborted']}
            if stop_when is not None and stop_when():
                goal_handle.cancel_goal_async()
                return {'success': False, 'reason': 'stopped early: target found', 'stopped': True}
            if self._now() - t0 > timeout_s:
                goal_handle.cancel_goal_async()
                pn = self.get_pose('map')
                detail = (f"distance_remaining={fb.get('dist', -1):.2f} m, recoveries={fb.get('rec', '?')}, "
                          f"goal=({x:.2f},{y:.2f}), robot={'(%.2f,%.2f)' % (pn['x'], pn['y']) if pn else '?'}, {self._rtf_text()}")
                self.get_logger().warn(f'nav timeout after {timeout_s:.0f} sim-s: {detail}')
                return {
                    'success': False,
                    'reason': (
                        f'navigation timed out after {timeout_s:.0f} sim-s ({detail}) - likely blocked by an '
                        'obstacle, OR the target coordinate itself is inside/near an '
                        'obstacle - check RViz costmap'
                    ),
                }
            time.sleep(0.05)
        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return {'success': True}
        names = {GoalStatus.STATUS_CANCELED: 'CANCELED', GoalStatus.STATUS_ABORTED: 'ABORTED'}
        return {'success': False, 'reason': f'Nav2 goal ended with status {names.get(status, status)}'}

    def skill_navigate(self, target: str) -> dict:
        target = str(target).strip()
        lookup = {k.lower(): k for k in POSITIONS}
        goal_yaw = 0.0
        ahead = re.fullmatch(r'ahead\s*:\s*([0-9]*\.?[0-9]+)', target.lower())
        obj_m = re.fullmatch(r'object\s*:\s*([a-z_ ]+?)(?:\s*:\s*([0-9]*\.?[0-9]+))?', target.lower())
        if ahead:
            p = self.get_pose('map')
            if p is None:
                return {'status': 'error', 'message': 'map-frame pose unavailable, cannot compute "ahead" target'}
            d = float(ahead.group(1))
            x = p['x'] + d * math.cos(p['yaw'])
            y = p['y'] + d * math.sin(p['yaw'])
            goal_yaw = p['yaw']
        elif obj_m:
            okey = self._norm_label(obj_m.group(1))
            if okey is None:
                return {'status': 'error',
                        'message': f'unknown object label: {obj_m.group(1)} (known: {list(KNOWN_LABELS)})'}
            goal, why = self._standoff_goal(okey, num(obj_m.group(2), STANDOFF_M))
            if goal is None:
                return {'status': 'error', 'message': why}
            x, y, goal_yaw = goal
        elif target.lower() in lookup:
            key = lookup[target.lower()]
            x, y = POSITIONS[key]['x'], POSITIONS[key]['y']
        else:
            try:
                x_str, y_str = target.strip('()[] ').split(',')
                x, y = float(x_str), float(y_str)
            except ValueError:
                return {'status': 'error', 'message': f'unknown location or bad coordinate: {target}'}

        if SAFE_BOUNDS and not (ahead or obj_m) and not (SAFE_BOUNDS['x_min'] <= x <= SAFE_BOUNDS['x_max']
                and SAFE_BOUNDS['y_min'] <= y <= SAFE_BOUNDS['y_max']):
            return {'status': 'error', 'message': (f"target ({x:.2f},{y:.2f}) is outside the allowed area "
                                                   f"(x {SAFE_BOUNDS['x_min']}..{SAFE_BOUNDS['x_max']}, "
                                                   f"y {SAFE_BOUNDS['y_min']}..{SAFE_BOUNDS['y_max']}); "
                                                   "use a smaller distance or give_up")}

        self.get_logger().info(f'navigate -> ({x}, {y})')
        nav = self._send_nav_goal(x, y, goal_yaw)
        if not nav['success']:
            return {'status': 'error', 'target': target, 'message': nav['reason']}

        # Verify with real pose instead of trusting the action status alone.
        p = self.get_pose('map')
        if p is not None:
            d = math.hypot(p['x'] - x, p['y'] - y)
            if d > NAV_ARRIVAL_TOLERANCE_M:
                return {'status': 'error', 'target': target,
                        'message': f'Nav2 reported success but robot is {d:.2f} m from the target'}
            return {'status': 'success', 'target': target, 'distance_to_target_m': round(d, 2)}
        return {'status': 'success', 'target': target}

    # -------------------------------------------------------------
    # Closed-loop drive primitive
    # -------------------------------------------------------------

    def _yaw_loop(self, frame, target_rad, sign, lin_speed, ang_fn, timeout_s, stop_when=None):
        """
        Drive until the accumulated signed yaw reaches target_rad.
        Returns (turned_rad, last_yaw_or_None, err_key_or_None).
        """
        p = self.get_pose(frame)
        if p is None:
            return 0.0, None, 'tf_lost'
        last = p['yaw']
        turned = 0.0
        t0 = self._now()
        lost_since = None
        while True:
            if self._abort_flag:
                return turned, last, 'aborted'
            if stop_when is not None and stop_when():
                return turned, last, 'stopped'
            remaining = target_rad - turned
            if remaining <= YAW_TOL:
                return turned, last, None
            if self._now() - t0 > timeout_s:
                return turned, last, 'timeout'
            if lin_speed > 0.0:
                err = self._forward_check()
                if err:
                    return turned, last, err
            self._publish_vel(lin_speed, sign * ang_fn(remaining))
            time.sleep(DT)
            p = self.get_pose(frame)
            if p is None:
                lost_since = lost_since or time.time()
                if time.time() - lost_since > 1.0:
                    return turned, last, 'tf_lost'
                continue
            lost_since = None
            turned += sign * wrap_pi(p['yaw'] - last)
            last = p['yaw']

    def _yaw_motion(self, frame, target_rad, sign, lin_speed, ang_fn, timeout_s, stop_when=None):
        """_yaw_loop + stop + settle, so coasting is included in the measurement."""
        turned, last, err = self._yaw_loop(frame, target_rad, sign, lin_speed, ang_fn, timeout_s, stop_when)
        self._stop()
        time.sleep(0.3)
        if last is not None:
            p = self.get_pose(frame)
            if p is not None:
                turned += sign * wrap_pi(p['yaw'] - last)
        return turned, err

    def _straight_motion(self, frame, dist_m):
        """Closed-loop straight line with heading hold. Returns (travelled_m, err)."""
        p0 = self.get_pose(frame)
        if p0 is None:
            return 0.0, 'tf_lost'
        x0, y0, yaw0 = p0['x'], p0['y'], p0['yaw']
        timeout_s = dist_m / LIN_SPEED * 4.0 + 10.0
        t0 = self._now()
        lost_since = None
        err = None
        while True:
            if self._abort_flag:
                err = 'aborted'
                break
            p = self.get_pose(frame)
            if p is None:
                lost_since = lost_since or time.time()
                if time.time() - lost_since > 1.0:
                    err = 'tf_lost'
                    break
                time.sleep(DT)
                continue
            lost_since = None
            remaining = dist_m - math.hypot(p['x'] - x0, p['y'] - y0)
            if remaining <= DIST_TOL:
                break
            if self._now() - t0 > timeout_s:
                err = 'timeout'
                break
            err = self._forward_check()
            if err:
                break
            lin = min(LIN_SPEED, max(0.04, 1.0 * remaining))
            ang = max(-0.3, min(0.3, 1.5 * wrap_pi(yaw0 - p['yaw'])))
            self._publish_vel(lin, ang)
            time.sleep(DT)

        self._stop()
        time.sleep(0.3)
        p = self.get_pose(frame)
        travelled = math.hypot(p['x'] - x0, p['y'] - y0) if p is not None else 0.0
        return travelled, err

    def skill_drive(self, mode: str, degrees=None, distance_m=None, radius_m=None,
                    direction: str = 'left', to_heading_degrees=None, to_compass=None) -> dict:
        """
        Closed-loop motion. The LLM supplies degrees / metres; code runs the
        loop on the measured TF pose and reports what ACTUALLY happened.
        """
        if to_compass is not None and to_heading_degrees is None:
            key = str(to_compass).strip().lower()
            if key not in COMPASS:
                return {'status': 'error', 'message': f"unknown compass direction '{to_compass}'"}
            to_heading_degrees = COMPASS[key]
        if direction not in ('left', 'right'):
            direction = 'left'
        sign = 1.0 if direction == 'left' else -1.0

        # Absolute heading needs the map frame; relative motions use odom,
        # which is smooth (no SLAM/localisation correction jumps mid-motion).
        absolute = (mode == 'rotate' and to_heading_degrees is not None)
        start = self._wait_for_pose(3.0, 'map' if absolute else 'odom') or self._wait_for_pose(1.0)
        if start is None:
            return {'status': 'error', 'message': ERR_TEXT['tf_lost']}
        frame = start['frame']

        achieved = {}
        err = None
        hint = None
        try:
            if mode == 'rotate':
                heading = num(to_heading_degrees)
                if heading is not None:
                    delta = wrap_pi(math.radians(heading) - start['yaw'])
                    sign = 1.0 if delta >= 0 else -1.0
                    target_rad = abs(delta)
                    requested = round(math.degrees(target_rad), 1)
                else:
                    deg = num(degrees, 90.0)
                    if abs(deg) > MAX_ROTATE_DEG:
                        return {'status': 'error', 'message': f'rotate limited to {MAX_ROTATE_DEG:.0f} degrees per call'}
                    target_rad = math.radians(abs(deg))
                    requested = round(abs(deg), 1)
                if target_rad < math.radians(NOOP_ROTATE_DEG):
                    pose, heading_deg = self._pose_summary()
                    return {'status': 'success',
                            'message': (f'already facing the requested direction (within {NOOP_ROTATE_DEG} degrees) '
                                        '- no rotation needed, do not repeat'),
                            'requested_degrees': requested, 'turned_degrees': 0.0,
                            'new_pose': pose, 'new_compass_heading_degrees': heading_deg}
                timeout_s = target_rad / ROT_MAX * 4.0 + 10.0
                turned, err = self._yaw_motion(
                    frame, target_rad, sign, 0.0,
                    lambda rem: max(ROT_MIN, min(ROT_MAX, 1.5 * rem)), timeout_s)
                achieved = {'requested_degrees': requested,
                            'turned_degrees': round(math.degrees(turned), 1)}

            elif mode == 'straight':
                dist = num(distance_m, 0.5)
                if dist <= 0:
                    return {'status': 'error', 'message': 'distance_m must be positive'}
                if dist > MAX_STRAIGHT_M:
                    return {'status': 'error',
                            'message': f'distance_m limited to {MAX_STRAIGHT_M} m per call - split it into several calls'}
                travelled, err = self._straight_motion(frame, dist)
                achieved = {'requested_m': round(dist, 2), 'travelled_m': round(travelled, 3),
                            'remaining_m': round(max(0.0, dist - travelled), 3)}
                if err == 'obstacle':
                    hint = ("straight path is blocked; call replan once, then "
                            "navigate(target='ahead:<remaining_m>') so Nav2 plans around it")

            elif mode == 'circle':
                radius = max(0.2, min(num(radius_m, 0.4), 0.6))
                deg = num(degrees, 360.0)
                if deg <= 0 or deg > MAX_ARC_DEG:
                    return {'status': 'error', 'message': f'arc degrees must be in (0, {MAX_ARC_DEG:.0f}]'}
                w = CIRCLE_LIN_SPEED / radius
                target_rad = math.radians(deg)
                timeout_s = target_rad / w * 3.0 + 10.0
                turned, err = self._yaw_motion(
                    frame, target_rad, sign, CIRCLE_LIN_SPEED, lambda _rem: w, timeout_s)
                achieved = {'radius_m': round(radius, 2), 'requested_degrees': round(deg, 1),
                            'turned_degrees': round(math.degrees(turned), 1)}
            else:
                return {'status': 'error',
                        'message': f"unknown drive mode '{mode}' - must be rotate, straight, or circle"}
        finally:
            self._stop()

        pose, heading_deg = self._pose_summary()
        out = {**achieved, 'new_pose': pose, 'new_compass_heading_degrees': heading_deg}
        if err:
            res = {'status': 'error',
                   'message': ERR_TEXT[err] + ' - did not complete the full requested motion', **out}
            if hint:
                res['hint'] = hint
            return res
        return {'status': 'success', **out}

    # -------------------------------------------------------------
    # Vision
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # Map helpers, frontier exploration, find_object
    # -------------------------------------------------------------

    def _map_grid(self):
        m = self.latest_map
        if m is None or m.info.width == 0 or m.info.height == 0:
            return None
        data = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        return data, m.info.resolution, m.info.origin.position.x, m.info.origin.position.y

    @staticmethod
    def _cell_clear(grid, x, y, radius_m, allow_unknown=False) -> bool:
        """True if (x, y) is a free cell (or unknown, if allow_unknown) with no occupied cell within radius_m."""
        data, res, ox, oy = grid
        h, w = data.shape
        ix, iy = int((x - ox) / res), int((y - oy) / res)
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        v = data[iy, ix]
        if v >= 50 or (v != 0 and not (allow_unknown and v == -1)):
            return False
        r = max(1, int(radius_m / res))
        win = data[max(0, iy - r): iy + r + 1, max(0, ix - r): ix + r + 1]
        return not bool(np.any(win >= 50))

    def _frontier_goals(self):
        """
        Frontier clusters (free cells touching unknown space) -> one VIEWPOINT per cluster:
        the safe free cell (>= FRONTIER_CLEARANCE_M from any obstacle) nearest the cluster
        centre. Frontier cells themselves often hug a pillar or wall, so they are poor Nav2
        goals; a viewpoint a bit away from the obstacle is reachable and sees the frontier.
        Returns [(x, y, size_cells)] or None without a map.
        """
        grid = self._map_grid()
        if grid is None:
            return None
        data, res, ox, oy = grid
        free = data == 0
        unknown = (data == -1).astype(np.uint8)
        near_unknown = cv2.dilate(unknown, np.ones((3, 3), np.uint8)) > 0
        mask = (free & near_unknown).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

        occ = data >= 50
        dist_m = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3) * res
        sy, sx = np.nonzero(free & (dist_m >= FRONTIER_CLEARANCE_M))
        max_d2 = (FRONTIER_VIEWPOINT_RADIUS_M / res) ** 2

        goals, clusters = [], 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < FRONTIER_MIN_CELLS:
                continue
            clusters += 1
            if sx.size == 0:
                continue
            d2 = (sx - cents[i][0]) ** 2 + (sy - cents[i][1]) ** 2
            k = int(np.argmin(d2))
            if d2[k] > max_d2:
                continue
            goals.append((float(ox + (sx[k] + 0.5) * res), float(oy + (sy[k] + 0.5) * res), area))
        self._fr_cache = (grid, mask)
        self._fr_stats = {'clusters': clusters, 'w': data.shape[1], 'h': data.shape[0],
                          'free': int(free.sum()), 'unknown': int(unknown.sum()), 'occ': int(occ.sum())}
        return goals

    def _dump_frontier_png(self, goals, chosen):
        """/tmp/nirva_frontiers.png: grey=unknown, white=free, black=occupied, red=frontier, blue=viewpoints, green ring=chosen."""
        try:
            (data, res, ox, oy), mask = self._fr_cache
            img = np.full(data.shape + (3,), 127, np.uint8)
            img[data == 0] = (255, 255, 255)
            img[data >= 50] = (0, 0, 0)
            img[mask > 0] = (0, 0, 255)
            for gx, gy, _a in goals:
                cv2.circle(img, (int((gx - ox) / res), int((gy - oy) / res)), 2, (255, 0, 0), -1)
            if chosen:
                cv2.circle(img, (int((chosen[0] - ox) / res), int((chosen[1] - oy) / res)), 5, (0, 200, 0), 1)
            img = cv2.resize(cv2.flip(img, 0), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('/tmp/nirva_frontiers.png', img)
        except Exception:
            pass

    def _pick_frontier(self):
        """Nearest not-yet-tried frontier viewpoint -> ((x, y), None) or (None, reason)."""
        goals = self._frontier_goals()
        if goals is None:
            return None, 'no /map yet'
        p = self.get_pose('map')
        if p is None:
            return None, 'map-frame pose unavailable'
        best, usable = None, 0
        for gx, gy, area in goals:
            if any(math.hypot(gx - bx, gy - by) < FRONTIER_BLACKLIST_RADIUS_M
                   for bx, by in self._frontier_blacklist):
                continue
            d = math.hypot(gx - p['x'], gy - p['y'])
            if d < 0.4:
                continue
            usable += 1
            score = d - 0.01 * area
            if best is None or score < best[0]:
                best = (score, gx, gy)
        st = self._fr_stats
        self.get_logger().info(
            f"frontier scan: clusters={st['clusters']} viewpoints={len(goals)} usable={usable} | "
            f"map {st['w']}x{st['h']} free={st['free']} unknown={st['unknown']} occupied={st['occ']}")
        self._dump_frontier_png(goals, (best[1], best[2]) if best else None)
        if best is None:
            return None, 'no unexplored frontiers left'
        return (best[1], best[2]), None

    def _coverage_goal(self, min_sep=COVERAGE_MIN_SEP_M):
        """
        Fallback when no frontier is usable: the nearest safe free cell that is at least
        COVERAGE_MIN_SEP_M from every place already scanned. Makes the robot tour the known
        free space ('move here and there') so occluded objects get seen from new angles.
        Returns (x, y) or None.
        """
        grid = self._map_grid()
        p = self.get_pose('map')
        if grid is None or p is None:
            return None
        data, res, ox, oy = grid
        free = data == 0
        dist_m = cv2.distanceTransform((data < 50).astype(np.uint8), cv2.DIST_L2, 3) * res
        safe = free & (dist_m >= COVERAGE_CLEARANCE_M)
        step = max(1, int(COVERAGE_SPACING_M / res))
        iy, ix = np.nonzero(safe[::step, ::step])
        if ix.size == 0:
            return None
        wx = ox + (ix * step + 0.5) * res
        wy = oy + (iy * step + 0.5) * res
        keep = np.ones(wx.shape, dtype=bool)
        for vx, vy in self._visited_sweeps + [(p['x'], p['y'])]:
            keep &= np.hypot(wx - vx, wy - vy) >= min_sep
        n_sep = int(keep.sum())
        for bx, by in self._frontier_blacklist:
            keep &= np.hypot(wx - bx, wy - by) >= FRONTIER_BLACKLIST_RADIUS_M
        self.get_logger().info(
            f'coverage(min_sep={min_sep:.1f}): robot=({p["x"]:.2f},{p["y"]:.2f}) free={int(free.sum())} cells, '
            f'safe stops={ix.size}, after visited-sep={n_sep}, after blacklist={int(keep.sum())}, '
            f'visited={len(self._visited_sweeps)}, blacklisted={len(self._frontier_blacklist)}')
        if not keep.any():
            return None
        wx, wy = wx[keep], wy[keep]
        k = int(np.argmin(np.hypot(wx - p['x'], wy - p['y'])))
        return float(wx[k]), float(wy[k])

    def _next_search_goal(self):
        """Frontier viewpoint first, else coverage stop. -> ((x, y), kind, None) or (None, None, reason)."""
        goal, why = self._pick_frontier()
        if goal is not None:
            return goal, 'frontier', None
        goal = self._coverage_goal() or self._coverage_goal(min_sep=0.5)   # relaxed spacing before giving up
        if goal is not None:
            return goal, 'coverage', None
        return None, None, why

    def _standoff_goal(self, key, standoff_m, min_conf=0.6):
        """Pose `standoff_m` from a located box, facing it, on a clear approach side. -> ((x, y, yaw), None) or (None, why)."""
        obj = self._get_object(key, min_conf=min_conf)
        if obj is None:
            return None, f'{key} is not in the world model - call find_object first'
        p = self.get_pose('map')
        if p is None:
            return None, 'map-frame pose unavailable'
        ox, oy = obj['x'], obj['y']
        base = math.atan2(p['y'] - oy, p['x'] - ox)       # direction from the box towards the robot
        grid = self._map_grid()
        offsets = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
        # Known-free points first (any radius), then points in unmapped space: the live map only
        # shows what the lidar has seen, and Nav2 can plan through unknown cells.
        for allow_unknown in (False, True):
            for radius in (standoff_m, max(0.45, standoff_m - 0.15)):
                for off in offsets:
                    a = base + math.radians(off)
                    gx, gy = ox + radius * math.cos(a), oy + radius * math.sin(a)
                    if grid is None or self._cell_clear(grid, gx, gy, 0.30, allow_unknown):
                        return (gx, gy, math.atan2(oy - gy, ox - gx)), None
        return None, f'no clear approach point around {key} (obstacles within 0.3 m of every approach side)'

    def _confirm_object(self, key, dwell_s: float = 6.0):
        """
        Stand still until perception's estimate settles (conf >= LOCATE_CONFIRM_CONF).
        While the robot is turning or driving the camera/lidar samples are noisy, so perception
        reports low confidence; standing still flushes those samples out within a couple of seconds.
        """
        self._stop()
        t0 = time.time()
        while time.time() - t0 < dwell_s:
            if self._abort_flag:
                return None
            if time.time() - t0 >= 2.0:
                obj = self._get_object(key, min_conf=LOCATE_CONFIRM_CONF)
                if obj is not None:
                    return obj
            time.sleep(0.3)
        return self._get_object(key, min_conf=LOCATE_CONFIRM_CONF)

    def _candidate(self, key):
        """An unconfirmed sighting (conf >= LOCATE_CANDIDATE_CONF) that has not already been chased and rejected."""
        o = self._get_object(key, min_conf=LOCATE_CANDIDATE_CONF)
        if o is None:
            return None
        for tx, ty in self._tried_candidates.get(key, []):
            if math.hypot(o['x'] - tx, o['y'] - ty) < 0.4:
                return None
        return o

    def _chase_candidate(self, key):
        """A sighting that did not confirm from here: remember it, drive to a closer viewpoint, look again."""
        cand = self._get_object(key, min_conf=LOCATE_CANDIDATE_CONF)
        if cand is None:
            return None
        self._tried_candidates.setdefault(key, []).append((cand['x'], cand['y']))
        goal, why = self._standoff_goal(key, 1.0, min_conf=LOCATE_CANDIDATE_CONF)
        if goal is None:
            self.get_logger().warn(f'find_object({key}): unconfirmed sighting at ({cand["x"]:.2f},{cand["y"]:.2f}) '
                                   f'conf={cand["conf"]}, no viewpoint for a closer look ({why})')
            return None
        self.get_logger().info(f'find_object({key}): unconfirmed sighting at ({cand["x"]:.2f},{cand["y"]:.2f}) '
                               f'conf={cand["conf"]}; closer look from ({goal[0]:.2f},{goal[1]:.2f})')
        nav = self._send_nav_goal(goal[0], goal[1], goal[2], timeout_s=90.0)
        if not nav['success']:
            self.get_logger().warn(f'find_object({key}): closer look not reached: {nav.get("reason")}')
        return self._confirm_object(key)

    def _sweep_for(self, key):
        """Rotate up to 360 deg looking for `key`; stop as soon as it is located, then hold still to refine."""
        start = self._wait_for_pose(2.0, 'odom') or self._wait_for_pose(1.0)
        if start is None:
            return None
        here = self.get_pose('map')
        if here is not None:
            self._visited_sweeps.append((here['x'], here['y']))
        _turned, err = self._yaw_motion(
            start['frame'], 2.0 * math.pi, 1.0, 0.0,
            lambda rem: max(ROT_MIN, min(0.4, 1.5 * rem)), 90.0,
            stop_when=lambda: self._candidate(key) is not None)
        if err == 'stopped':
            return self._confirm_object(key) or self._chase_candidate(key)
        return self._get_object(key)

    def skill_find_object(self, label: str, timeout_s: float = 420.0) -> dict:
        key = self._norm_label(label)
        if key is None:
            return {'status': 'error', 'message': f'unknown object "{label}". Known labels: {list(KNOWN_LABELS)}'}
        if time.time() - self._wm_rx > 15.0:
            return {'status': 'error', 'message': 'no /nirva/world_model messages - perception_node is not running'}
        obj = self._get_object(key)
        if obj:
            return {'status': 'success', 'label': key, 'x': round(obj['x'], 2), 'y': round(obj['y'], 2),
                    'already_known': True,
                    'next': f"Already located - do NOT call find_object({key}) again. Do NOT navigate to it unless "
                            f"the user asked to go to / approach it; otherwise do the next part of the request or task_complete."}
        if self._map_grid() is None:
            return {'status': 'error', 'message': 'no /map yet - is slam_toolbox running?'}

        t0 = time.time()
        tried = 0
        found = self._sweep_for(key)
        while found is None:
            if self._abort_flag:
                return {'status': 'error', 'message': ERR_TEXT['aborted']}
            if time.time() - t0 > float(timeout_s):
                return {'status': 'error', 'message': f'timed out searching for {key} after {tried} frontiers'}
            if tried >= MAX_EXPLORE_GOALS:
                return {'status': 'error', 'message': f'gave up searching for {key} after {tried} frontiers'}
            goal, kind, why = self._next_search_goal()
            if goal is None:
                return {'status': 'error',
                        'message': f'search finished: no frontiers or unvisited areas left ({why}). '
                                   f'{key} not found after {tried} search stops'}
            tried += 1
            self._frontier_blacklist.append(goal)
            self.get_logger().info(f'find_object({key}): {kind} stop {tried} -> ({goal[0]:.2f}, {goal[1]:.2f})')
            nav = self._send_nav_goal(goal[0], goal[1], 0.0, timeout_s=90.0,
                                      stop_when=lambda: self._candidate(key) is not None)
            if not nav['success'] and not nav.get('stopped'):
                self.get_logger().warn(f'find_object({key}): stop ({goal[0]:.2f}, {goal[1]:.2f}) not reached: {nav.get("reason")}')
            if self._candidate(key) is not None:
                found = self._confirm_object(key) or self._chase_candidate(key)
                if found is not None:
                    break
            if nav.get('reason') == ERR_TEXT['aborted']:
                return {'status': 'error', 'message': ERR_TEXT['aborted']}
            if not nav['success']:
                p_now = self.get_pose('map')
                last = self._visited_sweeps[-1] if self._visited_sweeps else None
                if p_now is not None and last is not None and math.hypot(p_now['x'] - last[0], p_now['y'] - last[1]) < 0.5:
                    continue          # goal unreachable and we have not moved: do not rescan the same spot
            found = self._sweep_for(key)

        return {'status': 'success', 'label': key, 'x': round(found['x'], 2), 'y': round(found['y'], 2),
                'conf': found['conf'], 'search_stops': tried}

    def skill_explore(self, max_goals=3) -> dict:
        n = int(max(1, min(num(max_goals, 3), 10)))
        if self._map_grid() is None:
            return {'status': 'error', 'message': 'no /map yet - is slam_toolbox running?'}
        reached = 0
        for _ in range(n):
            if self._abort_flag:
                return {'status': 'error', 'message': ERR_TEXT['aborted'], 'goals_reached': reached}
            goal, _kind, why = self._next_search_goal()
            if goal is None:
                return {'status': 'success', 'goals_reached': reached, 'map_complete': True, 'note': why}
            self._frontier_blacklist.append(goal)
            nav = self._send_nav_goal(goal[0], goal[1], 0.0, timeout_s=90.0)
            if nav['success']:
                reached += 1
        if reached == 0:
            return {'status': 'error', 'message': 'could not reach any frontier', 'goals_reached': 0}
        return {'status': 'success', 'goals_reached': reached, 'requested': n}

    # -------------------------------------------------------------
    # Pushing a box next to another box, and measuring the result
    # -------------------------------------------------------------

    @staticmethod
    def _push_plan(A, B, gap):
        """Geometry for pushing box A (x, y) towards box B until the centres are `gap` apart."""
        dx, dy = A[0] - B[0], A[1] - B[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None
        v = (dx / dist, dy / dist)                              # from B towards A
        u = (-v[0], -v[1])                                      # push direction
        G = (B[0] + gap * v[0], B[1] + gap * v[1])              # where A's centre should end up
        S = (A[0] - u[0] * STAGE_DIST_M, A[1] - u[1] * STAGE_DIST_M)   # staging point behind A
        return {'dist': dist, 'u': u, 'G': G, 'S': S, 'yaw': math.atan2(u[1], u[0]), 'push_d': dist - gap}

    def _near_ghost(self, pt, r=0.40):
        """True if pt is within r of a place where a pushed box used to be (stale occupied cells in the SLAM map)."""
        return any(math.hypot(pt[0] - gx, pt[1] - gy) < r for gx, gy in self._ghosts)

    def _push_path_clear(self, plan, A):
        """Map check along the staging->goal line (robot and box swept area). -> (True, None) or (False, (x, y))."""
        grid = self._map_grid()
        if grid is None:
            return True, None
        (sx, sy), (gx, gy), u = plan['S'], plan['G'], plan['u']
        length = math.hypot(gx - sx, gy - sy)
        robot_len = max(0.0, length - CONTACT_DIST_M)           # the robot stops here; beyond it only the box moves
        for k in range(int(length / 0.05) + 2):
            t = min(length, k * 0.05)
            qx, qy = sx + u[0] * t, sy + u[1] * t
            if math.hypot(qx - A[0], qy - A[1]) < 0.36 or self._near_ghost((qx, qy)):
                continue                                        # the box's own map cells / where it used to be
            radius = 0.23 if t <= robot_len else 0.13           # robot body, then just the 0.2 m box (+margin)
            if not self._cell_clear(grid, qx, qy, radius, allow_unknown=True):
                return False, (qx, qy)
        return True, None

    def _sector_clearance(self, center, half, skip=0.0):
        """Closest lidar return with skip <= |angle - center| <= half (radians). None = no fresh scan."""
        scan = self.latest_scan
        if scan is None or time.time() - self._scan_rx_wall > 1.0:
            return None
        best = float('inf')
        for i, r in enumerate(scan.ranges):
            a = abs(wrap_pi(scan.angle_min + i * scan.angle_increment - center))
            if a > half or a < skip:
                continue
            if not math.isfinite(r) or r < max(scan.range_min, 0.01) or r > scan.range_max:
                continue
            best = min(best, r)
        return best

    def _sector_closest(self, center, half, skip=0.0):
        """(range, angle_deg relative to center) of the closest lidar return in the sector, or None if no fresh scan."""
        scan = self.latest_scan
        if scan is None or time.time() - self._scan_rx_wall > 1.0:
            return None
        best = (float('inf'), 0.0)
        for i, r in enumerate(scan.ranges):
            ang = wrap_pi(scan.angle_min + i * scan.angle_increment - center)
            if abs(ang) > half or abs(ang) < skip:
                continue
            if not math.isfinite(r) or r < max(scan.range_min, 0.01) or r > scan.range_max:
                continue
            if r < best[0]:
                best = (r, math.degrees(ang))
        return best

    def _push_check(self, forward: bool):
        """Safety while pushing. Forward: ignore the central cone (that is the box) and watch the flanks."""
        if forward:
            # The pushed box (even if rotated / off-centre) fills the front +-65 deg: ignore it, watch only the sides.
            near = self._sector_closest(0.0, math.radians(110), skip=math.radians(65))
            limit = 0.15
        else:
            near = self._sector_closest(math.pi, math.radians(30))
            limit = SAFETY_STOP_DISTANCE_M
        if near is None:
            return 'no_lidar' if REQUIRE_LIDAR else None
        if near[0] < limit:
            self._push_trip = (round(near[0], 2), round(near[1], 0), 'forward' if forward else 'backward')
            return 'obstacle'
        return None

    def _push_motion(self, frame, dist_m, speed, forward, timeout_s):
        """Closed-loop straight motion (forward or backward) with heading hold. -> (travelled_m, err_key)."""
        p0 = self.get_pose(frame)
        if p0 is None:
            return 0.0, 'tf_lost'
        x0, y0, yaw0 = p0['x'], p0['y'], p0['yaw']
        sgn = 1.0 if forward else -1.0
        t0 = self._now()
        lost_since, err = None, None
        while True:
            if self._abort_flag:
                err = 'aborted'
                break
            p = self.get_pose(frame)
            if p is None:
                lost_since = lost_since or time.time()
                if time.time() - lost_since > 1.0:
                    err = 'tf_lost'
                    break
                time.sleep(DT)
                continue
            lost_since = None
            if dist_m - math.hypot(p['x'] - x0, p['y'] - y0) <= DIST_TOL:
                break
            if self._now() - t0 > timeout_s:
                err = 'timeout'
                break
            err = self._push_check(forward)
            if err:
                break
            self._publish_vel(sgn * speed, max(-0.3, min(0.3, 1.5 * wrap_pi(yaw0 - p['yaw']))))
            time.sleep(DT)
        self._stop()
        time.sleep(0.3)
        p = self.get_pose(frame)
        return (math.hypot(p['x'] - x0, p['y'] - y0) if p is not None else 0.0), err

    def _goto_precise(self, S, yaw):
        """Closed-loop correction after Nav2 (its goal tolerance is ~0.25 m): face S, drive to it, face yaw. -> None or why."""
        for _ in range(2):
            p = self.get_pose('map')
            if p is None:
                return 'map pose unavailable'
            dx, dy = S[0] - p['x'], S[1] - p['y']
            d = math.hypot(dx, dy)
            if d <= 0.04:
                break
            r = self.skill_drive('rotate', to_heading_degrees=math.degrees(math.atan2(dy, dx)))
            if r['status'] != 'success':
                return r.get('message', 'rotate failed')
            r = self.skill_drive('straight', distance_m=d)
            if r['status'] != 'success':
                return r.get('message', 'drive failed')
        r = self.skill_drive('rotate', to_heading_degrees=math.degrees(yaw))
        return None if r['status'] == 'success' else r.get('message', 'rotate failed')

    def _los_clear(self, grid, P, Q, ignore_near_m=0.30):
        """True if the straight line P->Q crosses no occupied map cell (cells within ignore_near_m of Q are the box itself)."""
        length = math.hypot(Q[0] - P[0], Q[1] - P[1])
        for k in range(int(length / 0.05) + 1):
            t = k * 0.05
            qx, qy = P[0] + (Q[0] - P[0]) * t / length, P[1] + (Q[1] - P[1]) * t / length
            if math.hypot(qx - Q[0], qy - Q[1]) < ignore_near_m:
                break
            if not self._cell_clear(grid, qx, qy, 0.08, allow_unknown=True):
                return False
        return True

    def _vantage_pose(self, A, B):
        """
        A standing point ~1 m from the A-B midpoint, roughly side-on (so the boxes appear side by side),
        facing the midpoint, with a clear line of sight to BOTH boxes. Nearest valid one wins.
        """
        L = math.hypot(A['x'] - B['x'], A['y'] - B['y'])
        p = self.get_pose('map')
        if L < 1e-6 or p is None:
            return None
        mx, my = (A['x'] + B['x']) / 2.0, (A['y'] + B['y']) / 2.0
        axis = math.atan2(A['y'] - B['y'], A['x'] - B['x'])
        grid = self._map_grid()
        out = []                                                  # valid vantage points, preferred distance first, nearest first
        for dist in (1.0, 0.8, 1.2, 1.5):
            cands = []
            for off in (90, 75, 105, 60, 120, -90, -75, -105, -60, -120):
                ang = axis + math.radians(off)
                vx, vy = mx + dist * math.cos(ang), my + dist * math.sin(ang)
                if grid is not None and not (
                        self._cell_clear(grid, vx, vy, 0.30, allow_unknown=True)
                        and self._los_clear(grid, (vx, vy), (A['x'], A['y']))
                        and self._los_clear(grid, (vx, vy), (B['x'], B['y']))):
                    continue
                cands.append((math.hypot(vx - p['x'], vy - p['y']), vx, vy))
            for _t, vx, vy in sorted(cands):
                if all(math.hypot(vx - ox, vy - oy) > 0.6 for ox, oy, _yw in out):
                    out.append((vx, vy, math.atan2(my - vy, mx - vx)))
            if len(out) >= 3:
                break
        return out[:3] if out else None

    def _verify_pair(self, a_key, b_key):
        """Drive to a side vantage point, look at both boxes, measure the centre gap. -> dict."""
        A = self._get_object(a_key, min_conf=SEEN_CONF)
        B = self._get_object(b_key, min_conf=SEEN_CONF)
        if A is None or B is None:
            return {'ok': False, 'distance': None, 'fresh': False, 'message': 'a box is missing from the world model'}
        fresh = False
        spots = self._vantage_pose(A, B)
        if not spots:
            self.get_logger().warn(f'verify {a_key}/{b_key}: no valid side vantage point (map clearance / line of sight)')
        for spot in spots or []:
            self.get_logger().info(f'verify {a_key}/{b_key}: vantage ({spot[0]:.2f}, {spot[1]:.2f}), '
                                   f'A=({A["x"]:.2f},{A["y"]:.2f}) B=({B["x"]:.2f},{B["y"]:.2f})')
            nav = self._send_nav_goal(spot[0], spot[1], spot[2], timeout_s=120.0)
            if not nav['success']:
                self.get_logger().warn(f'verify: could not reach vantage point: {nav.get("reason")}')
                if self._abort_flag:
                    break
                continue
            time.sleep(4.0)
            A2 = self._get_object(a_key, min_conf=SEEN_CONF, max_age_s=8.0)
            B2 = self._get_object(b_key, min_conf=SEEN_CONF, max_age_s=8.0)
            if A2 is not None and B2 is not None:
                A, B, fresh = A2, B2, True
                break
            self.get_logger().warn(f'verify: at vantage but fresh view missing: '
                                   f'{a_key}={"ok" if A2 else "none"} {b_key}={"ok" if B2 else "none"} (age>8s or conf<{SEEN_CONF})')
        d = math.hypot(A['x'] - B['x'], A['y'] - B['y'])
        return {'ok': GAP_MIN_M <= d <= GAP_MAX_M, 'distance': d, 'fresh': fresh,
                'a': (round(A['x'], 2), round(A['y'], 2)), 'b': (round(B['x'], 2), round(B['y'], 2))}

    def _push_once(self, a_key, b_key, gap):
        """Stage behind box A, push it towards box B, back off. -> {'ok': bool, 'message': ..., 'already': bool}."""
        A = self._get_object(a_key, min_conf=SEEN_CONF)
        B = self._get_object(b_key, min_conf=SEEN_CONF)
        if A is None or B is None:
            return {'ok': False, 'message': 'a box is missing from the world model'}
        plan = None
        for it in range(3):                                       # re-stage if the close-up view moves the box estimate
            plan = self._push_plan((A['x'], A['y']), (B['x'], B['y']), gap)
            if plan is None:
                return {'ok': False, 'message': 'the two boxes are at the same position'}
            if GAP_MIN_M <= plan['dist'] <= GAP_MAX_M:
                return {'ok': True, 'already': True}
            if plan['dist'] < GAP_MIN_M:
                return {'ok': False, 'message': f"boxes are already {plan['dist']:.2f} m apart (too close); a box cannot be pulled back"}
            clear, where = self._push_path_clear(plan, (A['x'], A['y']))
            if not clear:
                return {'ok': False, 'message': f'push path blocked by an obstacle near ({where[0]:.2f}, {where[1]:.2f})'}
            S = plan['S']
            self.get_logger().info(
                f'push plan[{it}]: A=({A["x"]:.2f},{A["y"]:.2f}) B=({B["x"]:.2f},{B["y"]:.2f}) dist={plan["dist"]:.2f} '
                f'stage S=({S[0]:.2f},{S[1]:.2f}) goal G=({plan["G"][0]:.2f},{plan["G"][1]:.2f}) yaw={math.degrees(plan["yaw"]):.0f}')
            grid = self._map_grid()
            if grid is not None and not self._near_ghost(S) and not self._cell_clear(grid, S[0], S[1], 0.28, allow_unknown=True):
                return {'ok': False, 'message': 'no room to stand behind the box (obstacle near the staging point)'}
            if it == 0:
                nav = self._send_nav_goal(S[0], S[1], plan['yaw'], timeout_s=120.0)
                if not nav['success']:
                    p_now = self.get_pose('map')
                    self.get_logger().warn(f'push: staging nav failed, S=({S[0]:.2f},{S[1]:.2f}), robot at '
                                           f'{("(%.2f,%.2f)" % (p_now["x"], p_now["y"])) if p_now else "?"}')
                    near = p_now is not None and math.hypot(S[0] - p_now['x'], S[1] - p_now['y']) <= 1.5
                    if not (self._near_ghost(S) and near):
                        return {'ok': False, 'message': f"could not reach the staging point ({S[0]:.2f}, {S[1]:.2f}): {nav['reason']}"}
                    self.get_logger().info('push: staging point is on the stale map cells of the moved box; driving there closed-loop')
            why = self._goto_precise(S, plan['yaw'])
            if why:
                return {'ok': False, 'message': f'could not line up behind the box: {why}'}
            time.sleep(3.0)                                       # close, face-on view of the box
            A2 = self._get_object(a_key, min_conf=SEEN_CONF, max_age_s=8.0)
            if A2 is None:
                break
            shift = math.hypot(A2['x'] - A['x'], A2['y'] - A['y'])
            A = A2
            if shift <= 0.06:
                break

        pose = self.get_pose('map')
        start = self._wait_for_pose(2.0, 'odom') or self._wait_for_pose(1.0)
        if pose is None or start is None:
            return {'ok': False, 'message': 'pose unavailable'}
        u, G = plan['u'], plan['G']
        self._ghosts.append((A['x'], A['y']))
        r_end = (G[0] - u[0] * CONTACT_DIST_M, G[1] - u[1] * CONTACT_DIST_M)
        travel = (r_end[0] - pose['x']) * u[0] + (r_end[1] - pose['y']) * u[1]
        if not 0.05 <= travel <= 3.0:
            return {'ok': False, 'message': f'unreasonable push distance {travel:.2f} m'}
        self.get_logger().info(f'push {a_key}: staged, pushing {travel:.2f} m towards {b_key} (gap goal {gap:.2f} m)')
        moved, err = self._push_motion(start['frame'], travel, PUSH_SPEED, True, travel / PUSH_SPEED * 3.0 + 15.0)
        back = max(BACKOFF_M, moved)                              # all the way back to the staging point: the robot must not
        self._push_motion(start['frame'], back, 0.12, False, back / 0.12 * 3.0 + 15.0)   # stand on the box's old (stale) map cells
        pe = self.get_pose('map')
        if pe is not None:
            near = [math.hypot(pe['x'] - gx, pe['y'] - gy) for gx, gy in self._ghosts]
            if near and min(near) < 0.55:
                extra = min(0.7, 0.60 - min(near))
                self.get_logger().info(f'push: still {min(near):.2f} m from a moved box\'s stale map cells; backing off {extra:.2f} m more')
                self._push_motion(start['frame'], extra, 0.12, False, extra / 0.12 * 3.0 + 15.0)
        if err == 'obstacle':
            self.get_logger().warn(f'push safety stop after {moved:.2f} m: closest return (range m, angle deg, dir) = {self._push_trip}')
        if err:
            return {'ok': False, 'message': f'push stopped after {moved:.2f} of {travel:.2f} m: {ERR_TEXT.get(err, err)}'}
        time.sleep(3.0)                                           # box is now ~0.8 m ahead, face-on: let perception update it
        return {'ok': True, 'already': False, 'pushed_m': round(moved, 2)}

    def _push_to(self, a_key, b_key, gap):
        """Push box a next to box b: close look at b, push, measure, retry once. -> result dict (+ 'nothing_pushed')."""
        # The target is hidden behind the pushed box during the push, so take a close look at it first.
        # Not fatal: if the approach fails, carry on with the remembered position.
        nav = self.skill_navigate(f'object:{b_key}')
        if nav['status'] != 'success':
            if self._abort_flag:
                return {'status': 'error', 'message': ERR_TEXT['aborted']}
            self.get_logger().warn(f"push: close look at {b_key} failed ({nav.get('message')}); using the remembered position")
        else:
            time.sleep(3.5)

        distance = None
        for attempt in (1, 2):
            if self._abort_flag:
                return {'status': 'error', 'message': ERR_TEXT['aborted']}
            r = self._push_once(a_key, b_key, gap)
            if not r['ok']:
                if attempt == 1 and 'push stopped' in r['message']:
                    self.get_logger().warn(f"push attempt 1 stopped early ({r['message']}); re-staging and trying once more")
                    continue
                return {'status': 'error', 'message': r['message'], 'attempts': attempt,
                        'nothing_pushed': attempt == 1 and 'push stopped' not in r['message']}
            v = self._verify_pair(a_key, b_key)
            if not v['fresh'] and not v['ok']:
                self.get_logger().warn('push: no fresh side view of both boxes; retrying the measurement once')
                v = self._verify_pair(a_key, b_key)
            distance = v.get('distance')
            if not v['fresh'] and distance is not None:
                return {'status': 'error', 'attempts': attempt, 'pushed': True,
                        'message': (f'{a_key} was pushed, but no side view of both boxes could be reached to measure the gap; '
                                    f'the remembered estimate is {distance:.2f} m (need {GAP_MIN_M:.2f}-{GAP_MAX_M:.2f} m) '
                                    f'but that is NOT a verified measurement')}
            if v['ok']:
                return {'status': 'success', 'label': a_key, 'target': b_key, 'distance_m': round(distance, 2),
                        'required_m': [GAP_MIN_M, GAP_MAX_M], 'attempts': attempt,
                        'measured_from_the_side': v['fresh'], 'already_next_to': bool(r.get('already'))}
            if distance is None or distance < GAP_MIN_M:
                break
            self.get_logger().info(f'push attempt {attempt}: gap is {distance:.2f} m, trying again')
        return {'status': 'error',
                'message': (f'after pushing, {a_key} and {b_key} are '
                            f'{"unknown" if distance is None else f"{distance:.2f}"} m apart; '
                            f'need {GAP_MIN_M:.2f}-{GAP_MAX_M:.2f} m')}

    def skill_push_object_to(self, label: str, target_label: str, gap_m=GAP_TARGET_M) -> dict:
        a_key, b_key = self._norm_label(label), self._norm_label(target_label)
        if a_key is None or b_key is None or a_key == b_key:
            return {'status': 'error', 'message': f'need two different known boxes (known: {list(KNOWN_LABELS)})'}
        gap = max(GAP_MIN_M + 0.03, min(GAP_MAX_M - 0.03, num(gap_m, GAP_TARGET_M)))
        for key in (a_key, b_key):
            if self._get_object(key, min_conf=SEEN_CONF) is None:
                r = self.skill_find_object(key)
                if r.get('status') != 'success':
                    return {'status': 'error', 'message': f"could not locate {key}: {r.get('message')}"}
        res = self._push_to(a_key, b_key, gap)
        if res['status'] == 'error' and res.pop('nothing_pushed', False) and not self._abort_flag:
            # 'Next to' is symmetric: if A cannot be pushed (staging unreachable / path blocked), push B instead.
            self.get_logger().warn(f"push: {a_key} could not be pushed ({res['message']}); trying to move {b_key} instead")
            res2 = self._push_to(b_key, a_key, gap)
            res2.pop('nothing_pushed', None)
            if res2['status'] == 'success':
                res2['moved'] = b_key
                res2['note'] = f"{a_key} could not be pushed ({res['message']}); moved {b_key} instead"
                return res2
            res['message'] += f" | moving {b_key} instead also failed: {res2['message']}"
        res.pop('nothing_pushed', None)
        return res

    def skill_verify_relation(self, label_a: str, label_b: str) -> dict:
        a_key, b_key = self._norm_label(label_a), self._norm_label(label_b)
        if a_key is None or b_key is None or a_key == b_key:
            return {'status': 'error', 'message': f'need two different known boxes (known: {list(KNOWN_LABELS)})'}
        v = self._verify_pair(a_key, b_key)
        if v.get('distance') is None:
            return {'status': 'error', 'message': v.get('message', 'could not measure')}
        out = {'label_a': a_key, 'label_b': b_key, 'distance_m': round(v['distance'], 2),
               'next_to': bool(v['ok']), 'required_m': [GAP_MIN_M, GAP_MAX_M], 'measured_from_the_side': v['fresh']}
        out['answer'] = (f"{a_key} and {b_key} are {v['distance']:.2f} m apart: "
                         f"{'YES, next to each other' if v['ok'] else 'NO, not next to each other'} "
                         f"(next-to = {GAP_MIN_M:.2f}-{GAP_MAX_M:.2f} m)")
        return {'status': 'success', **out}

    def skill_describe_surroundings(self) -> dict:
        image_b64 = self._encode_latest_frame()
        if image_b64 is None:
            return {'status': 'error', 'message': self._no_frame_msg()}
        try:
            text = self.vision_model.chat_with_image(
                VLLM_MODEL, VISION_SYSTEM_PROMPT,
                'Briefly describe what is visible in this image (objects, colors, layout). One or two sentences.',
                image_b64,
            )
        except Exception as e:
            return {'status': 'error', 'message': f'vision model call failed: {e}'}
        return {'status': 'success', 'description': text}

    def skill_detect_object(self, label: str) -> dict:
        image_b64 = self._encode_latest_frame()
        if image_b64 is None:
            return {'status': 'error', 'message': self._no_frame_msg()}
        prompt = (
            f'Is a "{label}" visible in this image? Respond with ONLY a JSON object, no other text: '
            '{"found": true/false, "position": "left"|"center"|"right"|"n/a"}'
        )
        try:
            raw = self.vision_model.chat_with_image(VLLM_MODEL, VISION_SYSTEM_PROMPT, prompt, image_b64)
        except Exception as e:
            return {'status': 'error', 'message': f'vision model call failed: {e}'}

        m = re.search(r'\{.*?\}', raw or '', re.S)
        if not m:
            return {'status': 'error', 'message': 'vision model reply was not parseable JSON', 'raw': raw}
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {'status': 'error', 'message': 'vision model reply was not valid JSON', 'raw': raw}

        found = bool(parsed.get('found'))
        position = parsed.get('position', 'n/a')
        if position not in ('left', 'center', 'right', 'n/a'):
            position = 'n/a'
        if not found:
            position = 'n/a'
        return {'status': 'success', 'label': label, 'found': found, 'position': position}

    # -------------------------------------------------------------
    # Other skills
    # -------------------------------------------------------------

    def skill_pick_place(self, action: str, object_label: str) -> dict:
        if action == 'pick':
            if self._carrying is not None:
                return {'status': 'error', 'message': f'already carrying "{self._carrying}" - place it first'}
            self._carrying = object_label
            verb = 'Picked up'
        elif action == 'place':
            if self._carrying is None:
                return {'status': 'error', 'message': 'nothing is being carried - pick something first'}
            self._carrying = None
            verb = 'Placed down'
        else:
            return {'status': 'error', 'message': f"unknown action '{action}' - must be pick or place"}
        self.get_logger().info(f'{verb} "{object_label}" (simulated - no gripper on this platform).')
        return {'status': 'success', 'action': action, 'object_label': object_label, 'simulated': True}

    def skill_replan(self, reason: str) -> dict:
        self.get_logger().info(f'replan: {reason}')
        return {'status': 'acknowledged', 'reason': reason,
                'next': 'Noted - replan does nothing physical. Call a DIFFERENT action now, or give_up. Do not replan again.'}

    def skill_give_up(self, reason: str) -> dict:
        self.get_logger().warn(f'give_up: {reason}')
        return {'status': 'give_up', 'reason': reason}

    @staticmethod
    def _same_pose(a, b) -> bool:
        return (math.hypot(b['x'] - a['x'], b['y'] - a['y']) < 0.10
                and abs(wrap_pi(b['yaw'] - a['yaw'])) < math.radians(6.0))

    @staticmethod
    def _moved(a, b) -> bool:
        return (math.hypot(b['x'] - a['x'], b['y'] - a['y']) > STALL_MOVE_M
                or abs(wrap_pi(b['yaw'] - a['yaw'])) > math.radians(STALL_TURN_DEG))

    def _audit_completion(self, summary: str):
        """None if the request is evidenced as done, else a string of what is missing. Fails open."""
        pose, heading = self._pose_summary()
        user = (
            f"REQUEST: {self._goal}\n\n"
            f"FINAL STATE: {json.dumps({'robot_pose': pose, 'compass_heading_degrees': heading, 'known_objects': self._objects_for_prompt()})}\n\n"
            f"HISTORY: {json.dumps(self.history)}\n\n"
            f"PLANNER'S CLAIM: {summary}\n\nCall verdict."
        )
        try:
            name, a, _raw = self.model.plan(LLM_MODEL, AUDIT_PROMPT, user, VERDICT_TOOL)
        except Exception as e:
            self.get_logger().warn(f'completion audit unavailable ({e!r}) - accepting')
            return None
        if name != 'verdict' or not isinstance(a, dict) or a.get('fulfilled', True):
            return None
        return str(a.get('missing') or 'part of the request is not evidenced')

    def _verify_complete(self, args: dict) -> dict:
        """task_complete is only accepted if the last physical action succeeded."""
        if self._last_effect is not None:
            tool, res = self._last_effect
            if res.get('status') != 'success':
                return {'status': 'rejected',
                        'message': (f'cannot complete: the last physical action ({tool}) failed '
                                    f'with: {res.get("message", res.get("status"))}. '
                                    'Call replan with a new approach, or give_up.')}
        if AUDIT_COMPLETION and self.history:
            missing = self._audit_completion(str(args.get('summary', '')))
            if missing:
                return {'status': 'rejected',
                        'message': f'completion audit: not everything is evidenced - {missing}. '
                                   'Do the missing part, or give_up if it is impossible.'}
        return {'status': 'task_complete', **args}

    def dispatch(self, tool_name: str, tool_args: dict) -> dict:
        if not isinstance(tool_args, dict):
            return {'status': 'error', 'message': 'tool arguments must be a JSON object'}
        if tool_name == 'task_complete':
            return self._verify_complete(tool_args)
        table = {
            'navigate': self.skill_navigate,
            'drive': self.skill_drive,
            'describe_surroundings': self.skill_describe_surroundings,
            'detect_object': self.skill_detect_object,
            'find_object': self.skill_find_object,
            'push_object_to': self.skill_push_object_to,
            'verify_relation': self.skill_verify_relation,
            'explore': self.skill_explore,
            'pick_place': self.skill_pick_place,
            'replan': self.skill_replan,
            'give_up': self.skill_give_up,
        }
        fn = table.get(tool_name)
        if fn is None:
            return {'status': 'error', 'message': f'unknown skill {tool_name}'}
        try:
            return fn(**tool_args)
        except TypeError as e:
            return {'status': 'error', 'message': f'bad arguments for {tool_name}: {e}'}
        except Exception as e:  # never let a skill crash the node
            if self._abort_flag or not rclpy.ok():
                return {'status': 'error', 'message': 'aborted (shutting down)'}
            self.get_logger().error(f'{tool_name} raised: {e!r}')
            return {'status': 'error', 'message': f'{tool_name} failed: {e}'}

    # -------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------

    def prompt_callback(self, msg: String):
        goal = msg.data.strip()
        self.history = []
        self._last_effect = None
        self._carrying = None
        self._abort_flag = False
        self._push_trip = None
        self._goal = goal
        self._frontier_blacklist = []
        self._visited_sweeps = []
        self._tried_candidates = {}
        self._ghosts = []
        self.get_logger().info(f'GOAL: {goal}')
        try:
            self._run_goal(goal)
        except Exception as e:
            if not self._abort_flag:
                self.get_logger().error(f'goal loop crashed: {e!r}')
        finally:
            self._stop()

    def _run_goal(self, goal: str):
        rejections = 0
        stalls = 0
        replans = 0
        repeats = {}
        p_start = self.get_pose()
        trail = [p_start] if p_start is not None else []
        for step in range(1, MAX_ITERATIONS + 1):
            if self._abort_flag:
                self.get_logger().warn('ABORTED by /nirva/stop')
                return

            pose, heading = self._pose_summary()
            world_state = {
                'robot_pose': pose,
                'compass_heading_degrees': heading,
                'known_locations': list(POSITIONS.keys()),
                'objects': self._objects_for_prompt(),
            }
            user_prompt = (
                f"REQUEST: {goal}\n\n"
                f"WORLD STATE: {json.dumps(world_state)}\n\n"
                f"HISTORY: {json.dumps(self.history)}\n\n"
                "Next action?"
            )
            try:
                tool_name, tool_args, raw = self.model.plan(LLM_MODEL, SYSTEM_PROMPT, user_prompt, TOOLS)
            except Exception as e:
                self.get_logger().error(f'planner call failed: {e!r}')
                return

            if tool_name is None:
                try:
                    content = raw.choices[0].message.content
                except Exception:
                    content = None
                self.get_logger().warn(f'No tool call returned. content={content!r}')
                return

            self.get_logger().info(f'[{step}/{MAX_ITERATIONS}] {tool_name}({tool_args})')
            before = self.get_pose()
            result = self.dispatch(tool_name, tool_args)
            if self._abort_flag or not rclpy.ok():
                self.get_logger().warn('ABORTED')
                return
            self.get_logger().info(f'  -> {result}')
            self.history.append({'action': f'{tool_name}({tool_args})', 'result': result})

            sig = (tool_name, json.dumps(tool_args, sort_keys=True, default=str),
                   json.dumps(result, sort_keys=True, default=str))
            repeats[sig] = repeats.get(sig, 0) + 1
            if repeats[sig] >= 4:
                self.get_logger().warn(
                    f'GAVE UP: {tool_name} was called 4 times with the same arguments and the same result - planner is looping.')
                return

            if tool_name in ACTION_TOOLS:
                self._last_effect = (tool_name, result)

            # Progress watchdog: several motion calls in a row that leave the
            # robot exactly where it was means the planner is stuck.
            if tool_name in ('navigate', 'drive'):
                after = self.get_pose()
                if before is not None and after is not None and not self._moved(before, after):
                    stalls += 1
                else:
                    stalls = 0
                if stalls >= STALL_LIMIT:
                    self.get_logger().warn(
                        f'GAVE UP: {STALL_LIMIT} motion calls in a row did not move the robot - planner is stuck.')
                    return
                if after is not None:
                    trail.append(after)
                    osc = 0
                    for i in range(len(trail) - 1, 1, -1):
                        if self._same_pose(trail[i], trail[i - 2]):
                            osc += 1
                        else:
                            break
                    if osc >= 3:
                        self.get_logger().warn(
                            'GAVE UP: robot keeps returning to the same pose - planner is oscillating.')
                        return

            if tool_name == 'replan':
                replans += 1
                if replans >= 3:
                    self.get_logger().warn('GAVE UP: planner called replan 3 times in a row without acting.')
                    return
            else:
                replans = 0

            speech = String()
            speech.data = f'{tool_name}: {result.get("status", "")}'
            self.speech_pub.publish(speech)

            status = result.get('status')
            if status == 'task_complete':
                self.get_logger().info(f"DONE: {tool_args.get('summary')}")
                return
            if status == 'give_up':
                self.get_logger().warn(f"GAVE UP: {tool_args.get('reason')}")
                return
            if status == 'rejected':
                rejections += 1
                self.get_logger().warn(f'task_complete REJECTED: {result.get("message")}')
                if rejections >= 2:
                    self.get_logger().warn('GAVE UP: planner kept claiming completion after a failed action.')
                    return

        self.get_logger().warn('Max iterations reached without completion.')


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._abort_flag = True      # lets any running motion/goal loop exit
        if rclpy.ok():
            try:
                node._stop()
            except Exception:
                pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():               # ros2 run's SIGINT handler may already have shut down
            rclpy.shutdown()


if __name__ == '__main__':
    main()