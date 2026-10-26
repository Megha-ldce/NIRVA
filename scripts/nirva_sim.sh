#!/bin/bash
# NIRVA simulation
#   TurtleBot3 Waffle Pi (the Burger has NO camera) + red/green/blue boxes
#   + slam_toolbox ONLINE mapping + Nav2 navigation (no pre-built map, no AMCL).
#
# Usage:   ./nirva_sim.sh            (RViz opens)
#          NO_RVIZ=1 ./nirva_sim.sh  (lighter on an integrated GPU)
#
# NOTE: with SLAM the map frame ORIGIN is where the robot starts, so the robot
# starts at map (0,0). Set POSITIONS["start"] = (0,0) in nirva_agent/config.py
# and widen SAFE_BOUNDS (e.g. +-6) - the old (-2,-0.5) start no longer applies.

source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-waffle_pi}

# ---- GPU: this laptop is hybrid (Intel UHD 630 + NVIDIA GTX 1050 Ti). Run Gazebo/RViz rendering on the
# NVIDIA GPU when its driver works; otherwise fall back to Intel. Disable with NIRVA_NO_NVIDIA=1.
if [ -z "$NIRVA_NO_NVIDIA" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
  [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && \
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  echo "GPU: NVIDIA offload ON ($(nvidia-smi -L | head -1))"
else
  echo "GPU: NVIDIA offload OFF (driver not usable or NIRVA_NO_NVIDIA set) - rendering on Intel"
fi

# Box positions in GAZEBO WORLD coordinates. These assume the stock
# turtlebot3_world 3x3 pillar grid (pillars ~1.1 m apart); open the Gazebo GUI,
# check the boxes are clear of pillars and move them here if not.
RED_X=${RED_X:--0.55};   RED_Y=${RED_Y:--0.55}
GREEN_X=${GREEN_X:--0.55}; GREEN_Y=${GREEN_Y:-0.55}
BLUE_X=${BLUE_X:-0.55};  BLUE_Y=${BLUE_Y:--0.55}   # distractor

echo "╔══════════════════════════════════════╗"
echo "║        NIRVA SIM (model: $TURTLEBOT3_MODEL)"
echo "╚══════════════════════════════════════╝"

PIDS=""
cleanup() { echo "Shutting down..."; kill $PIDS 2>/dev/null; exit 0; }
trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------- box models
make_box() {  # name r g b
er=$(awk "BEGIN{print $2*0.6}"); eg=$(awk "BEGIN{print $3*0.6}"); eb=$(awk "BEGIN{print $4*0.6}")
cat > "/tmp/$1.sdf" <<EOF
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="$1">
    <link name="link">
      <inertial>
        <mass>0.3</mass>
        <inertia><ixx>0.00256</ixx><iyy>0.00256</iyy><izz>0.002</izz></inertia>
      </inertial>
      <collision name="c">
        <geometry><box><size>0.2 0.2 0.25</size></box></geometry>
        <surface><friction><ode><mu>0.4</mu><mu2>0.4</mu2></ode></friction></surface>
      </collision>
      <visual name="v">
        <geometry><box><size>0.2 0.2 0.25</size></box></geometry>
        <material>
          <ambient>$2 $3 $4 1</ambient>
          <diffuse>$2 $3 $4 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <emissive>$er $eg $eb 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
}
make_box red_box   1 0 0
make_box green_box 0 1 0
make_box blue_box  0 0 1

# ------------------------------------------------------------------- gazebo
echo "[1/4] Launching Gazebo ($TURTLEBOT3_MODEL)..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
PIDS="$PIDS $!"

echo "      waiting for the Gazebo world..."
WORLD=""
for i in $(seq 1 40); do
  WORLD=$(gz service -l 2>/dev/null | grep -m1 -oP '(?<=/world/)[^/]+(?=/create)')
  [ -n "$WORLD" ] && break
  sleep 1
done
if [ -z "$WORLD" ]; then
  echo "ERROR: could not find the Gazebo world. Is Gazebo running?"; cleanup
fi
echo "      world: $WORLD"
sleep 5

echo "[2/4] Spawning boxes..."
ros2 run ros_gz_sim create -world "$WORLD" -name red_box   -x "$RED_X"   -y "$RED_Y"   -z 0.13 -file /tmp/red_box.sdf
ros2 run ros_gz_sim create -world "$WORLD" -name green_box -x "$GREEN_X" -y "$GREEN_Y" -z 0.13 -file /tmp/green_box.sdf
ros2 run ros_gz_sim create -world "$WORLD" -name blue_box  -x "$BLUE_X"  -y "$BLUE_Y"  -z 0.13 -file /tmp/blue_box.sdf

# --------------------------------------------------------------------- slam
echo "[3/4] Launching slam_toolbox (online mapping)..."
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true &
PIDS="$PIDS $!"
sleep 5

# --------------------------------------------------------------------- nav2
echo "[4/4] Launching Nav2 (navigation only - no map_server/AMCL)..."
NAV2_PARAMS="/opt/ros/jazzy/share/turtlebot3_navigation2/param/${TURTLEBOT3_MODEL}.yaml"
if [ ! -f "$NAV2_PARAMS" ]; then
  echo "WARNING: $NAV2_PARAMS not found - Nav2 will use its default params"
  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True &
else
  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True params_file:="$NAV2_PARAMS" &
fi
PIDS="$PIDS $!"

if [ -z "$NO_RVIZ" ]; then
  sleep 5
  ros2 launch nav2_bringup rviz_launch.py &
  PIDS="$PIDS $!"
fi

echo ""
echo "✅ NIRVA sim up. Verify in another terminal:"
echo "   ros2 topic list --no-daemon | grep -E 'camera|scan|map'"
echo "   ros2 topic hz /camera/image_raw --no-daemon"
echo "Then (new terminals): ros2 run nirva_agent perception_node --ros-args -p use_sim_time:=true"
echo ""
wait