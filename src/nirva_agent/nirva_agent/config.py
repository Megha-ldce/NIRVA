#!/usr/bin/env python3
"""config.py - central settings."""
import os

LLM_API_KEY = os.environ.get('NEBIUS_API_KEY', '')
LLM_BASE_URL = os.environ.get('NIRVA_LLM_BASE_URL', 'https://api.tokenfactory.nebius.com/v1')
LLM_MODEL = os.environ.get('NIRVA_LLM_MODEL', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B')

# Vision-language model used by describe_surroundings / detect_object.
# Run `python3 test_vision_models.py` to see which model ids on your Token Factory
# account accept images, then set NIRVA_VLM_MODEL. If the id is invalid those two
# skills return an error and the agent carries on with HSV+lidar perception.
VLLM_API_KEY = os.environ.get('NEBIUS_API_KEY', '')
VLLM_BASE_URL = os.environ.get('NIRVA_VLM_BASE_URL', 'https://api.tokenfactory.nebius.com/v1')
VLLM_MODEL = os.environ.get('NIRVA_VLM_MODEL', 'nvidia/Nemotron-Nano-V2-12b')

CAMERA_TOPIC = '/camera/image_raw'  # confirm this matches your xacro's camera plugin topic

MAX_ITERATIONS = 30  # polygons need 2 calls/side (turn+straight), e.g.
                     # an octagon alone needs 16 - give headroom

# Named places are in the MAP frame. The map origin is the robot's start pose
# (world (-2.0, -0.5)), so "start" is (0, 0) here, not the world coordinates.
POSITIONS = {
    "start": {"x": 0.0, "y": 0.0},
}

# Map-frame bounds for x,y navigation goals (ahead:/object: goals skip this check).
SAFE_BOUNDS = {"x_min": -6.0, "x_max": 6.0, "y_min": -6.0, "y_max": 6.0}