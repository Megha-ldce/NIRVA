# NIRVA architecture

## Two ROS 2 nodes plus the stock navigation stack

| Node | Role |
|---|---|
| `agent_node` | The brain. Receives a prompt, runs the plan → act → measure loop against Nemotron on Nebius Token Factory, executes skills, verifies completion. |
| `perception_node` | Camera + lidar → world model (JSON, 2 Hz). Independent of the LLM; the agent only reads its output. |
| `slam_toolbox` (online async) | Builds the map while the robot moves; no pre-built map. |
| Nav2 (navigation only) | Global/local planning and `navigate_to_pose`. No `map_server`/AMCL. |

### Topics and actions

| Name | Type | Direction (agent view) |
|---|---|---|
| `/nirva/prompt` | `std_msgs/String` | in: the user's command |
| `/nirva/stop` | `std_msgs/String` | in: abort motion immediately |
| `/nirva/speech` | `std_msgs/String` | out: short status text |
| `/nirva/world_model` | `std_msgs/String` (JSON) | in: boxes with map-frame position, confidence, range |
| `/nirva/debug_image` | `sensor_msgs/Image` | out of perception: detections drawn on the frame |
| `/nirva/snapshot` | `std_msgs/String` | in to perception: save a frame + HSV report |
| `/camera/image_raw` | `sensor_msgs/Image` | in |
| `/scan` | `sensor_msgs/LaserScan` | in: safety stop, ranging |
| `/map` | `nav_msgs/OccupancyGrid` | in: exploration frontiers, stale-cell handling |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | out: closed-loop `drive` and pushing |
| `navigate_to_pose` | Nav2 action | client |

## One iteration of the loop

1. Build the user message: the request, the **measured** state (pose, world model, map coverage) and the action **HISTORY** so far.
2. Call Nemotron with the system prompt and the tool schema; expect exactly one tool call.
3. Execute that skill in code. Skills run closed-loop on TF pose / lidar, not on timers.
4. Return a structured result (`success`, achieved distance/yaw, `reason`, `next` hint) into HISTORY.
5. If the model calls `task_complete`:
   - reject it if the last physical action failed;
   - otherwise ask Nemotron (verdict tool) whether every part of the request is evidenced in HISTORY. If not, the missing part is fed back and the loop continues.
6. Stop on `task_complete` (verified), `give_up`, the stall limit, `/nirva/stop`, or 30 steps.

## Failure handling

| Failure | Detection | Response |
|---|---|---|
| Nav2 reports success but robot is far from the goal | actual distance > 0.6 m | `navigate` returns an error with the real distance |
| Path blocked / no plan | Nav2 goal rejected or aborted | planner is told; may `replan` once with a new approach, else `give_up` |
| Robot not moving | 4 consecutive navigate/drive calls with no pose change | task ends |
| Obstacle ahead during `drive` | lidar front arc < 0.25 m | stop, report |
| Target not visible | `find_object` coverage exhausted | report; planner may explore elsewhere or give up |
| Box ends up too far / too close | `push_object_to` measures the gap after backing off | one automatic re-stage and push; result reports the measured gap |
| LLM claims it is done too early | code check + audit call | rejected, missing part returned to the planner |
| Phantom obstacle where a pushed box used to be | map vs. world model | stale occupied cells are ignored |

## `push_object_to` in detail

1. Compute the push line from box A (to move) toward box B; stage `STAGE_DIST_M` (0.55 m) behind A on that line via Nav2.
2. Drive forward at 0.10 m/s in contact, ignoring A in the front cone but watching the flanks, until A's centre is within the target band of B or a limit is hit.
3. Back off 0.45 m, re-observe both boxes from the side.
4. Measure the centre-to-centre gap. "Next to" is 0.30–0.50 m. If outside, re-stage and push once more.
5. Return the measured gap and verdict to the planner.

## Perception pipeline

HSV mask per colour → connected blobs → bearing from pixel column → lidar range at that bearing → size-consistency check (apparent size vs. range; mismatch ⇒ wall of the same hue, rejected, see `docs/assets/perception_debug.png`) → TF to map frame → rolling median over recent observations → confidence → JSON.
