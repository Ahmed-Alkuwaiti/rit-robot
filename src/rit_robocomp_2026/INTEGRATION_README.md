# ERC 2026 Integrated Solution

This package merges the delivered perception, navigation, and manipulation
handoffs and adds the missing RGB-D target bridge and mission task manager.

## Confirmed team interface

Perception publishes both targets as `geometry_msgs/msg/PoseStamped` in
`base_footprint`, with positions in metres:

- `/rit/perception/book_target`: final left-gripper grasp point
- `/rit/perception/bin_target`: gentle release point inside the red bin
- `/rit/perception/book_pixel_x`: live horizontal book center
- `/rit/perception/book_bbox_height`: live visual scale for optional calibration

Manipulation already subscribes to both. Navigation also consumes them for
live search, shelf approach/alignment, and bin alignment. Navigation publishes a
zero velocity as soon as a required target is older than 0.75 seconds, then
fails the motion if the signal does not recover during the initial acquisition
grace period. Manipulation rejects targets older than five seconds.

## Mission sequence

The task manager waits for `SUCCEEDED:<COMMAND>` before advancing and aborts
both subsystems on `FAILED:<COMMAND>:<REASON>`:

1. `TUCK_ARMS`
2. `SEARCH_SHELF`, `ALIGN_SHELF`, `APPROACH_SHELF`
3. `PREPARE_GRASP`, `GRASP`, `LIFT_AND_RETRACT`, `TRANSPORT_POSE`
4. `SEARCH_BIN`, `ALIGN_BIN` (combined lateral alignment and safe approach)
5. `PREPARE_PLACE`, `PLACE`, `SAFE_POSE`

## Install and run in the existing simulator container

Copy this `rit_robocomp_2026` directory over the package at
`/opt/erc_ws/src/rit_robocomp_2026`, then run:

```bash
cd /opt/erc_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rit_robocomp_2026
source install/setup.bash
ros2 launch rit_robocomp_2026 solution.launch.py \
  book_colour:=red shelf_column_number:=1 auto_start:=false
```

Keep `auto_start:=false` for the first interface check. In another terminal:

```bash
source /opt/erc_ws/install/setup.bash
ros2 topic list | grep -E 'head_front_camera|rit/perception|rit/navigation|rit/manipulation|rit/task'
ros2 topic echo /rit/perception/book_target --once
ros2 topic echo /rit/perception/bin_target --once
ros2 topic echo /rit/task/status
```

Confirm that each target prints `frame_id: base_footprint` and metre-scale
coordinates. Then relaunch with `auto_start:=true` for the full mission.

The simulator intentionally reshuffles book colours and cabinet number markers
on each launch. For repeatable debugging, set a fixed seed before starting the
simulator, for example `export ERC_SEED=2026`.

## Confirmed RGB-D topics

The supplied simulator publishes the working depth stream and matching camera
intrinsics on:

```text
/head_front_camera/head_front_camera/depth/image_rect_raw
/head_front_camera/head_front_camera/depth/camera_info
```

If a different simulator configuration uses other names, pass them at launch:

```bash
ros2 launch rit_robocomp_2026 solution.launch.py \
  book_colour:=red shelf_column_number:=1 auto_start:=false \
  depth_topic:=/ACTUAL/REGISTERED_DEPTH_TOPIC \
  camera_info_topic:=/ACTUAL/COLOR_CAMERA_INFO_TOPIC
```

The launch file extracts `/robot_state_publisher`'s `robot_description` into
`/tmp/tiago.urdf` before starting manipulation, matching the tested handoff.

## Calibration before autonomous testing

The 3D calculation is real camera projection plus TF, not pixel coordinates in
a pose. However, the final grasp/release contact points may require small
simulator-specific offsets. Parameters `book_offset_x/y/z` and
`bin_offset_x/y/z` are metres in `base_footprint`.

Start with `auto_start:=false`, inspect a target against TF/RViz, and tune these
parameters before allowing a physical grasp. Base positioning defaults to a
0.65 m target distance in `base_footprint`, with a 0.50 m hard target-distance
floor and 0.30 m LiDAR stop distance. Override them at launch with
`shelf_standoff_distance`, `bin_standoff_distance`,
`minimum_target_distance`, and `obstacle_stop_distance`.

`book_bbox_stop_height` defaults to `0` (diagnostic only). After observing the
bounding-box height at a verified safe working pose, set it to that calibrated
pixel height to add a visual hard stop. Search rotation remains `-0.15` rad/s,
the sign present in the supplied handoff; change `search_angular_speed` only
after a short observed direction check in this simulator.

A complete randomized-book mission still must be observed in the official
simulator; static checks cannot validate runtime TF, controller availability,
contacts, or competition-world geometry.
