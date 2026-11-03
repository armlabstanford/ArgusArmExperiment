# Recording OptiTrack Mocap Data to a ROS 2 Bag (mocap4r2 / NatNet)

This workspace streams OptiTrack Motive data into ROS 2 Humble using the
[mocap4ros2_optitrack](https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack)
driver (built on OptiTrack's NatNet SDK) and records it to a bag file.

| | |
|---|---|
| Workspace | `~/ArgusArmExperiment/mocap_connect` |
| Motive PC | Motive Tracker 3.0.3, `192.169.1.12` |
| ROS PC | Ubuntu 22.04 / ROS 2 Humble, `192.169.1.24` (`arm-rog-2`) |
| Recorded topics | `/markers`, `/rigid_bodies` @ 120 Hz |
| Bags saved to | `~/ArgusArmExperiment/mocap_bags/<bag_name>/` |

---

## Quick start (every recording session)

```bash
cd ~/ArgusArmExperiment/mocap_connect
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ./launch/record_mocap.launch.py bag_name:=trial_01
# ... collect data ...
# Ctrl-C to stop
```

Use a new `bag_name` for every trial. Existing bags are never overwritten;
reusing a name fails with an error.

The rest of this document covers setup, the full session checklist, and
troubleshooting.

---

## 1. One-time setup

### 1.1 Build the workspace

No conda environment should be active when building or running (`conda deactivate`
until no env name shows in the prompt). Conda's Python breaks ROS 2 builds.

```bash
mkdir -p ~/ArgusArmExperiment/mocap_connect/src
cd ~/ArgusArmExperiment/mocap_connect/src
git clone https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack.git
git clone -b rolling https://github.com/MOCAP4ROS2-Project/mocap4r2_msgs.git
git clone -b rolling https://github.com/MOCAP4ROS2-Project/mocap4r2.git
touch mocap4r2/mocap4r2_control/rqt_mocap4r2_control/COLCON_IGNORE   # needs Qt dev libs; not used

cd ~/ArgusArmExperiment/mocap_connect
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to mocap4r2_optitrack_driver mocap4r2_marker_viz
```

The driver only has a `rolling` branch, but its CI builds `rolling` (with
`rolling` dependencies) on Humble. The NatNet SDK (`libNatNet.so`, x86-64) is
bundled in the repo.

### 1.2 Local driver modifications

`src/mocap4ros2_optitrack/mocap4r2_optitrack_driver/src/mocap4r2_optitrack_driver/mocap4r2_optitrack_driver.cpp`
has three changes in `process_frame()` compared with upstream:

1. **Untracked rigid bodies are skipped** (checks NatNet's tracking-valid flag,
   `params & 0x01`). Upstream publishes lost bodies as a zero pose at the origin.
2. **Marker-to-rigid-body grouping always runs.** Upstream only fills
   `rb.markers` when something subscribes to `/markers`.
3. **`marker_index = markerID` for labeled markers**, which is Motive's marker
   number within its rigid body (1…N), stable across frames. Upstream used the
   marker's position in the frame's list, which shifts when markers drop out.

If the driver is re-cloned, reapply these changes (see `git diff` in that repo)
and rebuild:

```bash
colcon build --symlink-install --packages-select mocap4r2_optitrack_driver
```

### 1.3 Driver config

`src/mocap4ros2_optitrack/mocap4r2_optitrack_driver/config/mocap4r2_optitrack_driver_params.yaml`:

```yaml
mocap4r2_optitrack_driver_node:
  ros__parameters:
    connection_type: "Unicast"
    server_address: "192.169.1.12"   # Motive PC
    local_address: "192.169.1.24"    # this machine (check with: ip -4 addr)
    multicast_address: "239.255.42.99"
    server_command_port: 1510
    server_data_port: 1511
```

Because of `--symlink-install`, edits to this file take effect on the next
launch with no rebuild.

### 1.4 Launch file

`launch/record_mocap.launch.py` (in the workspace root, outside `src/`):

- starts the driver and auto-configures + activates it (lifecycle node),
- starts `ros2 bag record` on `/markers` and `/rigid_bodies` **only after** the
  driver is active (no bag is created if it can't connect to Motive),
- writes to `~/ArgusArmExperiment/mocap_bags/<bag_name>/`, outside the colcon
  workspace so builds don't crawl recordings.

### 1.5 Motive PC (Windows)

- Network profile for the Wi-Fi set to **Private**.
- Firewall allows Motive, plus inbound UDP 1510/1511. In an admin PowerShell:

  ```powershell
  New-NetFirewallRule -DisplayName "NatNet" -Protocol UDP -LocalPort 1510,1511 -Direction Inbound -Action Allow
  ```

---

## 2. Recording session checklist

### 2.1 Motive

In **View → Data Streaming**:

- [ ] Streaming **enabled**
- [ ] **Local Interface:** `192.169.1.12` (the Wi-Fi adapter, *not* loopback or
      the camera-network adapter)
- [ ] **Transmission Type:** `Unicast` (must match `connection_type` in the config)
- [ ] **Up Axis:** `Z Up`
- [ ] **Rigid Bodies** and **Labeled Markers** checked

In the scene:

- [ ] Cameras calibrated, ground plane set
- [ ] Any rigid bodies you need are created and **enabled** in the Assets pane,
      and show as tracked in the viewport
- [ ] Note each rigid body's **Streaming ID**; this is the `rigid_body_name` in ROS

### 2.2 Network check (ROS PC)

```bash
ip -4 addr show wlp1s0      # confirm this machine is still 192.169.1.24
ping -c 3 192.169.1.12      # Motive PC reachable
```

The ROS PC's address comes from DHCP and can change. If it has, update
`local_address` in the config.

### 2.3 Record

```bash
cd ~/ArgusArmExperiment/mocap_connect
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ./launch/record_mocap.launch.py bag_name:=<subject>_<condition>_<trial>
```

Confirm these lines appear before starting the trial:

```
... connected!
Application: Motive (ver. 3.0.3.1)
Mocap Framerate : 120.00
Configured!
Activated!
Driver active, recording /markers /rigid_bodies to /home/argus/ArgusArmExperiment/mocap_bags/<bag_name>
Recording...
```

Press **Ctrl-C** to stop. The recorder flushes and closes the bag. A driver
`exit code -11` on shutdown is a known, harmless crash in NatNet teardown that
happens after recording stops.

### 2.4 (Optional) Live check while recording

In a second terminal:

```bash
source ~/ArgusArmExperiment/mocap_connect/install/setup.bash
ros2 topic hz /markers                  # ~120 Hz
ros2 topic echo /rigid_bodies --once
```

For visualization:

```bash
ros2 launch mocap4r2_marker_viz mocap4r2_marker_viz.launch.py mocap4r2_system:=optitrack
```

If markers aren't visible in RViz, zoom in toward the origin. They're drawn as
1.4 cm spheres with a 10 ms lifetime; run the viz node with
`-p marker_lifetime:=0.1` to reduce flicker.

### 2.5 Verify the bag

```bash
ros2 bag info ~/ArgusArmExperiment/mocap_bags/<bag_name>
```

Expect `/markers` and `/rigid_bodies` with **equal counts of about 120 × duration (s)**.
For example, a 4.57 s test produced 544 messages per topic (~119 Hz).

---

## 3. What's in the bag

Bag folder contents: `<bag_name>_0.db3` (SQLite data) and `metadata.yaml`.

**`/markers`** (`mocap4r2_msgs/Markers`): markers **not** assigned to a rigid body
(unlabeled or active markers). `marker_index` is only the position in that
frame's list. It is **not** a stable identity.

**`/rigid_bodies`** (`mocap4r2_msgs/RigidBodies`): one entry per **tracked**
rigid body:

- `rigid_body_name`: Motive's numeric Streaming ID (e.g. `"1"`), not the asset name
- `pose`: position (m) and orientation (quaternion)
- `markers`: that body's labeled markers; `marker_index` = Motive's marker
  number within the body (stable)

A rigid body that loses tracking is **absent** from that frame's message.
Downstream code should look bodies up by `rigid_body_name`, not list position.

**Frames and timestamps:**

- `header.frame_id` is `"map"` (the Motive world frame, Z up). No tf is published.
- `header.stamp` is the ROS PC's receive time. Motive currently reports
  "system latency not available," so no latency correction is applied. Account
  for this when aligning with other sensors.

**Labels:** Motive Tracker only labels markers that belong to a rigid body.
For per-marker identity (e.g. OpenSim), group markers into rigid-body clusters
in Motive; loose markers have no persistent ID.

**Reading bags outside ROS:** the message types are custom. In MATLAB, run
`ros2genmsg` on `src/mocap4r2_msgs` once. In Python, use `rosbag2_py` with the
workspace sourced, or the `rosbags` library with the `.msg` files.

---

## 4. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `... is not a valid package name` | The launch file path is wrong or the file lacks the `.py` extension. Check `ls launch/`; it must be `record_mocap.launch.py`. |
| Hangs at "Trying to connect…" / no "Configured!" | Wrong IPs in the config, Motive streaming off, or the Windows firewall is blocking UDP 1510/1511. Check `ping 192.169.1.12`. |
| Ping fails | The ROS PC IP changed (`ip -4 addr`), or the Windows firewall is blocking ICMP (set the network profile to Private). |
| "Configured!" but topics are silent | The driver isn't active. Check with `ros2 lifecycle get /mocap4r2_optitrack_driver_node`; the launch file normally activates it automatically. |
| No data even though connected | Motive's Transmission Type doesn't match `connection_type` (both must be Unicast). |
| `/rigid_bodies` is empty | No rigid bodies are tracked: create or enable them in Motive and check they're visible to the cameras. |
| `rigid_body_name` isn't what you named it | Expected: it's the Streaming ID. Check the rigid body's properties in Motive. |
| Bag name already exists | Use a new `bag_name`; bags are never overwritten. |
| Build fails with Python / `catkin_pkg` errors | A conda env is active: `conda deactivate`, then `rm -rf build install log` and rebuild. |
| RViz crashes with a GLib-GIO `antialiasing` error | Snap environment leaking into the terminal (e.g. VS Code snap). Run from GNOME Terminal or `unset GTK_PATH GIO_MODULE_DIR`. |

### Network note

Data currently goes over Wi-Fi. Unicast avoids the multicast burst/stall
behavior seen on Wi-Fi (~220 ms gaps followed by bursts), but a **wired
connection** (separate NIC on the Motive PC, static IPs) gives the most
consistent timing and is recommended for experiment data.