from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import get_types_from_msg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import os
import sys
from pathlib import Path # Using pathlib for more modern path manipulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml
import json  # Added for JSON export

# Ensure Pillow is installed for saving GIFs
try:
    import PIL
except ImportError:
    print("Pillow library not found. Please install it for saving GIFs: pip install Pillow")
    # Optionally exit if Pillow is strictly required
    # sys.exit(1)

figsize = (10, 8)

def set_axes_equal(ax):
  """Set equal scale for all axes."""
  x_limits = ax.get_xlim()
  y_limits = ax.get_ylim()
  z_limits = ax.get_zlim()

  x_range = abs(x_limits[1] - x_limits[0])
  y_range = abs(y_limits[1] - y_limits[0])
  z_range = abs(z_limits[1] - z_limits[0])

  # Handle cases where range is zero
  x_range = x_range if x_range > 1e-6 else 1.0
  y_range = y_range if y_range > 1e-6 else 1.0
  z_range = z_range if z_range > 1e-6 else 1.0

  max_range = max(x_range, y_range, z_range) / 2.0

  x_middle = (x_limits[0] + x_limits[1]) * 0.5
  y_middle = (y_limits[0] + y_limits[1]) * 0.5
  z_middle = (z_limits[0] + z_limits[1]) * 0.5

  ax.set_xlim(x_middle - max_range, x_middle + max_range)
  ax.set_ylim(y_middle - max_range, y_middle + max_range)
  ax.set_zlim(z_middle - max_range, z_middle + max_range)

def create_wall_vertices(wall_params):
  """Create vertices for a wall given its parameters.
  
  Args:
    wall_params: dict with keys:
      - size: (width, length, height) tuple
      - center: (x, y, z) tuple for center position
      - rotation: optional rotation angle in radians (around z-axis)
  
  Returns:
    List of vertices for the 6 faces of the cuboid wall
  """
  width, length, height = wall_params['size']
  cx, cy, cz = wall_params['center']
  
  # Create vertices for a cuboid centered at origin
  dx, dy, dz = width/2, length/2, height/2
  vertices = np.array([
    [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],  # bottom
    [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],      # top
  ])
  
  # Apply rotation if specified
  if 'rotation' in wall_params:
    alpha = wall_params['rotation']
    # Create rotation matrix for z-axis rotation
    rot_matrix = np.array([
      [np.cos(alpha), -np.sin(alpha), 0],
      [np.sin(alpha), np.cos(alpha), 0],
      [0, 0, 1]
    ])
    vertices = vertices @ rot_matrix.T
  
  # Translate to center position
  vertices += np.array([cx, cy, cz])
  
  # Define the 6 faces of the cuboid
  faces = [
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
    [vertices[0], vertices[3], vertices[2], vertices[1]],  # bottom
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
    [vertices[0], vertices[4], vertices[7], vertices[3]],  # left
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
  ]
  
  return faces

def draw_walls(ax):
  """Draw walls on a 3D axis."""
  # Wall 8 parameters
  wall8 = {
    'size': (0.2, 1.3, 8.0),
    'center': (0.0, 7.65, 4.0)
  }
  
  # Wall 9 parameters
  wall9 = {
    'size': (0.2, 3.0, 8.0),
    'center': (-1.0, 6.0, 4.0),
    'rotation': -0.78
  }
  
  # Create and draw wall 8
  wall8_faces = create_wall_vertices(wall8)
  for face in wall8_faces:
    poly = Poly3DCollection([face], color='gray', alpha=0.3, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(poly)
  
  # Create and draw wall 9
  wall9_faces = create_wall_vertices(wall9)
  for face in wall9_faces:
    poly = Poly3DCollection([face], color='gray', alpha=0.3, edgecolor='black', linewidth=0.5)
    ax.add_collection3d(poly)

def analyze_ros2_bag(bag_path, namespace, t0=0, tf=float('inf'), export_angular_velocity=True):
  yaml_file = '/project_code/racing/ese651_sim2real/src/jirl_bringup/config/config.yaml'
  with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)
  waypoints = np.array(data["/*/controller"]["ros__parameters"]["policy"]["waypoints"]).reshape(-1, 6)

  print("Waypoints:", waypoints)

  d = 0.5
  local_square = np.array([
      [0,  d,  d],
      [0, -d,  d],
      [0, -d, -d],
      [0,  d, -d]
  ])  # shape (4, 3)

  wp_pos = waypoints[:, :3]             # shape (N, 3)
  wp_euler = waypoints[:, 3:]           # shape (N, 3)

  rotations = R.from_euler('xyz', wp_euler).as_matrix()  # shape (N, 3, 3)
  # Use rot.T: einsum 'ij,nkj->nik' contracts j with the transposed rotation (same as local_square @ rot.T)
  verts_all = np.einsum('ij,nkj->nik', local_square, rotations) + wp_pos[:, np.newaxis, :]  # shape (N, 4, 3)

  # --- Path setup for saving plots ---
  bag_path_obj = Path(bag_path).resolve() # Get absolute path

  # Determine the base experiment name (directory name)
  # Handle based on whether input path is file or dir
  if bag_path_obj.is_file():
      experiment_dir = bag_path_obj.parent
  elif bag_path_obj.is_dir():
      experiment_dir = bag_path_obj
  else:
      # Fallback if the path doesn't exist yet (use input path structure)
      experiment_dir = Path(bag_path) # Use the input path directly

  experiment_name = experiment_dir.name
  if not experiment_name: # Handle root path case
      print(f"Warning: Cannot determine experiment name from path '{bag_path}'. Using 'unknown_experiment'.")
      experiment_name = "unknown_experiment"
      # Decide where to save plots if base path is weird
      if experiment_dir.parent == experiment_dir: # Check if it's root
           plots_parent_dir = Path("./plots") # Save in current dir subfolder
      else:
           plots_parent_dir = experiment_dir.parent / 'plots'
  else:
      # Go up one level from experiment_dir to find the base 'logs' dir
      logs_base_dir = experiment_dir.parent
      plots_parent_dir = logs_base_dir / 'plots'

  output_plot_dir = plots_parent_dir / experiment_name
  output_plot_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
  print(f"Saving plots and animation to: {output_plot_dir}")
  # --- End Path setup ---

  # --- Determine the path rosbags should open ---
  # rosbags.rosbag2.Reader accepts:
  #   - a directory containing metadata.yaml + .db3 or .mcap files
  #   - a direct path to an .mcap file
  resolved_bag_path = Path(bag_path).resolve()

  if not resolved_bag_path.exists():
      print(f"Error: Bag path '{bag_path}' does not exist.")
      sys.exit(1)
  elif resolved_bag_path.is_file() and resolved_bag_path.suffix == '.db3':
      # rosbags expects the *directory* for sqlite3 bags
      reader_path = resolved_bag_path.parent
      print(f"Input is a .db3 file; using parent directory as bag path: {reader_path}")
  else:
      # Directory (sqlite3 or mcap) or direct .mcap file — rosbags handles both
      reader_path = resolved_bag_path

  print(f"Opening bag at: {reader_path}")

  # --- Build typestore for deserialization ---
  # Uses ROS2 Humble built-in types. For other distros swap Stores.ROS2_HUMBLE
  # with e.g. Stores.ROS2_IRON or Stores.LATEST.
  typestore = get_typestore(Stores.ROS2_HUMBLE)

  # --- Register jirl_interfaces custom message types ---
  # Definitions mirror the MSG files in src/jirl_interfaces/msg/.
  # Update these if the message definitions change.
  _JIRL_MSG_DEFS = {
      'jirl_interfaces/msg/CommandCTBR': (
          'string crazyflie_name\n'
          'uint16 thrust_pwm\n'
          'float64 thrust_n\n'
          'float64 roll_rate\n'
          'float64 pitch_rate\n'
          'float64 yaw_rate\n'
      ),
      'jirl_interfaces/msg/Trajectory': (
          'float64[3] x\n'
          'float64[3] x_dot\n'
          'float64[3] x_ddot\n'
          'float64[3] x_dddot\n'
          'float64[3] x_ddddot\n'
          'float64 yaw\n'
          'float64 yaw_dot\n'
          'float64 yaw_ddot\n'
      ),
      'jirl_interfaces/msg/Observations': (
          'float64[3] lin_vel\n'
          'float64[9] rot\n'
          'float64[12] corners_pos_b_curr\n'
          'float64[12] corners_pos_b_next\n'
          'float64[2] cond\n'
      ),
      'jirl_interfaces/msg/OdometryArray': (
          'nav_msgs/msg/Odometry[] odom_array\n'
      ),
  }

  _all_custom_types = {}
  for _msgtype, _msgdef in _JIRL_MSG_DEFS.items():
      try:
          _all_custom_types.update(get_types_from_msg(_msgdef, _msgtype))
      except Exception as _e:
          print(f"Warning: Could not parse definition for {_msgtype}: {_e}")
  if _all_custom_types:
      try:
          typestore.register(_all_custom_types)
          print(f"Registered {len(_all_custom_types)} custom jirl_interfaces type(s).")
      except Exception as _e:
          print(f"Warning: Could not register custom types: {_e}")

  # --- Data containers ---
  timestamps = []
  timestamps_cmd = []
  timestamps_traj = []
  timestamps_obs = []
  gt_pos = {"x": [], "y": [], "z": []}
  gt_quat = {"x": [], "y": [], "z": [], "w": []}
  gt_euler = {"roll": [], "pitch": [], "yaw": []}
  gt_lin_vel = {"x": [], "y": [], "z": []}
  gt_ang_vel = {"x": [], "y": [], "z": []}
  thrust_pwm = []
  thrust_N = []
  roll_rate = []
  pitch_rate = []
  yaw_rate = []
  dist_next_gate = []
  pose_wrt_gate_body = []  # Gate center position in body frame
  traj = {"x": [], "x_dot": [], "x_ddot": [], "x_dddot": [], "x_ddddot": [],
          "yaw": [], "yaw_dot": [], "yaw_ddot": []}

  first_timestamp = None

  # Topic name patterns we care about (normalized, no leading slash)
  odom_topic  = f"{namespace}/odom"
  cmd_topic   = f"/ctbr_cmd"
  traj_topic  = f"{namespace}/trajectory"
  obs_topic   = f"{namespace}/observations"
  global_cmd_topic = "ctbr_cmd"

  try:
    with Rosbag2Reader(str(reader_path)) as reader:

      # --- Build topic→msgtype map from connections ---
      type_map = {conn.topic: conn.msgtype for conn in reader.connections}

      print("\nAvailable topics:")
      for topic, msg_type in type_map.items():
        print(f"- {topic}: {msg_type}")

      # Warn about missing required topics
      required_topics = [
          f"/{namespace}/observations", f"/{namespace}/odom",
          f"/ctbr_cmd",    f"/{namespace}/trajectory",
        #   "/ctbr_cmd",
      ]
      missing_topics = []
      for req_topic in required_topics:
          base = req_topic.lstrip('/')
          if req_topic not in type_map and f"/{base}" not in type_map and base not in type_map:
              missing_topics.append(req_topic)
      if missing_topics:
          print("\nWarning: The following required topics were not found in the bag:")
          for t in missing_topics:
              print(f"- {t}")

      # --- Identify any types still not in the typestore after pre-registration ---
      unknown_types: set = set()
      for conn in reader.connections:
          if conn.msgtype not in typestore.fielddefs:
              unknown_types.add(conn.msgtype)

      if unknown_types:
          print(
              "\nNote: The following message types are not in the typestore and will be "
              "skipped if deserialization fails:\n"
              + "\n".join(f"  - {t}" for t in unknown_types)
              + "\nAdd their MSG definitions to _JIRL_MSG_DEFS at the top of this script "
              "to support them."
          )

      # --- Select only connections for topics we actually use ---
      wanted = {odom_topic, cmd_topic, traj_topic, obs_topic, global_cmd_topic}
      relevant_connections = [
          conn for conn in reader.connections
          if conn.topic.lstrip('/') in wanted
      ]

      print("\nReading bag data...")
      message_count = 0
      processed_count = 0

      for connection, timestamp_ns, rawdata in reader.messages(connections=relevant_connections):
        try:
          message_count += 1
          timestamp_sec = timestamp_ns * 1e-9  # nanoseconds → seconds

          if first_timestamp is None:
            first_timestamp = timestamp_sec

          rel_time = timestamp_sec - first_timestamp
          if rel_time < t0 or rel_time > tf:
            continue  # outside requested time window

          # --- Deserialize ---
          try:
            message = typestore.deserialize_cdr(rawdata, connection.msgtype)
          except Exception as deser_err:
            if connection.msgtype not in unknown_types:
              print(f"Warning: Could not deserialize '{connection.msgtype}' "
                    f"on topic '{connection.topic}': {deser_err}")
              unknown_types.add(connection.msgtype)
            continue

          processed_count += 1
          normalized_topic = connection.topic.lstrip('/')

          # --- Topic processing ---
          if normalized_topic == odom_topic:
            timestamps.append(rel_time)
            gt_pos["x"].append(message.pose.pose.position.x)
            gt_pos["y"].append(message.pose.pose.position.y)
            gt_pos["z"].append(message.pose.pose.position.z)
            gt_quat["x"].append(message.pose.pose.orientation.x)
            gt_quat["y"].append(message.pose.pose.orientation.y)
            gt_quat["z"].append(message.pose.pose.orientation.z)
            gt_quat["w"].append(message.pose.pose.orientation.w)

            quat = [
              message.pose.pose.orientation.x,
              message.pose.pose.orientation.y,
              message.pose.pose.orientation.z,
              message.pose.pose.orientation.w
            ]
            rot = R.from_quat(quat)
            rot_T = rot.as_matrix().T

            lin_vel_world = np.array([
              message.twist.twist.linear.x,
              message.twist.twist.linear.y,
              message.twist.twist.linear.z
            ])
            ang_vel_world = np.array([
              message.twist.twist.angular.x,
              message.twist.twist.angular.y,
              message.twist.twist.angular.z
            ])

            lin_vel_body = rot_T @ lin_vel_world
            ang_vel_body = rot_T @ ang_vel_world * 180.0 / np.pi

            gt_lin_vel["x"].append(lin_vel_body[0])
            gt_lin_vel["y"].append(lin_vel_body[1])
            gt_lin_vel["z"].append(lin_vel_body[2])
            gt_ang_vel["x"].append(ang_vel_body[0])
            gt_ang_vel["y"].append(ang_vel_body[1])
            gt_ang_vel["z"].append(ang_vel_body[2])

          elif normalized_topic in (cmd_topic, global_cmd_topic):
            cf_name_in_msg = getattr(message, 'crazyflie_name', None)
            if cf_name_in_msg is not None and namespace not in cf_name_in_msg:
              continue  # belongs to a different drone
            timestamps_cmd.append(rel_time)
            thrust_pwm.append(getattr(message, 'thrust_pwm', float('nan')))
            thrust_N.append(getattr(message, 'thrust_n', float('nan')))
            roll_rate.append(getattr(message, 'roll_rate', float('nan')))
            pitch_rate.append(getattr(message, 'pitch_rate', float('nan')))
            yaw_rate.append(getattr(message, 'yaw_rate', float('nan')))

          elif normalized_topic == traj_topic:
            timestamps_traj.append(rel_time)
            traj["x"].append(getattr(message, 'x', [float('nan')] * 3))
            traj["x_dot"].append(getattr(message, 'x_dot', [float('nan')] * 3))
            traj["x_ddot"].append(getattr(message, 'x_ddot', [float('nan')] * 3))
            traj["x_dddot"].append(getattr(message, 'x_dddot', [float('nan')] * 3))
            traj["x_ddddot"].append(getattr(message, 'x_ddddot', [float('nan')] * 3))
            traj["yaw"].append(getattr(message, 'yaw', float('nan')))
            traj["yaw_dot"].append(getattr(message, 'yaw_dot', float('nan')))
            traj["yaw_ddot"].append(getattr(message, 'yaw_ddot', float('nan')))

          elif normalized_topic == obs_topic:
            timestamps_obs.append(rel_time)
            corners = np.asarray(getattr(message, 'corners_pos_b_curr', np.full(12, float('nan')))).reshape(4, 3)
            mean_point = corners.mean(axis=0)
            dist_next_gate.append(np.linalg.norm(mean_point))
            pose_wrt_gate_body.append(mean_point.copy())

        except Exception as e:
          print(f"\nError processing message #{message_count} on '{connection.topic}': {e}")
          continue

  except Exception as e:
    print(f"Error opening bag file '{reader_path}': {e}")
    print("Ensure the path is correct and the bag is not corrupted.")
    return

  print(f"Finished reading bag data. Read {message_count} messages total, processed {processed_count} within time range [{t0}, {tf}].")

  # -- Print initial pose ---
  if gt_quat["x"]:
      quat = [gt_quat["x"][0], gt_quat["y"][0], gt_quat["z"][0], gt_quat["w"][0]]
      rpy = R.from_quat(quat).as_euler('xyz', degrees=True)  # 'xyz' = roll, pitch, yaw

      print("Initial pose:")
      print(f"Position: x = {gt_pos['x'][0]:.3f}, y = {gt_pos['y'][0]:.3f}, z = {gt_pos['z'][0]:.3f}")
      print(f"Orientation (RPY): roll = {rpy[0]:.3f}°, pitch = {rpy[1]:.3f}°, yaw = {rpy[2]:.3f}°")

  # -- Print average velocity within timeframe ---
  if gt_lin_vel["x"]:
      vel_x = np.array(gt_lin_vel["x"])
      vel_y = np.array(gt_lin_vel["y"])
      vel_z = np.array(gt_lin_vel["z"])
      vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
      avg_velocity = np.mean(vel_magnitude)
      print(f"\nAverage velocity (t={t0}s to t={tf}s): {avg_velocity:.3f} m/s")
      print(f"  Components (body frame): vx={np.mean(vel_x):.3f}, vy={np.mean(vel_y):.3f}, vz={np.mean(vel_z):.3f} m/s")

  # --- Data processing and Plotting ---

  # Check if essential data was loaded
  if not timestamps:
      print("\nError: No odometry data (topic: /{}/odom) found for the specified namespace and time range.".format(namespace))
      print("Cannot generate plots.")
      return

  # Convert trajectory lists to numpy arrays if they contain data
  if traj["x"]:
      try:
          traj["x"] = np.vstack(traj["x"])
          traj["x_dot"] = np.vstack(traj["x_dot"])
          traj["x_ddot"] = np.vstack(traj["x_ddot"])
          traj["x_dddot"] = np.vstack(traj["x_dddot"])
          traj["x_ddddot"] = np.vstack(traj["x_ddddot"])
      except ValueError as e:
          print(f"\nWarning: Could not stack trajectory arrays, likely due to inconsistent shapes or NaNs: {e}")
          print("Trajectory plots might be incorrect or fail.")
          # Set to empty arrays to prevent downstream errors if stacking fails
          for k in traj: traj[k] = np.array([])
      # Check if trajectory array is actually usable after stacking
      if not traj["x"].size:
           print("Warning: Trajectory data was found but resulted in empty arrays after processing.")

  else:
      print("\nWarning: No trajectory data (topic: /{}/trajectory) found or processed.".format(namespace))
      # Ensure keys exist but are empty numpy arrays for consistency downstream
      for k in traj: traj[k] = np.array([])


  # Convert quaternion to Euler angles
  if gt_quat["x"]:
      quaternions = np.column_stack((gt_quat["x"], gt_quat["y"], gt_quat["z"], gt_quat["w"]))
      # Check for invalid quaternions (e.g., all zeros, NaNs) before conversion
      valid_quat_mask = np.all(np.isfinite(quaternions), axis=1) & (np.linalg.norm(quaternions, axis=1) > 1e-6)
      if not np.all(valid_quat_mask):
          print(f"\nWarning: Found {len(quaternions) - np.sum(valid_quat_mask)} invalid quaternions (NaNs or zero norm). Replacing with identity/NaN.")
          # Option 1: Replace invalid ones with identity (0,0,0,1) -> (0,0,0) Euler
          # quaternions[~valid_quat_mask] = [0, 0, 0, 1]
          # Option 2: Keep them as NaN Euler angles (perhaps better to show data issues)
          euler_angles = np.full((len(quaternions), 3), np.nan) # Initialize with NaNs
          try:
               if np.any(valid_quat_mask): # Only convert if there are valid ones
                    valid_quats = quaternions[valid_quat_mask]
                    euler_angles[valid_quat_mask] = R.from_quat(valid_quats).as_euler('xyz', degrees=True)
          except ValueError as e:
               print(f"Error converting valid quaternions to Euler angles: {e}. Euler angles will contain NaNs.")
               # Keep euler_angles as initialized with NaNs
      else:
            # All quaternions are valid
            try:
                euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
            except ValueError as e:
                print(f"Error converting quaternions to Euler angles: {e}. Filling with NaNs.")
                euler_angles = np.full((len(quaternions), 3), np.nan)

      gt_euler["roll"] = euler_angles[:, 0].tolist()
      gt_euler["pitch"] = euler_angles[:, 1].tolist()
      gt_euler["yaw"] = euler_angles[:, 2].tolist()

  else:
      print("\nWarning: No quaternion data found for Euler conversion.")
      # Ensure gt_euler lists have the same length as timestamps if needed elsewhere, filled with NaN
      nan_list = [np.nan] * len(timestamps)
      gt_euler = {"roll": nan_list, "pitch": nan_list, "yaw": nan_list}

  # --- Export angular velocity and position data to JSON ---
  if export_angular_velocity and timestamps:
      # Export angular velocity data
      angular_velocity_data = {
          'timestamp': timestamps,
          'actual_x': gt_ang_vel["x"],
          'actual_y': gt_ang_vel["y"], 
          'actual_z': gt_ang_vel["z"],
          'timestamp_cmd': timestamps_cmd,
          'desired_x': roll_rate,
          'desired_y': pitch_rate,
          'desired_z': yaw_rate
      }
      
      angular_json_filename = output_plot_dir / f"{namespace}_angular_velocity_data.json"
      with open(angular_json_filename, 'w') as f:
          json.dump(angular_velocity_data, f)
      
      print(f"\nExported angular velocity data to: {angular_json_filename}")
      print(f"Data points: {len(timestamps)} actual, {len(timestamps_cmd)} desired")
      
      # Export position and trajectory data
      position_data = {
          'timestamp': timestamps,
          'position': {
              'x': gt_pos["x"],
              'y': gt_pos["y"],
              'z': gt_pos["z"]
          },
          'velocity': {
              'x': gt_lin_vel["x"],
              'y': gt_lin_vel["y"],
              'z': gt_lin_vel["z"]
          },
          'orientation': {
              'qx': gt_quat["x"],
              'qy': gt_quat["y"],
              'qz': gt_quat["z"],
              'qw': gt_quat["w"]
          },
          'euler_angles': {
              'roll': gt_euler["roll"],
              'pitch': gt_euler["pitch"],
              'yaw': gt_euler["yaw"]
          }
      }
      
      # Add desired trajectory data if available
      if timestamps_traj and isinstance(traj["x"], np.ndarray) and traj["x"].size > 0:
          position_data['timestamp_traj'] = timestamps_traj
          position_data['desired_position'] = {
              'x': traj["x"][:, 0].tolist() if traj["x"].ndim == 2 else [],
              'y': traj["x"][:, 1].tolist() if traj["x"].ndim == 2 else [],
              'z': traj["x"][:, 2].tolist() if traj["x"].ndim == 2 else []
          }
          position_data['desired_velocity'] = {
              'x': traj["x_dot"][:, 0].tolist() if traj["x_dot"].ndim == 2 else [],
              'y': traj["x_dot"][:, 1].tolist() if traj["x_dot"].ndim == 2 else [],
              'z': traj["x_dot"][:, 2].tolist() if traj["x_dot"].ndim == 2 else []
          }
          position_data['desired_yaw'] = traj["yaw"]
          position_data['desired_yaw_dot'] = traj["yaw_dot"]
      
      position_json_filename = output_plot_dir / f"{namespace}_position_data.json"
      with open(position_json_filename, 'w') as f:
          json.dump(position_data, f)
      
      print(f"Exported position data to: {position_json_filename}")
      print(f"Data points: {len(timestamps)} position/orientation points")
      if 'timestamp_traj' in position_data:
          print(f"             {len(timestamps_traj)} desired trajectory points")

  print("\nGenerating plots...")

  # --- Ground truth data plot ---
  fig_gt, axs_gt = plt.subplots(4, 1, figsize=figsize, sharex=True)
  fig_gt.suptitle(f"Ground Truth Data ({namespace})")

  axs_gt[0].plot(timestamps, gt_pos["x"], label="$x$")
  axs_gt[0].plot(timestamps, gt_pos["y"], label="$y$")
  axs_gt[0].plot(timestamps, gt_pos["z"], label="$z$")
  axs_gt[0].set_ylabel("Position [m]")
  axs_gt[0].legend()
  axs_gt[0].grid(True)

  axs_gt[1].plot(timestamps, gt_euler["roll"], label="Roll")
  axs_gt[1].plot(timestamps, gt_euler["pitch"], label="Pitch")
  axs_gt[1].plot(timestamps, gt_euler["yaw"], label="Yaw")
  axs_gt[1].set_ylabel("Euler Angles [deg]")
  axs_gt[1].legend()
  axs_gt[1].grid(True)

  axs_gt[2].plot(timestamps, gt_lin_vel["x"], label="$v_{x}$")
  axs_gt[2].plot(timestamps, gt_lin_vel["y"], label="$v_{y}$")
  axs_gt[2].plot(timestamps, gt_lin_vel["z"], label="$v_{z}$")
  axs_gt[2].set_ylabel("Linear Velocity [m/s]")
  axs_gt[2].legend()
  axs_gt[2].grid(True)

  axs_gt[3].plot(timestamps, gt_ang_vel["x"], label=r"$\omega_{x}$")
  axs_gt[3].plot(timestamps, gt_ang_vel["y"], label=r"$\omega_{y}$")
  axs_gt[3].plot(timestamps, gt_ang_vel["z"], label=r"$\omega_{z}$")
  axs_gt[3].set_xlabel("Time [s]")
  axs_gt[3].set_ylabel("Angular Velocity [deg/s]")
  axs_gt[3].set_ylim(-250, 250)
  axs_gt[3].legend()
  axs_gt[3].grid(True)

  fig_gt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
  gt_filename = output_plot_dir / f"{namespace}_ground_truth"
  try:
      print(f"Saving ground truth plot to {gt_filename}")
      fig_gt.savefig(gt_filename.with_suffix('.eps'), format='eps', bbox_inches='tight', dpi=300, facecolor=fig_gt.get_facecolor())
      fig_gt.savefig(gt_filename.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300, facecolor=fig_gt.get_facecolor())
  except Exception as e:
      print(f"Error saving ground truth plot: {e}")

  # --- Positions comparison plot ---
  if timestamps_traj and traj["x"].ndim == 2 and traj["x"].shape[0] > 0: # Check if traj data is valid 2D array
    fig_pos, axs_pos = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig_pos.suptitle(f"Position Comparison ({namespace})")
    axs_pos[0].plot(timestamps, gt_pos["x"], label="Actual")
    axs_pos[0].plot(timestamps_traj, traj["x"][:, 0], label="Desired", linestyle='--')
    axs_pos[0].set_ylabel("x [m]")
    axs_pos[0].legend()
    axs_pos[0].grid(True)

    axs_pos[1].plot(timestamps, gt_pos["y"], label="Actual")
    axs_pos[1].plot(timestamps_traj, traj["x"][:, 1], label="Desired", linestyle='--')
    axs_pos[1].set_ylabel("y [m]")
    axs_pos[1].legend()
    axs_pos[1].grid(True)

    axs_pos[2].plot(timestamps, gt_pos["z"], label="Actual")
    axs_pos[2].plot(timestamps_traj, traj["x"][:, 2], label="Desired", linestyle='--')
    axs_pos[2].set_xlabel("Time [s]")
    axs_pos[2].set_ylabel("z [m]")
    axs_pos[2].legend()
    axs_pos[2].grid(True)

    fig_pos.tight_layout(rect=[0, 0.03, 1, 0.97])
    pos_filename = output_plot_dir / f"{namespace}_position_comparison"
    try:
        print(f"Saving position comparison plot to {pos_filename}")
        fig_pos.savefig(pos_filename.with_suffix('.eps'), format='eps', bbox_inches='tight', dpi=300, facecolor=fig_pos.get_facecolor())
        fig_pos.savefig(pos_filename.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300, facecolor=fig_pos.get_facecolor())
    except Exception as e:
        print(f"Error saving position comparison plot: {e}")
  else:
    print("Skipping position comparison plot (no valid trajectory data found/processed).")


  # --- Angular velocities comparison plot ---
  if timestamps_cmd: # Check if there's command data to compare
    fig_rates, axs_rates = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig_rates.suptitle(f"Angular Velocity Comparison ({namespace})")

    axs_rates[0].plot(timestamps, gt_ang_vel["x"], label="Actual")
    axs_rates[0].plot(timestamps_cmd, roll_rate, label="Desired", linestyle='--')
    axs_rates[0].set_ylabel("Roll Rate [deg/s]")
    axs_rates[0].set_ylim(-250, 250)
    axs_rates[0].legend()
    axs_rates[0].grid(True)

    axs_rates[1].plot(timestamps, gt_ang_vel["y"], label="Actual")
    axs_rates[1].plot(timestamps_cmd, pitch_rate, label="Desired", linestyle='--')
    axs_rates[1].set_ylabel("Pitch Rate [deg/s]")
    axs_rates[1].set_ylim(-250, 250)
    axs_rates[1].legend()
    axs_rates[1].grid(True)

    axs_rates[2].plot(timestamps, gt_ang_vel["z"], label="Actual")
    axs_rates[2].plot(timestamps_cmd, yaw_rate, label="Desired", linestyle='--')
    axs_rates[2].set_xlabel("Time [s]")
    axs_rates[2].set_ylabel("Yaw Rate [deg/s]")
    axs_rates[2].set_ylim(-250, 250)
    axs_rates[2].legend()
    axs_rates[2].grid(True)

    fig_rates.tight_layout(rect=[0, 0.03, 1, 0.97])
    rates_filename = output_plot_dir / f"{namespace}_angular_velocity_comparison"
    try:
        print(f"Saving angular velocity comparison plot to {rates_filename}")
        fig_rates.savefig(rates_filename.with_suffix('.eps'), format='eps', bbox_inches='tight', dpi=300, facecolor=fig_rates.get_facecolor())
        fig_rates.savefig(rates_filename.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300, facecolor=fig_rates.get_facecolor())
    except Exception as e:
        print(f"Error saving angular velocity plot: {e}")
  else:
      print("Skipping angular velocity comparison plot (no command data found/processed).")

  # --- CTBR data plot ---
  if timestamps_cmd:
    fig_ctbr, axs_ctbr = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig_ctbr.suptitle(f"Commanded CTBR Data ({namespace})")

    axs_ctbr[0].plot(timestamps_cmd, thrust_pwm)
    axs_ctbr[0].set_ylabel("Thrust PWM")
    axs_ctbr[0].grid(True)

    axs_ctbr[1].plot(timestamps_cmd, thrust_N)
    axs_ctbr[1].set_ylabel("Thrust [N]")
    axs_ctbr[1].grid(True)

    axs_ctbr[2].plot(timestamps_cmd, roll_rate, label="Roll Rate Cmd")
    axs_ctbr[2].plot(timestamps_cmd, pitch_rate, label="Pitch Rate Cmd")
    axs_ctbr[2].plot(timestamps_cmd, yaw_rate, label="Yaw Rate Cmd")
    axs_ctbr[2].set_xlabel("Time [s]")
    axs_ctbr[2].set_ylabel("Commanded Rates [deg/s]")
    axs_ctbr[2].set_ylim(-250, 250)
    axs_ctbr[2].legend()
    axs_ctbr[2].grid(True)

    fig_ctbr.tight_layout(rect=[0, 0.03, 1, 0.97])
    ctbr_filename = output_plot_dir / f"{namespace}_ctbr_commands"
    try:
        print(f"Saving CTBR command plot to {ctbr_filename}")
        fig_ctbr.savefig(ctbr_filename.with_suffix('.eps'), format='eps', bbox_inches='tight', dpi=300, facecolor=fig_ctbr.get_facecolor())
        fig_ctbr.savefig(ctbr_filename.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300, facecolor=fig_ctbr.get_facecolor())
    except Exception as e:
        print(f"Error saving CTBR plot: {e}")
  else:
    print("Skipping CTBR command plot (no command data found/processed).")

  # --- Observations ---
  times_pass_gates = []
  if timestamps_obs:
    fig_obs, axs_obs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig_obs.suptitle(f"Distance from next gate ({namespace})")

    axs_obs[0].plot(timestamps_obs, dist_next_gate)
    axs_obs[0].set_ylabel("m")
    axs_obs[0].grid(True)

    fig_obs.tight_layout(rect=[0, 0.03, 1, 0.97])
    obs_filename = output_plot_dir / f"{namespace}_obs_commands"

    # Detect gate passing times
    for i in range(1, len(dist_next_gate)):
        if abs(dist_next_gate[i-1] - dist_next_gate[i]) > 0.3:
            times_pass_gates.append(timestamps_obs[i])

    try:
        print(f"Saving obs command plot to {obs_filename}")
        fig_obs.savefig(obs_filename.with_suffix('.eps'), format='eps', bbox_inches='tight', dpi=300, facecolor=fig_obs.get_facecolor())
        fig_obs.savefig(obs_filename.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300, facecolor=fig_obs.get_facecolor())
    except Exception as e:
        print(f"Error saving obs plot: {e}")
  else:
    print("Skipping obs command plot (no command data found/processed).")

  # --- Trajectory 3D plot ---
  fig3d = plt.figure(figsize=figsize)
  ax3d = fig3d.add_subplot(111, projection='3d')
  # Plot actual trajectory only if position data exists
  if gt_pos["x"]:
       ax3d.plot(gt_pos["x"], gt_pos["y"], gt_pos["z"], label="Actual Trajectory")

  # Plot desired trajectory if available and valid
  if timestamps_traj and traj["x"].ndim == 2 and traj["x"].shape[0] > 0:
       ax3d.plot(traj["x"][:, 0], traj["x"][:, 1], traj["x"][:, 2], label="Desired Trajectory", linestyle='--', color='orange')

  # Plot red points at gate intersection times
  if gt_pos["x"] and timestamps and times_pass_gates:
      t_array = np.array(timestamps)
      x_array = np.array(gt_pos["x"])
      y_array = np.array(gt_pos["y"])
      z_array = np.array(gt_pos["z"])

      for i, t_gate in enumerate(times_pass_gates):
          idx = np.argmin(np.abs(t_array - t_gate))
          ax3d.scatter(x_array[idx], y_array[idx], z_array[idx], color='red', s=50, marker='o',
                       label="Gate pass" if i == 0 else None)

  square_color = 'cyan'
  square_alpha = 0.3
  for square in verts_all:
      poly = Poly3DCollection([square], color=square_color, alpha=square_alpha, edgecolor='k')
      ax3d.add_collection3d(poly)

  # Draw walls
#   draw_walls(ax3d)

  ax3d.set_xlabel("x [m]")
  ax3d.set_ylabel("y [m]")
  ax3d.set_zlabel("z [m]")
  ax3d.set_title(f"3D Position Trajectory ({namespace})")
  ax3d.legend()
  # Apply equal scaling only if there's data plotted
  if gt_pos["x"] or (timestamps_traj and traj["x"].ndim == 2 and traj["x"].shape[0] > 0):
      set_axes_equal(ax3d) # Apply equal scaling after plotting

  traj3d_filename = output_plot_dir / f"{namespace}_trajectory_3d"
  try:
      print(f"Saving 3D trajectory plot to {traj3d_filename}")
      fig3d.savefig(traj3d_filename.with_suffix('.eps'), format='eps', bbox_inches='tight', dpi=300, facecolor=fig3d.get_facecolor())
      fig3d.savefig(traj3d_filename.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300, facecolor=fig3d.get_facecolor())
  except Exception as e:
      print(f"Error saving 3D trajectory plot: {e}")


  # --- Animation ---
  print("\nGenerating animation...")
  # Need timestamps AND valid euler angles (check if list isn't just NaNs)
  if not timestamps or not gt_euler["roll"] or all(np.isnan(gt_euler["roll"])):
        print("Skipping animation generation due to missing timestamps or valid orientation data.")
  else:
      def rotate_points(points, roll, pitch, yaw):
        # Ensure angles are valid numbers before attempting rotation
        if any(np.isnan([roll, pitch, yaw])):
            # print(f"Warning: NaN Euler angle encountered. Using identity rotation.") # Optional warning
            return points # Return original points if rotation is not possible
        try:
            # Filter out potentially huge angle values if they are unrealistic outliers
            # if abs(roll)>360*5 or abs(pitch)>360*5 or abs(yaw)>360*5:
            #     print(f"Warning: Large Euler angle encountered ({roll:.1f}, {pitch:.1f}, {yaw:.1f}). Using identity.")
            #     return points

            r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            return r.apply(points)
        except ValueError as e:
             # This might happen with extreme gimbal lock angles, though 'xyz' is usually robust
             print(f"Warning: Skipping rotation due to invalid Euler angles for SciPy: roll={roll}, pitch={pitch}, yaw={yaw} ({e})")
             return points # Return original points if rotation fails

      frame_idx = 0

      def update_animation(_):
        nonlocal paused, frame_idx, drone_quiver

        if paused or frame_idx >= len(timestamps):
            return drone_x, drone_y, trail, current_pos_marker, drone_quiver

        x = gt_pos["x"][frame_idx]
        y = gt_pos["y"][frame_idx]
        z = gt_pos["z"][frame_idx]
        roll = gt_euler["roll"][frame_idx]
        pitch = gt_euler["pitch"][frame_idx]
        yaw = gt_euler["yaw"][frame_idx]

        drone_size = 0.1
        base_points = np.array([
            [-drone_size, -drone_size, 0], [drone_size, drone_size, 0],
            [-drone_size, drone_size, 0], [drone_size, -drone_size, 0]
        ])
        rotated_points = rotate_points(base_points, roll, pitch, yaw)

        drone_x.set_data(
            [x + rotated_points[0, 0], x + rotated_points[1, 0]],
            [y + rotated_points[0, 1], y + rotated_points[1, 1]]
        )
        drone_x.set_3d_properties(
            [z + rotated_points[0, 2], z + rotated_points[1, 2]]
        )

        drone_y.set_data(
            [x + rotated_points[2, 0], x + rotated_points[3, 0]],
            [y + rotated_points[2, 1], y + rotated_points[3, 1]]
        )
        drone_y.set_3d_properties(
            [z + rotated_points[2, 2], z + rotated_points[3, 2]]
        )

        trail.set_data(gt_pos["x"][:frame_idx + 1], gt_pos["y"][:frame_idx + 1])
        trail.set_3d_properties(gt_pos["z"][:frame_idx + 1])

        current_pos_marker.set_data([x], [y])
        current_pos_marker.set_3d_properties([z])

        x_dir_body = rotate_points(np.array([[0.2, 0, 0]]), roll, pitch, yaw)[0]
        drone_quiver.remove()
        drone_quiver = anim_ax.quiver(
            x, y, z,
            x_dir_body[0], x_dir_body[1], x_dir_body[2],
            color='red', length=0.2, normalize=True
        )

        anim_ax.set_title(f"Drone Trajectory Animation ({namespace}) - Time: {timestamps[frame_idx]:.2f}s")

        frame_idx += 1
        return drone_x, drone_y, trail, current_pos_marker, drone_quiver


      # --- Set up Animation Plot ---
      anim_fig = plt.figure(figsize=figsize)
      def on_key(event):
        nonlocal paused
        if event.key == ' ':
          paused = not paused
          print("Paused" if paused else "Resumed")
      anim_fig.canvas.mpl_connect('key_press_event', on_key)

      anim_ax = anim_fig.add_subplot(111, projection='3d')

      # Determine axis limits from the *entire* dataset (actual and desired if present)
      all_x = list(gt_pos["x"])
      all_y = list(gt_pos["y"])
      all_z = list(gt_pos["z"])
      if timestamps_traj and traj["x"].ndim == 2 and traj["x"].shape[0] > 0:
            all_x.extend(traj["x"][:, 0])
            all_y.extend(traj["x"][:, 1])
            all_z.extend(traj["x"][:, 2])

      # Filter out NaNs before finding min/max
      all_x = [v for v in all_x if np.isfinite(v)]
      all_y = [v for v in all_y if np.isfinite(v)]
      all_z = [v for v in all_z if np.isfinite(v)]

      if not all_x or not all_y or not all_z:
          print("Error: No valid position data available for setting animation axes limits.")
          # Set default limits if no data is valid
          min_x, max_x, min_y, max_y, min_z, max_z = -1, 1, -1, 1, 0, 1
      else:
          min_x, max_x = min(all_x), max(all_x)
          min_y, max_y = min(all_y), max(all_y)
          min_z, max_z = min(all_z), max(all_z)

      # Add padding to limits
      x_range = max(max_x - min_x, 0.1) # Ensure range is not zero
      y_range = max(max_y - min_y, 0.1)
      z_range = max(max_z - min_z, 0.1)
      padding_x = x_range * 0.1
      padding_y = y_range * 0.1
      padding_z = z_range * 0.1

      anim_ax.set_xlim(min_x - padding_x, max_x + padding_x)
      anim_ax.set_ylim(min_y - padding_y, max_y + padding_y)
      anim_ax.set_zlim(max(0, min_z - padding_z), max_z + padding_z) # Ensure z starts >= 0

      # Calculate a reasonable size for the drone marker based on overall trajectory size
      drone_size = max(x_range, y_range, z_range) # Use max range as a scale factor

      set_axes_equal(anim_ax) # Apply equal scaling

      anim_ax.set_xlabel("x [m]")
      anim_ax.set_ylabel("y [m]")
      anim_ax.set_zlabel("z [m]")

      # Plot desired trajectory statically if available
      if timestamps_traj and traj["x"].ndim == 2 and traj["x"].shape[0] > 0:
            anim_ax.plot(traj["x"][:, 0], traj["x"][:, 1], traj["x"][:, 2], label="Desired Traj.", linestyle='--', color='orange', alpha=0.7)

      # Plot the full actual trajectory statically and faintly
      anim_ax.plot(gt_pos["x"], gt_pos["y"], gt_pos["z"], 'b-', linewidth=1, alpha=0.3, label="Full Actual Traj.")

      # Initialize animated elements (lines/markers)
      # Trail will grow, others will move/rotate
      trail, = anim_ax.plot([], [], [], 'b-', linewidth=1.5, label="Actual Trail") # Growing trail
      drone_x, = anim_ax.plot([], [], [], 'r-', linewidth=3) # Drone X marker
      drone_y, = anim_ax.plot([], [], [], 'g-', linewidth=3) # Drone Y marker
      current_pos_marker, = anim_ax.plot([], [], [], 'ko', markersize=4, label='Current Pos') # Current position dot
      drone_quiver = anim_ax.quiver(0, 0, 0, 0, 0, 0, color='red', length=0.2)
      anim_ax.legend(fontsize='small')

      square_color = 'cyan'
      square_alpha = 0.3
      for square in verts_all:
        poly = Poly3DCollection([square], color=square_color, alpha=square_alpha, edgecolor='k')
        anim_ax.add_collection3d(poly)
      
      # Draw walls in animation
    #   draw_walls(anim_ax)

      # --- Calculate Animation Timing ---
      if len(timestamps) > 1:
          # Calculate median time difference between frames
          median_dt = np.median(np.diff(timestamps))
          # Ensure dt is positive and non-zero; default if calculation fails
          if not np.isfinite(median_dt) or median_dt <= 0:
              median_dt = 0.05 # Default to 50ms (20 FPS)
          interval_ms = max(1, int(median_dt * 1000)) # Interval in milliseconds
          fps = min(30, 1000.0 / interval_ms) # Calculate FPS, cap at 30 for GIF
      else:
          interval_ms = 50 # Default interval for single frame
          fps = 20 # Default FPS

      print(f"Animation settings: interval={interval_ms}ms, target save fps={fps:.1f}")

      num_frames = len(timestamps)

      paused = False

      # Create the animation object
      _ = animation.FuncAnimation(
          anim_fig,
          update_animation,
          frames=range(num_frames),
          interval=interval_ms,
          blit=False
      )

  # --- Show plots interactively at the end ---
  print("\nDisplaying plots...")
  plt.show()
  print("\nAnalysis complete.")
  plt.close('all') # Close all figures


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("\nUsage: python process_bag.py <bag_path_or_dir> <namespace> [t0] [tf]")
    print("  <bag_path_or_dir>: Path to the ROS2 bag directory or a specific .mcap/.db3 file.")
    print("  <namespace>:       The namespace of the vehicle (e.g., 'cf1', 'crazy_jirl_02').")
    print("  [t0]:              Optional start time in seconds relative to the bag start (default: 0).")
    print("  [tf]:              Optional end time in seconds relative to the bag start (default: inf).")
    print("\nExample (directory): python process_bag.py logs/20250403_test1 cf1")
    print("Example (mcap file): python process_bag.py logs/my_run.mcap cf2 5 25")
    print("Example (db3 file):  python process_bag.py logs/old_bag/data.db3 drone_A\n")
    sys.exit(1) # Exit if not enough arguments
  else:
    bag_path_arg = sys.argv[1]
    namespace_arg = sys.argv[2]
    t0_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    tf_arg = float(sys.argv[4]) if len(sys.argv) > 4 else float('inf')

    # --- Basic Validation ---
    # We now handle non-existent paths inside the main function's detection logic
    # if not Path(bag_path_arg).exists():
    #      print(f"Error: Bag path '{bag_path_arg}' not found.")
    #      sys.exit(1)

    if tf_arg <= t0_arg:
        print(f"Error: End time (tf={tf_arg}) must be greater than start time (t0={t0_arg}).")
        sys.exit(1)

    # Call the main analysis function
    analyze_ros2_bag(bag_path_arg, namespace_arg, t0_arg, tf_arg)