#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import random
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_solution import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab10_solution import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 3 beams (-15°, 0°, +15° in the robot frame) from /scan
        to update the particle filter via its measurement model.
        """
        if self.laserscan is None:
            return

        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        mid_idx = num_ranges // 2
        def deg_to_offset(deg: float) -> int:
            return int(deg / (angle_increment * 180.0 / math.pi))

        offsets = [deg_to_offset(d) for d in (-45.0, -15.0, 0.0, 15.0, 45.0)]
        indices = [max(0, min(num_ranges - 1, mid_idx + off)) for off in offsets]
        measurements = []

        for idx in indices:
            z = ranges[idx]
            if z == inf or np.isinf(z):
                if hasattr(self.laserscan, "range_max"):
                    z = self.laserscan.range_max
                else:
                    z = 10.0  # fallback
            angle = angle_min + idx * angle_increment  # angle in robot frame
            measurements.append((z, angle))

        for z, a in measurements:
            self._pf.measure(z, a)

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
        - If front is free, go forward.
        - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """
        rate = rospy.Rate(1.0)  # ~1 Hz loop
        steps = 0

        while steps < max_steps and not rospy.is_shutdown():
            # ---- 1) Get current scan and decide motion ----
            scan = self.laserscan
            if scan is None or len(scan.ranges) == 0:
                # nothing yet: just spin a bit
                self.rotate_in_place(math.radians(30))
                steps += 1
                rate.sleep()
                continue

            ranges = scan.ranges
            n = len(ranges)
            angle_min = scan.angle_min
            angle_inc = scan.angle_increment

            def get_range_at_angle(rel_angle: float) -> float:
                """
                Return the range at a desired angle in the ROBOT frame (0 = straight ahead),
                using angle_min and angle_increment from LaserScan.
                """
                idx = int(round((rel_angle - angle_min) / angle_inc))
                idx = max(0, min(n - 1, idx))
                d = ranges[idx]
                if np.isinf(d) or np.isnan(d):
                    # Treat "no return" as max-range
                    if hasattr(scan, "range_max") and scan.range_max > 0:
                        d = scan.range_max
                    else:
                        d = 10.0
                return d

            d_front = get_range_at_angle(0.0)
            d_left  = get_range_at_angle(pi / 2.0)

            # Thresholds (tweak if needed)
            clear_thresh  = 0.7   # "enough space ahead" to step forward
            danger_thresh = 0.3   # "too close" -> back up before turning

            if d_front > clear_thresh:
                # free ahead → move forward
                self.move_forward(0.35)
            else:
                # Something is in front
                if d_front < danger_thresh:
                    self.move_forward(-0.25)

                # turn toward the side with more free space
                if d_left > d_front:
                    turn_angle = +math.pi / 3.0
                else:
                    turn_angle = -math.pi / 3.0

                self.rotate_in_place(turn_angle)

            # ---- 2) After motion, take measurements & update PF ----
            self.take_measurements()

            # ---- 3) Check convergence based on spread + sensor consistency ----
            x_est, y_est, theta_est = self._pf.get_estimate()
            pts = np.array([[p.x, p.y] for p in self._pf._particles])
            if pts.shape[0] > 0:
                dists = np.linalg.norm(pts - np.array([x_est, y_est]), axis=1)
                std_dev = float(np.std(dists))
            else:
                std_dev = float("inf")

            # Actual front range from lidar
            front_range = None
            if self.laserscan is not None:
                scan = self.laserscan
                angle_min = scan.angle_min
                angle_inc = scan.angle_increment
                ranges = scan.ranges
                n = len(ranges)

                zero_idx = int(round((0.0 - angle_min) / angle_inc))
                zero_idx = max(0, min(n - 1, zero_idx))
                front_range = ranges[zero_idx]
                if np.isinf(front_range) or np.isnan(front_range):
                    front_range = None

            # Predicted front range from map at PF estimate
            sensor_ok = False
            predicted_front = None
            if front_range is not None:
                predicted_front = self._pf.map_.closest_distance(
                    (x_est, y_est), theta_est
                )
                if predicted_front is None:
                    predicted_front = 10.0
                if abs(predicted_front - front_range) < 0.25:
                    sensor_ok = True

            rospy.loginfo(
                f"[PF] step={steps}, std_dev={std_dev:.3f}, "
                f"front={front_range}, pred_front={predicted_front}"
            )

            if std_dev < 0.18 and sensor_ok and steps > 20:
                rospy.loginfo(
                    f"PF converged after {steps} steps: "
                    f"x={x_est:.2f}, y={y_est:.2f}, th={theta_est:.2f}"
                )
                break

            steps += 1
            rate.sleep()

        # Final estimate log
        est_x, est_y, est_th = self._pf.get_estimate()
        rospy.loginfo(
            f"Localization finished (steps={steps}): "
            f"x={est_x:.2f}, y={est_y:.2f}, th={est_th:.2f}"
        )


        ######### Your code ends here #########

        

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        # Get current PF estimate for start position
        x, y, _ = self._pf.get_estimate()
        start = {"x": float(x), "y": float(y)}
        goal = {"x": float(self.goal_position["x"]), "y": float(self.goal_position["y"])}

        rospy.loginfo(
            f"Planning with RRT from start=({start['x']:.2f}, {start['y']:.2f}) "
            f"to goal=({goal['x']:.2f}, {goal['y']:.2f})"
        )

        plan, graph = self._planner.generate_plan(start, goal)
        self.plan = plan

        if not plan:
            rospy.logwarn("RRT failed to find a path (empty plan).")
            return

        # Visualize for RViz
        self._planner.visualize_plan(plan)
        self._planner.visualize_graph(graph)
        rospy.loginfo("RRT found a plan with %d waypoints.", len(plan))

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
    #Follow the RRT waypoints using PID on (distance, heading) error.

    ######### Your code starts here #########
        if not self.plan:
            rospy.logwarn("No plan to follow.")
            return

        local_goal_threshold = 0.1  #  cm

        # 1) Pick starting waypoint = closest to current PF estimate
        if self.current_position is not None:
            x0 = self.current_position["x"]
            y0 = self.current_position["y"]
        else:
            x0, y0, _ = self._pf.get_estimate()

        dists = [math.hypot(wp["x"] - x0, wp["y"] - y0) for wp in self.plan]
        self.current_wp_idx = int(np.argmin(dists))

        rospy.loginfo(
            "Following plan starting from waypoint %d/%d",
            self.current_wp_idx + 1, len(self.plan)
        )

        while self.current_wp_idx < len(self.plan) and (not rospy.is_shutdown()):
            # 2) Update PF with latest measurements
            self.take_measurements()

            if self.current_position is None:
                self.cmd_pub.publish(Twist())
                self.rate.sleep()
                continue
        
            x = self.current_position["x"]
            y = self.current_position["y"]
            theta = self.current_position["theta"]

            # 3) Get current target waypoint
            wp = self.plan[self.current_wp_idx]
            dx = wp["x"] - x
            dy = wp["y"] - y

            distance_error = math.hypot(dx, dy)
            desired_theta = math.atan2(dy, dx)
            heading_error = angle_to_neg_pi_to_pi(desired_theta - theta)

            # 4) Advance waypoint if close enough 
            while (
                self.current_wp_idx < len(self.plan) - 1
                and distance_error < local_goal_threshold
            ):
                self.current_wp_idx += 1
                wp = self.plan[self.current_wp_idx]
                dx = wp["x"] - x
                dy = wp["y"] - y
                distance_error = math.hypot(dx, dy)
                desired_theta = math.atan2(dy, dx)
                heading_error = angle_to_neg_pi_to_pi(desired_theta - theta)

            # If we're at the final waypoint and close enough, we're done
            if self.current_wp_idx == len(self.plan) - 1 and distance_error < local_goal_threshold:
                break

            t_now = rospy.get_time()
            linear_cmd = self.linear_pid.control(distance_error, t_now)
            angular_cmd = self.angular_pid.control(heading_error, t_now)
            
            heading_slow  = math.radians(35)   # start slowing when misaligned by > 35°
            heading_stop  = math.radians(60)   # turn in place when misaligned by > 60°


            heading_slow = math.radians(35)   # start slowing when > 35°
            heading_stop = math.radians(60)   # turn in place when > 60°

            # Gate **using the one PID output**
            if abs(heading_error) > math.radians(90) and distance_error > 0.3:
                linear_cmd = 0.0
            elif abs(heading_error) > math.radians(45):
                linear_cmd *= 0.4
            
            scan = self.laserscan
            if scan is not None and len(scan.ranges) > 0:
                angle_min = scan.angle_min
                angle_inc = scan.angle_increment
                ranges = scan.ranges
                n = len(ranges)

                def get_range_at_angle(rel_angle: float) -> float:
                    idx = int(round((rel_angle - angle_min) / angle_inc))
                    idx = max(0, min(n - 1, idx))
                    d = ranges[idx]
                    if np.isinf(d) or np.isnan(d):
                        if hasattr(scan, "range_max") and scan.range_max > 0:
                            d = scan.range_max
                        else:
                            d = 10.0
                    return d

                # Distances in robot frame
                d_front = get_range_at_angle(0.0)
                d_left  = get_range_at_angle(+math.pi / 4.0)   # ~45°
                d_right = get_range_at_angle(-math.pi / 4.0)   # ~-45°

                wall_slow = 0.45   # start slowing here
                wall_stop = 0.25   # emergency stop distance
                side_thresh = 0.25 # how close is "too close" on the sides

                # 1) Front obstacle handling
                if d_front < wall_stop:
                    # Very close to a wall in front: stop and turn toward open side
                    linear_cmd = 0.0
                    if d_left > d_right:
                        angular_cmd = max(angular_cmd, 0.8)   # turn left
                    else:
                        angular_cmd = min(angular_cmd, -0.8)  # turn right
                elif d_front < wall_slow:
                    # Scale down linear speed smoothly as we approach the wall
                    factor = (d_front - wall_stop) / (wall_slow - wall_stop)
                    factor = max(0.0, min(1.0, factor))
                    linear_cmd *= factor

                # 2) Side "push" away from nearby walls
                if d_left < side_thresh < d_right:
                    angular_cmd += 0.3   # steer right
                elif d_right < side_thresh < d_left:
                    angular_cmd -= 0.3   # steer left

            linear_cmd = max(self.linear_pid.u_min, min(self.linear_pid.u_max, linear_cmd))
            angular_cmd = max(self.angular_pid.u_min, min(self.angular_pid.u_max, angular_cmd))
                
            # 5) Publish velocity command
            twist = Twist()
            twist.linear.x = linear_cmd
            twist.angular.z = angular_cmd
            self.cmd_pub.publish(twist)

            self.rate.sleep()

        # 6) Stop when done
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Finished following plan.")
    ######### Your code ends here #########


    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.3

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
