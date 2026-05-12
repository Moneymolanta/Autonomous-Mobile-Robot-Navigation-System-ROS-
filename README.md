# TurtleBot Autonomous Maze Navigation

This project combines probabilistic localization, autonomous path planning, and closed-loop waypoint navigation to enable a TurtleBot to fully navigate a maze environment using onboard sensors and ROS.

The system integrates three major robotics components into a single autonomous pipeline:

* **Particle Filter Localization** — estimates the robot’s pose within the maze using odometry and LiDAR measurements.
* **Rapidly-Exploring Random Tree (RRT) Planning** — generates a collision-free path from the estimated robot position to a goal location.
* **PID-based Waypoint Following** — drives the robot along the generated path while continuously updating localization estimates.

Using real-time LaserScan and odometry data, the robot autonomously explores the environment, converges on an estimated pose, plans a valid trajectory, and dynamically follows waypoints while avoiding nearby obstacles.

## Features

* ROS-based autonomous navigation pipeline
* Real-time particle filter localization
* Sensor fusion using LiDAR and odometry
* Autonomous exploration behavior for localization convergence
* RRT path planning in obstacle-constrained environments
* PID waypoint tracking controller
* Dynamic obstacle-aware motion adjustments
* RViz-compatible path and graph visualization

## Technologies Used

* Python
* ROS (Robot Operating System)
* TurtleBot
* OpenCV
* NumPy
* LiDAR / LaserScan
* Particle Filters
* RRT Motion Planning
* PID Control Systems

## System Pipeline

1. The robot explores the maze while updating particle weights using LiDAR measurements.
2. Once localization converges, the estimated pose is used as the starting point for RRT planning.
3. RRT generates a collision-free waypoint path to the goal.
4. A PID controller drives the robot through each waypoint while continuously refining localization.

This project demonstrates the integration of localization, planning, and control into a complete autonomous robotics navigation system.
