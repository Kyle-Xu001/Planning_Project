## Quadrotor Planning (RO47005 Project)

### Simulation environment
This project is built base on this [simulation environment](https://github.com/Bharath2/Quadrotor-Simulation).  The already built functionalities in this environment are:

- Informed-RRT* path planner.
- Minisnap trajectory generation without collision avoidance (solved by closed-form solution).
- A non-linear controller for path following.
- A quadrotor simulator.

### Our contribution
The **newly added** functionalities by us are:

- PRM* path planner.
- Collision free minisnap trajectory optimization using two methods, namely corridor bounding method and waypoint insertion method.
- A simplified version of Kinodynamics RRT*.
- Velocity Obstacles.
- Dynamic window approach.
- Random generated map for simulation.

### Dependencies
numpy, matplotlib, rtree-linux, scipy, CVXPY, CVXOPT

### Usage
#### K-PRM* with Minisnap and Corridor Bounding Method
```
python runsim_prm.py
```
#### Kinodynamics RRT*
```
python runsim_kinorrt.py
```
