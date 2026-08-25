---
status: accepted
---

# Publish confirmed hazards as an odom-frame point cloud

After two odometry-aligned observations confirm a hazard, perception will publish its points as a timestamped `sensor_msgs/PointCloud2` in the continuous `odom` frame for the unified obstacle-response logic. Candidate observations remain diagnostic-only, perception health is published separately, and an unstamped boolean is not an operational navigation interface.
