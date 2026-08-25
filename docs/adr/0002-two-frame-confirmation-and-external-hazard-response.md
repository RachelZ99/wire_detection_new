---
status: accepted
---

# Confirm in two frames and leave hazard response downstream

The robot operates at 0.3 m/s with a 10 fps camera, so the perception system will confirm a hazard after two consistent observations. It will publish the hazard observation and health state but will not issue slowdown, stop, or replanning commands; the robot's unified obstacle-response logic owns those actions.
