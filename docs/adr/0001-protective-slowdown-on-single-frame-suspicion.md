---
status: superseded by ADR-0002
---

# Slow down on a single-frame cable suspicion

At the 0.8 m/s operating speed and 10 fps camera rate, waiting for multi-frame confirmation consumes too much stopping distance. A single-frame cable suspicion in the far observation zone will therefore trigger a reversible protective slowdown; confirmation can then stop the robot or make the region impassable, while rejected suspicions recover speed conservatively.
