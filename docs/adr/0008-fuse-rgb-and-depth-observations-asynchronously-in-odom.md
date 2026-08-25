---
status: accepted
---

# Fuse RGB and depth observations asynchronously in odom

RGB cable evidence and depth geometric evidence will be processed at their own capture timestamps rather than requiring pixel-synchronous image pairs. Depth frames will produce ground-relative protrusion and invalid-depth observations. RGB cable masks will be projected onto the observed ground model to obtain conservative three-dimensional locations.

Each observation will then be transformed into `odom` at its own timestamp. Spatial association, evidence fusion, and two-observation confirmation will occur in `odom`.

This avoids discarding frames or introducing unmodelled spatial error merely to enlarge an approximate-time synchronizer window. In the first DCW2 bag, a 30 ms window covered about 60 percent of nearest RGB-depth pairs; a 50 ms window covered about 98 percent but could represent 15 mm of robot motion at 0.3 m/s.
