---
status: accepted
---

# Segment cables and detect other hazards geometrically

The first NPU model will perform binary `CABLE / BACKGROUND` semantic segmentation. Power strips, thin boxes, raised mat edges, and other unknown low-profile protrusions will be detected by their geometry relative to the estimated floor, so safety does not depend on assigning every obstacle a known semantic class.
