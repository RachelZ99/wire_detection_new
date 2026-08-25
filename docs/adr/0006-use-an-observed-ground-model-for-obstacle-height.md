---
status: accepted
---

# Use an observed ground model for obstacle height

Each usable depth frame will robustly fit an observed ground plane and smooth it over a short temporal window. Obstacle height will be measured relative to this observed model. The nominal TF camera pose will provide an initial estimate and an independent consistency check, but will not be treated as the floor truth.

Ground-model quality must be reported from inlier support, residual error, spatial coverage, and temporal consistency. When quality is insufficient, perception will publish a degraded diagnostic and will not interpret unobserved or invalid-depth regions as free space.

This decision is supported by the first DCW2 bag: TF declared a 0.15 m horizontal mount, while repeated depth fits and physical measurement agreed on an approximately 0.22–0.23 m mount with about 2.5–3 degrees of downward pitch.
