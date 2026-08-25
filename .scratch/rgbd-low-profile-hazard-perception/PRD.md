# RGB-D Low-Profile Hazard Perception

Status: ready-for-agent

## Problem Statement

An indoor factory robot must detect low-profile hazards such as loose cables, power strips, plugs, thin boxes, and raised mat edges early enough for the robot's unified obstacle-response logic to avoid them. These hazards are difficult because a cable can be only a few image pixels wide, can be nearly flush with the floor, and can produce invalid or noisy depth on reflective surfaces. A fixed color threshold only works for a controlled demonstration, while a depth-only height threshold misses thin cables.

The robot has a fixed DCW2 RGB-D camera without an IMU. The camera mount can be adjusted and its nominal TF may disagree with the physical installation. The current operating speed is 0.3 m/s with configured deceleration of 1 m/s². CPU use must remain low, although the RK3588 NPU is available. Perception must report hazards and health, but must not directly command slowdown, stopping, or replanning.

The current implementation was built around approximately synchronized RGB and depth images, color-specific masks, pixel-area thresholds, and camera-frame output. Recorded data shows that these assumptions do not hold: aligned output is 640×360@10 fps, RGB and depth timestamps often differ by more than 30 ms, RGB arrives roughly 180–218 ms after its message timestamp, odometry is approximately 25 Hz, and the physical camera pose differs from its nominal TF.

## Solution

Build an explainable, low-compute perception pipeline with independent RGB and depth evidence paths.

The depth path robustly estimates the currently observed floor, measures valid points relative to that floor, and detects generic protrusions without requiring semantic class names. The RGB path initially uses training-free, floor-constrained thin-line structure to propose cable observations. If this path cannot meet event-level recall and false-positive targets across independent scenes, replace or augment it with a small binary `CABLE / BACKGROUND` segmentation model running on the RK3588 NPU.

RGB and depth observations retain their own timestamps. Each is projected into three-dimensional space and transformed into the continuous `odom` frame using interpolated robot motion. Spatial association and two-observation confirmation happen in `odom`, avoiding the frame loss and motion error caused by forcing pixel-synchronous image pairs. Confirmed hazards are published as an `odom`-frame point cloud, while candidates and health remain separate diagnostic outputs.

The pipeline validates its input profile at runtime, estimates the ground from observations instead of hard-coding camera height, retains confirmed hazards while they cross the near-field observation blind zone, and reports degraded or invalid sensing rather than declaring unobserved space free.

## User Stories

1. As a robot operator, I want the robot to report a cable crossing its swept path, so that the unified obstacle-response logic can stop or route around it.
2. As a robot operator, I want power strips and plugs detected even when their color has never been seen before, so that safety does not depend on a class-specific model.
3. As a robot operator, I want thin white, black, gray, and colored cables handled by the same architecture, so that changing cable color does not require rewriting the system.
4. As a robot operator, I want curved, crossed, stacked, partially raised, and partially occluded cables represented by their floor occupancy, so that a bounding box does not distort the hazard shape.
5. As a robot operator, I want confirmed hazards retained after they enter the camera's near-field blind zone, so that they do not disappear immediately before the robot reaches them.
6. As a safety integrator, I want perception to publish observations rather than motion commands, so that all stopping and replanning policy remains in the unified obstacle-response layer.
7. As a safety integrator, I want every operational hazard message timestamped and located in `odom`, so that age and spatial consistency can be evaluated downstream.
8. As a safety integrator, I want candidate observations excluded from the operational point cloud until two independent observations agree, so that isolated image or depth noise does not become a navigation obstacle.
9. As a safety integrator, I want sensing failures reported as degraded or invalid rather than free space, so that missing evidence is never interpreted as proof of safety.
10. As a safety integrator, I want each detection profile bound to a validated speed, camera profile, and installation, so that a configuration validated at 0.3 m/s is not silently used at 0.8 m/s.
11. As a perception engineer, I want RGB and depth processed at their own timestamps, so that useful frames are not discarded by an arbitrary synchronizer window.
12. As a perception engineer, I want odometry interpolation for observation-time transforms, so that the measured 25 Hz odometry remains usable with 10 Hz cameras.
13. As a perception engineer, I want image width, height, stride, encoding, and intrinsics read from messages, so that algorithms do not hard-code a 400-row image.
14. As a perception engineer, I want startup validation to state the actual camera profile, so that requested and delivered stream formats cannot be confused.
15. As a perception engineer, I want a robust observed ground model per usable depth frame, so that obstacle height follows the real camera pose and floor rather than stale TF assumptions.
16. As a perception engineer, I want ground-model quality measured from support, residual, coverage, and temporal consistency, so that a wall or competing plane is rejected.
17. As a perception engineer, I want nominal TF compared with the observed ground model, so that loose mounts and stale calibration produce actionable diagnostics.
18. As a perception engineer, I want strong and weak height evidence kept separate, so that power strips can be confirmed geometrically without forcing the same threshold onto thin cables.
19. As a perception engineer, I want invalid depth preserved as a distinct evidence type, so that reflective-floor holes can assist a cable hypothesis without confirming hazards alone.
20. As a perception engineer, I want RGB cable candidates projected by ray-ground intersection when depth is absent, so that missing cable pixels still receive a conservative three-dimensional location.
21. As a perception engineer, I want the first RGB detector to use local contrast and thin-line structure rather than absolute pink thresholds, so that it can transfer across cable colors.
22. As a perception engineer, I want thin-line filtering constrained to the observed floor, so that hanging wires, tripod legs, and background edges are rejected.
23. As a perception engineer, I want length, width consistency, curvature, and physical span used instead of a large connected-component area threshold, so that two-to-five-pixel cables survive post-processing.
24. As a perception engineer, I want NPU segmentation introduced only after rule-based failure modes are measured, so that training work is driven by evidence rather than assumption.
25. As an ML engineer, I want the optional model to perform only binary cable segmentation, so that data requirements and deployment complexity remain small.
26. As an ML engineer, I want video-assisted annotation and failure-driven sampling, so that manual labeling effort is minimized.
27. As an ML engineer, I want training, validation, and test data separated by complete scene or video, so that adjacent frames cannot leak into evaluation.
28. As an embedded engineer, I want dense point-cloud publication avoided inside the processing pipeline, so that CPU and memory are spent only on the useful floor ROI.
29. As an embedded engineer, I want stale-frame queues bounded to one latest item, so that overload increases frame drops rather than decision latency.
30. As an embedded engineer, I want NPU inference asynchronous and optional, so that depth geometry remains operational if the model fails.
31. As a test engineer, I want deterministic rosbag replay from sensor topics to operational outputs, so that algorithm and parameter changes can be compared repeatably.
32. As a test engineer, I want event-level recall and sustained false hazards measured instead of only pixel accuracy, so that evaluation reflects robot behavior.
33. As a test engineer, I want results stratified by distance, cable appearance, floor, light, motion, and depth validity, so that average scores cannot hide a safety-relevant failure mode.
34. As a test engineer, I want injected RGB, depth, odom, TF, ground-model, and NPU failures, so that conservative degradation is verified explicitly.
35. As a maintainer, I want every observation to carry evidence and decision reasons, so that field failures can be diagnosed without guessing.
36. As a maintainer, I want detection profiles and thresholds versioned with code and model versions, so that a result can be reproduced.
37. As a project owner, I want a no-training home feasibility milestone before model development, so that the project proves the sensing and geometry pipeline quickly.
38. As a project owner, I want factory data to remain a mandatory release gate, so that home testing is not presented as industrial validation.

## Implementation Decisions

- The delivered sensor profile is discovered and validated at runtime. The initial validated profile is aligned RGB and `16UC1` depth at 640×360 and approximately 10 fps.
- Nominal TF is an initialization and consistency input, not the floor truth. Each usable depth frame produces an observed ground model with quality state.
- Ground estimation uses constrained robust fitting followed by refinement and short temporal smoothing. Initial thresholds are guided by the recorded bag and remain profile parameters.
- Generic geometry detects power strips, plugs, thin boxes, raised mat edges, and other low protrusions without assigning a semantic class.
- Strong geometric evidence begins near 15 mm above the observed ground. Weak evidence begins near the greater of three robust ground-noise deviations or approximately 6 mm and requires shape, RGB, or temporal support.
- Invalid depth is never a standalone confirmation source. It can support a floor-constrained RGB cable candidate or explain missing depth along a valid structure.
- The initial RGB path is training-free and uses floor restriction, local photometric contrast, multi-scale thin-line or paired-edge response, and curve-shape scoring.
- The color-specific pink profile is demonstration-only. Formal detection does not use one absolute cable color or a fixed large pixel-area threshold.
- An optional NPU model performs binary cable segmentation only. It preserves the 640×360 content without geometric stretching, uses thin-structure-friendly output resolution, and leaves the geometric path independent.
- RGB and depth observations are independent timestamped products. The architecture does not use approximate image synchronization as its primary processing seam.
- RGB cable pixels obtain a conservative position through ray intersection with the observed floor; depth geometric points use their measured three-dimensional position.
- Observation-time odom poses are interpolated from a bounded cache. Observations with missing, stale, or discontinuous pose support cannot enter cross-frame confirmation.
- Spatial association and confirmation happen in `odom`. Confirmation requires two independent, spatially consistent observations within a bounded time window.
- Confirmed hazards remain available long enough to cross the measured near-field observation blind zone; the initial minimum retention is two seconds.
- The operational output is a timestamped `sensor_msgs/PointCloud2` in continuous `odom`. Candidate masks, overlays, evidence, and health are separate diagnostic outputs.
- Perception publishes no slowdown, stop, or replanning commands. The unified obstacle-response layer owns those effects.
- The current detection profile is valid only up to 0.3 m/s. Higher speeds, a 60 cm camera installation, or a different delivered camera profile require a separate validation profile.
- Input and work queues are bounded and drop old frames. Every stage reports message age and processing latency.
- CPU processes a sparse floor ROI and candidate regions rather than materializing and repeatedly transforming a full dense cloud. NPU inference is asynchronous when enabled.
- The legacy implementation may contribute build scaffolding and proven mathematical/image utilities, but its synchronized node interface and color-specific behavior are not adopted as the new architecture.

## Testing Decisions

- The primary test seam is a black-box rosbag replay: publish RGB, depth, both CameraInfo streams, TF, and odom, then assert the operational `odom` point cloud, health state, confirmation timing, and obstacle retention behavior.
- Tests assert externally meaningful events rather than internal call sequences, private classes, exact RANSAC samples, or implementation-specific masks.
- A fixed home replay suite covers the existing reflective-floor power-strip/cable bag plus additional empty-floor, white-cable, colored-cable, shadow, reflection, floor-seam, and occlusion scenes.
- Replay results are deterministic for a fixed code, configuration, and model version. Each run records event detections, false persistent events, detection distance, message age, processing latency, and health transitions.
- Mathematical unit tests cover only seams whose correctness is easier to prove below the ROS graph: projection and coordinate conventions, ray-plane intersection, signed height, robust plane acceptance, odom interpolation, spatial association, and state expiry.
- Synthetic tests include plane noise, outliers, walls, competing planes, missing regions, timestamp disorder, odom gaps, and single-frame flying depth points.
- Integration tests verify that invalid ground, stale CameraInfo, stale TF, missing odom, delayed frames, and NPU failure cannot publish a newly confirmed operational hazard from invalid evidence or claim unobserved space is free.
- Motion replay verifies that independently stamped RGB and depth observations align in `odom` without point-cloud trails caused by direct frame accumulation.
- Near-field tests verify that a confirmed hazard remains published after it leaves the visible floor region and until the configured retention/clearing policy permits removal.
- Performance tests measure camera-to-recorder age separately from perception processing time. The initial processing target is P95 at or below 80 ms, excluding camera/driver transport.
- Resource tests target no queue growth, stable memory, and average depth-geometry CPU within approximately one RK3588 A76 core.
- The 0.3 m/s system gate uses physical robot trials to validate the complete perception-to-response stopping envelope. Calculated distance is supporting evidence, not a substitute for the trial.
- Rule-only feasibility does not require training data, but it does require independent replay scenes that were not used for tuning.
- Any learned model is evaluated on complete held-out scenes or videos. No adjacent frames from one video may cross train, validation, or test boundaries.
- Legacy C++ GTest patterns for synthetic images/depth and legacy deployment-level pytest patterns are prior art that may be selectively ported after their assumptions are updated.
- Factory release evaluation is event-based and stratified. Proposed initial targets are at least 99% for power strips/obvious protrusions, 95% for all cable hazard events, and 98% for stacked or raised cables within the required distance, subject to product and safety approval.

## Out of Scope

- Directly issuing slowdown, stop, emergency-stop, or replanning commands.
- Owning the unified obstacle-response policy or general Nav2 configuration.
- Safety certification based only on home testing.
- Guaranteeing performance on an unseen factory before factory data and acceptance testing exist.
- Distinguishing individual cable identities at crossings or counting cables.
- Finding cable endpoints, manipulating cables, or using a robotic arm to move them.
- Classifying every generic low-profile obstacle by semantic name.
- Outdoor operation, rain, standing water, grass, or unstructured terrain.
- Treating a 60 cm camera installation or 0.8 m/s operation as covered by the current detection profile.
- Replacing primary safety sensors or mechanical anti-entanglement protections.
- Large-scale manual labeling before rule-based failure modes justify a learned model.

## Further Notes

- Recorded evidence shows the physical camera is approximately 0.22–0.23 m above the floor with about 2.5°–3° downward pitch, despite nominal TF stating 0.15 m and no pitch.
- The current footprint extends to `x=0.375 m`, while the nearest visible floor is approximately `x=0.74 m`, producing about 0.365 m of near-field unobserved space ahead of the chassis.
- At 0.3 m/s, a conservative envelope containing RGB P95 age, worst-case two-frame wait, 100 ms external-response allowance, and 1 m/s² braking is approximately 0.20 m. The remaining theoretical margin is approximately 0.165 m and still requires whole-robot validation.
- The same assumptions at 0.8 m/s require approximately 0.734 m and are incompatible with the current detection profile.
- The first bag recorded odom near 25 Hz and robot motion below the 0.3 m/s maximum. Future recordings must include full-speed straight and turning tests.
- Home feasibility is the current phase. Factory validation is an explicit later release gate.
