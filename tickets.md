# Tickets: RGB-D Low-Profile Hazard Perception

These tickets build the perception system specified in [the RGB-D low-profile hazard PRD](.scratch/rgbd-low-profile-hazard-perception/PRD.md).

Work the **frontier**: any ticket whose blockers are all complete. Tickets 3 and 4 can proceed independently after ticket 2. Ticket 8 is conditional and is only executed if ticket 7 selects the NPU path.

## 1. Establish deterministic RGB-D replay and health output

**What to build:** A runnable new-project skeleton that can replay the recorded RGB, depth, CameraInfo, TF, and odom streams through the highest test seam and publish an observable perception-health result. Selectively reuse legacy build or test scaffolding only when it does not import the legacy synchronous architecture.

**Blocked by:** None — can start immediately.

- [ ] A documented command builds the project and runs its tests in the supported ROS 2 environment.
- [ ] The reference bag can be replayed repeatedly with deterministic input counts and health results.
- [ ] Health reports the delivered 640×360 image profile, encodings, approximate rates, frame IDs, and CameraInfo consistency.
- [ ] Health reports sensor-stamp age, receive age, missing streams, stale streams, and queue drops without conflating those measurements.
- [ ] Invalid image dimensions, stride, encoding, CameraInfo, or TF produce an explicit degraded/invalid state.
- [ ] No operational hazard point cloud is published merely because input transport is healthy.

## 2. Carry a strong geometric hazard through to an odom point cloud

**What to build:** On replay of the reference scene, robustly observe the floor, recognize an obvious protrusion such as a power strip, transform independently timestamped observations with interpolated odom, confirm the hazard twice, and publish it as an operational `odom`-frame point cloud.

**Blocked by:** 1. Establish deterministic RGB-D replay and health output.

- [ ] A valid depth frame produces an observed ground model with support, residual, spatial coverage, and consistency metrics.
- [ ] The observed ground model reflects the measured installation rather than blindly accepting the nominal 0.15 m TF height.
- [ ] Strong protrusions use robust local height support rather than one maximum depth point.
- [ ] Two spatially consistent observations confirm the reference power strip; one isolated observation does not.
- [ ] Published operational points use the continuous `odom` frame and carry a meaningful observation timestamp.
- [ ] Replayed robot motion does not create an unaligned point-cloud trail.
- [ ] Empty reflective floor regions do not form a persistent confirmed geometric hazard in the reference replay.

## 3. Make degradation and blind-zone retention conservative

**What to build:** Preserve confirmed hazards while they pass into the measured near-field observation blind zone, and expose conservative health/output behavior when the floor, TF, odom, or sensor streams cannot support a safe interpretation.

**Blocked by:** 2. Carry a strong geometric hazard through to an odom point cloud.

- [ ] A confirmed hazard remains available for at least the configured two-second minimum after losing direct visibility.
- [ ] Candidate expiry and confirmed-hazard retention are distinct behaviors.
- [ ] Ground-model failure cannot clear an existing confirmed hazard or claim unobserved space is free.
- [ ] Missing, stale, disordered, or discontinuous odom prevents new cross-frame confirmation and produces a diagnostic reason.
- [ ] Missing/stale TF or CameraInfo prevents unsupported projection and produces a diagnostic reason.
- [ ] Competing planes, insufficient ground coverage, and large invalid-depth regions exercise deterministic degraded/invalid transitions.
- [ ] Recovery from a transient failure does not duplicate, teleport, or prematurely clear the retained hazard.

## 4. Carry a training-free RGB cable through to the odom point cloud

**What to build:** Detect visible pink and white floor cables using local thin-line structure rather than one fixed color, project their RGB evidence onto the observed floor, align observations in `odom`, and confirm them through the same operational output used by geometric hazards.

**Blocked by:** 2. Carry a strong geometric hazard through to an odom point cloud.

- [ ] Candidate generation is restricted to the observed floor and rejects hanging wires and background structure.
- [ ] Local contrast, thin-line or paired-edge response, physical span, width consistency, and curve continuity replace the legacy 5000-pixel area gate.
- [ ] Both the recorded pale cable and a white cable scene can form two spatially consistent observations without a trained model.
- [ ] A cable pixel without valid depth receives a conservative position through ray-ground intersection.
- [ ] Candidate masks preserve two-to-five-pixel structures without strong erosion or destructive resizing.
- [ ] Empty reflective floor, long shadows, floor seams, table/tripod legs, and cable reflections are represented in a negative replay set.
- [ ] Color-specific demo configuration remains available only as a diagnostic comparison and is not the formal detection path.

## 5. Fuse RGB, weak height, and invalid-depth evidence

**What to build:** Associate asynchronous RGB and depth observations in `odom`, combine strong geometry, weak height, RGB cable shape, and continuous invalid-depth evidence, and expose why each candidate was confirmed or rejected.

**Blocked by:** 3. Make degradation and blind-zone retention conservative; 4. Carry a training-free RGB cable through to the odom point cloud.

- [ ] RGB and depth retain their own sensor stamps; approximate image synchronization is not the primary processing gate.
- [ ] Strong geometry can confirm through repeated geometric observations without a semantic class.
- [ ] High-confidence RGB cable observations can confirm without requiring valid depth at every cable pixel.
- [ ] Weak height and continuous invalid depth can strengthen a spatially matching cable observation.
- [ ] Invalid depth without cable shape or valid geometric support cannot confirm a hazard.
- [ ] Each candidate reports evidence sources, ground-model quality, confidence, and a machine-readable decision reason.
- [ ] Candidate/debug topics remain separate from the operational confirmed point cloud.
- [ ] Mixed-evidence replay remains deterministic under different message arrival orders that preserve the same sensor stamps.

## 6. Bind detection profiles and meet the resource budget

**What to build:** Turn the working pipeline into a versioned detection profile with bounded queues, measured latency, stable resource use, and an explicit 0.3 m/s validity limit.

**Blocked by:** 5. Fuse RGB, weak height, and invalid-depth evidence.

- [ ] The initial profile records the delivered image profile, camera installation, footprint, thresholds, retention, model/rule version, and maximum validated speed.
- [ ] A changed image profile, 60 cm installation, or speed above 0.3 m/s cannot silently reuse the initial validated profile.
- [ ] Image and work queues drop old work instead of accumulating latency.
- [ ] Per-stage processing latency, end-to-end message age, frame drops, CPU, memory, and optional NPU state are observable.
- [ ] Perception processing meets the provisional P95 target of 80 ms excluding camera/driver transport, or produces an evidence-backed exception and revised budget.
- [ ] Depth geometry averages no more than approximately one RK3588 A76 core under the reference workload, or produces an evidence-backed optimization ticket.
- [ ] A two-hour replay/soak test shows no unbounded queue or memory growth.
- [ ] Formal configuration no longer describes the delivered stream as 640×400 or the physical mount as a verified 0.15 m horizontal installation.

## 7. Build the home regression suite and decide the NPU gate

**What to build:** Evaluate the complete rule-based pipeline at event level across independent home scenes and make a recorded decision on whether NPU cable segmentation is necessary.

**Blocked by:** 6. Bind detection profiles and meet the resource budget.

- [ ] The suite covers multiple cable colors/materials, straight and curved layouts, crossings, partial occlusion, power strips, plugs, and raised sections.
- [ ] Difficult negatives include reflections, shadows, floor seams, scratches, tape, furniture legs, mat edges, and empty floor.
- [ ] Scenes used to tune parameters are separated from held-out acceptance scenes.
- [ ] Reports include event recall, persistent false events, confirmed detection distance, confirmation latency, health failures, and resource use.
- [ ] Results are stratified by distance, cable appearance, floor, light, robot motion, and depth validity.
- [ ] Full-speed 0.3 m/s straight and turning recordings supplement the initial bag, which only reached roughly 0.14 m/s.
- [ ] A decision record states either that the rule path passes the agreed home gate or identifies specific failure classes that justify NPU work.
- [ ] Home results are explicitly labeled feasibility evidence and not factory validation.

## 8. Optionally deploy binary cable segmentation on the RK3588 NPU

**What to build:** If ticket 7 identifies rule-based failure classes that require learning, deliver a binary `CABLE / BACKGROUND` segmentation implementation on RK3588 NPU and demonstrate an event-level improvement on the same held-out regression suite.

**Blocked by:** 7. Build the home regression suite and decide the NPU gate. Execute only when its decision selects the NPU path.

- [ ] Training examples are selected from measured failure classes rather than collected without a target.
- [ ] Video masks are propagated with an assisted annotation workflow and corrected on key frames.
- [ ] Train, validation, and test splits are separated by complete scene/video.
- [ ] Model input preserves 640×360 content without geometric stretching, and its effective output resolution preserves two-to-five-pixel cables.
- [ ] Quantized RKNN output is regression-tested against the pre-quantized model on held-out scenes.
- [ ] NPU inference meets the provisional P95 target of 30 ms without shifting the CPU budget materially.
- [ ] The same operational point-cloud and health interfaces work whether the RGB provider is rules, NPU, or unavailable.
- [ ] The NPU path improves the failure classes that justified it without introducing unacceptable persistent false hazards.

## 9. Integrate the unified obstacle response and validate the 0.3 m/s envelope

**What to build:** Connect confirmed hazard observations and health to the robot's existing unified obstacle-response logic, then validate detection, retention, stopping/avoidance, failure handling, and long-running behavior on the physical robot at the current operating limit.

**Blocked by:** 7. Build the home regression suite and decide the NPU gate; also 8. Optionally deploy binary cable segmentation on the RK3588 NPU if ticket 7 selects it.

- [ ] Perception publishes no direct slowdown, stop, emergency-stop, or replanning command.
- [ ] The downstream system consumes only confirmed operational observations and interprets health/age according to an agreed policy.
- [ ] Physical trials cover 0.3 m/s straight travel, turns, approach from multiple angles, acceleration, braking, and target entry into the near-field blind zone.
- [ ] Measured camera-to-response and braking distances fit within the available geometry plus the agreed safety margin.
- [ ] RGB, depth, CameraInfo, TF, odom, ground-model, and optional NPU failures produce the agreed conservative whole-robot response.
- [ ] Repeated power-strip and cable trials report event-level detection and whole-robot outcomes rather than only frame metrics.
- [ ] An eight-hour run shows no crash, resource growth, stale confirmed hazards, or silent health failure.
- [ ] The release report states that factory data and factory acceptance remain mandatory before industrial safety claims.
