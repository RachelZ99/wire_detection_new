# Low-Profile Hazard Perception

This context defines the safety concepts used to detect near-floor hazards in a factory and represent them for robot motion planning.

## Language

**Low-profile hazard**:
An object or structure close to the floor that can make robot passage unsafe through collision, wheel entanglement, snagging, dragging, or loss of traction.
_Avoid_: Low obstacle, floor object

**Cable hazard**:
A low-profile hazard formed by one or more cables, including flush, curved, crossed, stacked, or locally raised cable segments. Its risk is not determined by height alone.
_Avoid_: Cable bump, thin obstacle

**Observation blind zone**:
The near-floor region that the mounted camera cannot currently observe because of its field of view, minimum range, or robot-body occlusion. Lack of a current observation in this region is not evidence that it is safe.
_Avoid_: Free zone, dead zone

**Temporal observation alignment**:
Transforming hazard observations from different capture times into one local coordinate frame using the robot's motion estimate. This term does not mean generating odometry by fusing wheel, IMU, or visual sensors.
_Avoid_: Multi-frame fusion, odometry fusion

**Capture time**:
The frame exposure time expressed in the robot's monotonic clock domain after accounting for camera clock offset and drift. Host receipt time is not capture time.
_Avoid_: Arrival time, callback time, raw device timestamp

**Nominal camera pose**:
The camera mounting transform published through TF or written in configuration. It is an initial estimate and must not be treated as the measured floor relationship when mounting height or pitch can change.
_Avoid_: Calibrated ground plane, measured installation

**Observed ground model**:
The robustly fitted floor plane and its quality metrics derived from current valid depth samples. It is used to measure relative obstacle height and to detect disagreement with the nominal camera pose.
_Avoid_: Fixed camera height, hard-coded horizon

**Detection profile**:
A validated set of perception parameters for a bounded combination of cable appearance, floor, lighting, camera settings, and operating distance. A detection profile is not a general cable detector.
_Avoid_: Wire model, universal configuration

**Protective slowdown**:
A reversible safety response that reduces robot speed when an observation indicates a plausible low-profile hazard but has not yet accumulated enough evidence to stop or replan.
_Avoid_: Emergency stop, confirmed obstacle

**Hazard observation**:
A timestamped perception result that locates a suspected or confirmed low-profile hazard and reports its evidence and quality. It is not a motion command.
_Avoid_: Stop command, avoidance action

**Hazard response**:
The downstream decision to slow, stop, or replan after receiving a hazard observation. Hazard response is outside the low-profile perception boundary.
_Avoid_: Detection, classification

**Home feasibility prototype**:
A controlled residential evaluation used to prove the sensing, geometry, timing, and deployment pipeline before factory data exists. Its results do not establish factory safety performance.
_Avoid_: Product validation, factory pilot

**Factory validation**:
An evaluation on production-representative factory scenes that were not used to tune the detector. Passing factory validation is required before making operational safety claims.
_Avoid_: Home test, development replay
