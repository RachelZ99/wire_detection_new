---
status: proposed
---

# Use an explicit provisional obstacle-response input state machine

For offline replay and isolated ROS-domain pre-integration, the adapter uses an
explicit input state machine:

- fresh `HEALTHY`, matching profile, valid age: replace the confirmed snapshot;
  an explicit empty snapshot may clear it;
- fresh `DEGRADED`, matching profile, valid age: forward non-empty confirmed
  snapshots while marking the source degraded; do not clear from an empty
  snapshot;
- `INVALID`, stale/missing health, profile mismatch, speed above 0.3 m/s, or
  stale/future observation: block new snapshots and preserve the last snapshot;
- silence on the cloud topic does not clear anything;
- recovery after health-liveness loss, or a health heartbeat clock regression,
  starts a new source generation and the first subsequently accepted snapshot
  becomes that generation's snapshot.

The configured 600 ms health timeout and 2.5 s maximum confirmed observation
age are provisional integration values. The latter covers the current two
second perception retention interval plus transport/dispatch allowance. They
are not validated stopping-policy values.

Before physical use, the product and safety owners must approve or replace:
whether `DEGRADED` may admit new confirmed observations, when degraded clearing
is permitted, how long the unified layer retains an unavailable source, and
which robot response (`stop`, `avoid`, or `reject`) corresponds to every source
state. The perception adapter does not make those motion-policy decisions.
