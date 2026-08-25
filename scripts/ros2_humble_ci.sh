#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble was not found at /opt/ros/humble" >&2
  exit 2
fi

cd "$workspace_root"
source /opt/ros/humble/setup.bash

colcon build \
  --symlink-install \
  --packages-select low_profile_hazard_perception
source install/setup.bash
colcon test \
  --packages-select low_profile_hazard_perception \
  --event-handlers console_direct+
colcon test-result --verbose
