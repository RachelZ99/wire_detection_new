#!/usr/bin/env bash
set -eo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble was not found at /opt/ros/humble" >&2
  exit 2
fi

if [[ ! "${ROS_DOMAIN_ID:-}" =~ ^[1-9][0-9]{0,2}$ ]] \
  || (( ROS_DOMAIN_ID > 232 )) \
  || [[ "${ROS_LOCALHOST_ONLY:-}" != "1" ]]; then
  echo "Set a non-default ROS_DOMAIN_ID (1-232) and ROS_LOCALHOST_ONLY=1 before running" >&2
  exit 4
fi

cd "$workspace_root"
source /opt/ros/humble/setup.bash
set -u

colcon build \
  --symlink-install \
  --packages-select low_profile_hazard_perception
set +u
source install/setup.bash
set -u
colcon test \
  --packages-select low_profile_hazard_perception \
  --python-testing pytest \
  --event-handlers console_direct+
colcon test-result --verbose
