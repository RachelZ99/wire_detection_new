from glob import glob
from setuptools import find_packages, setup


package_name = "low_profile_hazard_perception"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml") + glob("config/*.json"),
        ),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Low-profile perception maintainers",
    maintainer_email="maintainers@example.invalid",
    description=(
        "Asynchronous RGB-D health and odom-aligned low-profile hazards."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "input_health_node = low_profile_hazard_perception.node:main",
            "geometric_hazard_node = "
            "low_profile_hazard_perception.geometric_node:main",
            "replay_rgbd_health = low_profile_hazard_perception.replay:main",
            "replay_geometric_hazard = "
            "low_profile_hazard_perception.geometric_replay:main",
            "replay_rgb_cable = "
            "low_profile_hazard_perception.geometric_replay:main_rgb_cable",
            "soak_detection_profile = "
            "low_profile_hazard_perception.profile_soak:main",
        ],
    },
)
