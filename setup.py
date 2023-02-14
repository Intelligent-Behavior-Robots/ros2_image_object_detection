from setuptools import setup
from setuptools import find_packages

package_name = "image_object_detection"

setup(
    name=package_name,
    version="1.0.0",
    packages= ["image_object_detection", "models", "utils"], # find_packages(exclude=["test"]),

    package_dir={"image_object_detection": "src/image_object_detection", "models": "src/models", "utils": "src/utils"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/ament_index/resource_index/packages", ["resource/models"]),
        ("share/ament_index/resource_index/packages", ["resource/utils"]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["yolov7-tiny.pt"]),
        ("share/" + package_name + "/launch", ["launch/image_object_detection_launch.py"]),
        ("share/" + package_name + "/config", ["config/image_object_detection.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pablo Iñigo Blasco",
    maintainer_email="pablo@ibrobotics.com",
    author="Pablo Iñigo Blasco",
    author_email="pablo@ibrobotics.com",
    description="Object detection node using YOLOv7-Tiny",
    license="BSDv3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "image_object_detection_node = image_object_detection.image_object_detection_node:main"
        ],
    },
)
