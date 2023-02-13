from setuptools import setup

package_name = "image_object_detection"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name, "models", "utils", "models.experimental"],
    package_dir={"": "src", "models": "src/models", "utils": "src/utils"},
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["yolov7-tiny.pt"]),
        ("share/" + package_name + "/launch", ["launch/image_object_detection_launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pablo Iñigo Blasco",
    maintainer_email="pablo@ibrobotics.com",
    description="Object detection node using YOLOv7-Tiny",
    license="BSDv3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "image_object_detection_node = image_object_detection.image_object_detection_node:main"
        ],
    },
)
