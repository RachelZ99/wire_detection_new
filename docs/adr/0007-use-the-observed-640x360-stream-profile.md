---
status: accepted
---

# Use the observed 640x360 stream profile

The initial DCW2 development and validation baseline will use the stream profile observed in the recorded ROS 2 bag: aligned RGB and depth at 640x360 and approximately 10 frames per second. Perception code will read image dimensions, stride, encoding, and camera intrinsics from each ROS message and must not hard-code a 400-row image.

If a later driver or hardware profile produces 640x400, it will be treated as a separate detection profile and validated independently. Software will not stretch 640x360 images to claim a 640x400 sensor output.
