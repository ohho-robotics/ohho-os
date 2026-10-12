#!/usr/bin/env python3
"""Lightweight fake robot for CI: integrate /cmd_vel → publish /odom.

No Gazebo required. Run inside a sourced ROS 2 Jazzy environment::

    source /opt/ros/jazzy/setup.bash
    python examples/fake_ros2_robot_node.py
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header


class FakeRos2Robot(Node):
    def __init__(self) -> None:
        super().__init__("ohho_fake_robot")
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._last_cmd = self.get_clock().now()
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self._pub = self.create_publisher(Odometry, "/odom", 10)
        self.create_timer(0.02, self._tick)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._vx = float(msg.linear.x)
        self._vy = float(msg.linear.y)
        self._omega = float(msg.angular.z)
        self._last_cmd = self.get_clock().now()

    def _tick(self) -> None:
        now = self.get_clock().now()
        dt = 0.02
        # Stop if no recent command (matches teleop safety posture).
        if (now - self._last_cmd).nanoseconds > 300_000_000:
            self._vx = self._vy = self._omega = 0.0
        cos_t = math.cos(self._theta)
        sin_t = math.sin(self._theta)
        self._x += (cos_t * self._vx - sin_t * self._vy) * dt
        self._y += (sin_t * self._vx + cos_t * self._vy) * dt
        self._theta += self._omega * dt
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.orientation.z = math.sin(self._theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self._theta / 2.0)
        msg.twist.twist.linear.x = self._vx
        msg.twist.twist.linear.y = self._vy
        msg.twist.twist.angular.z = self._omega
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = FakeRos2Robot()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
