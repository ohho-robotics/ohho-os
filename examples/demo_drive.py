#!/usr/bin/env python3
"""OhhO OS one-argument switch demo: sim:// vs ros2://.

Examples::

    # In-process simulator (native runtime)
    python examples/demo_drive.py --transport sim://

    # ROS 2 stack / Gazebo / fake robot (ros2 runtime, same script)
    python examples/demo_drive.py --transport ros2://
"""

from __future__ import annotations

from ohho.demo_drive import main

if __name__ == "__main__":
    raise SystemExit(main())
