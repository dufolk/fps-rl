"""Stereo placement: a source on the camera's right goes to the right ear."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl.render.audio import ear_offset  # noqa: E402


def test_sound_on_the_right_reaches_the_right_ear():
    # Camera looks along +Y; its right axis is +X. The lateral coordinate is negated.
    right_x, right_y, right_z = ear_offset((4.0, 1.0, 0.0), (0, 1, 0), (1, 0, 0), (0, 0, 1))
    left_x, left_y, left_z = ear_offset((-4.0, 1.0, 0.0), (0, 1, 0), (1, 0, 0), (0, 0, 1))
    assert right_x < 0 and left_x > 0
    assert right_y > 0 and left_y > 0
    assert abs(right_z) < 1e-6 and abs(left_z) < 1e-6

    behind_x, behind_y, _ = ear_offset((0.0, -4.0, 0.0), (0, 1, 0), (1, 0, 0), (0, 0, 1))
    assert abs(behind_x) < 1e-6 and behind_y < 0
