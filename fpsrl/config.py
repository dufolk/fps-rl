"""Global gameplay constants. SI units, Z-up world, angles in degrees.

Yaw is measured counter-clockwise from +X (yaw=0 looks along +X, yaw=90 along +Y).
Pitch is positive when looking up.
"""

TICK_RATE = 60

GRAVITY = 20.0
JUMP_SPEED = 7.8  # apex ~1.46 m at 60 Hz (clears 1.2-1.4 m ledges); crouch jump (legs tucked) clears ~2.0 m
JUMP_COOLDOWN = 0.2

PLAYER_RADIUS = 0.3
STAND_HEIGHT = 1.8
CROUCH_HEIGHT = 1.2
EYE_FROM_TOP = 0.12
CROUCH_SPEED = 7.0  # metres of height change per second

RUN_SPEED = 5.4
WALK_MULT = 0.52
CROUCH_MULT = 0.42
GROUND_ACCEL = 12.0
AIR_ACCEL = 2.0
FRICTION = 10.0
STOP_SPEED = 1.6
STEP_HEIGHT = 0.45
FOOTSTEP_STRIDE = 2.1

MAX_HP = 100
RESPAWN_TIME = 3.0
SPAWN_PROTECT = 1.5
MAX_PITCH = 89.0

GRENADE_RADIUS = 0.07
GRENADE_RESTITUTION = 0.42
GRENADE_FRICTION = 0.72
