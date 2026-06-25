from enum import IntEnum


class StateIndex:       # TODO: revise
    """Index mapping for array representation of ego states."""

    # Single indices (integer constants)
    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    STEERING_ANGLE = 7
    STEERING_RATE = 8
    ANGULAR_VELOCITY = 9
    ANGULAR_ACCELERATION = 10

    # Contiguous ranges (slice constants)
    POINT = slice(X, Y + 1)                         # (X, Y)
    STATE_SE2 = slice(X, HEADING + 1)               # (X, Y, HEADING)
    VELOCITY_2D = slice(VELOCITY_X, VELOCITY_Y + 1) # (Vx, Vy)
    ACCELERATION_2D = slice(ACCELERATION_X, ACCELERATION_Y + 1)  # (Ax, Ay)

    @classmethod
    def size(cls) -> int:
        # Largest index + 1
        return 1 + max(
            cls.X, cls.Y, cls.HEADING,
            cls.VELOCITY_X, cls.VELOCITY_Y,
            cls.ACCELERATION_X, cls.ACCELERATION_Y,
            cls.STEERING_ANGLE, cls.STEERING_RATE,
            cls.ANGULAR_VELOCITY, cls.ANGULAR_ACCELERATION,
        )
        
class SE2Index(IntEnum):
    """Index mapping for state se2 (x,y,θ) arrays."""

    X = 0
    Y = 1
    HEADING = 2


class PointIndex(IntEnum):
    """Index mapping for (x,y) arrays."""

    X = 0
    Y = 1


class DynamicStateIndex(IntEnum):
    """Index mapping for dynamic car state (output of controller)."""

    ACCELERATION_X = 0
    STEERING_RATE = 1


class StateIDMIndex(IntEnum):
    """Index mapping for IDM states."""

    PROGRESS = 0
    VELOCITY = 1


class LeadingAgentIndex(IntEnum):
    """Index mapping for leading agent state (for IDM policies)."""

    PROGRESS = 0
    VELOCITY = 1
    LENGTH_REAR = 2


class BBCoordsIndex(IntEnum):
    """Index mapping for corners and center of bounding boxes."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


class EgoAreaIndex(IntEnum):
    """Index mapping for area of ego agent (used in PDMScorer)."""

    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2


class MultiMetricIndex(IntEnum):
    """Index mapping multiplicative metrics (used in PDMScorer)."""

    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    TRAFFIC_LIGHT_COMPLIANCE = 2
    DRIVING_DIRECTION = 3


class WeightedMetricIndex(IntEnum):
    """Index mapping weighted metrics (used in PDMScorer)."""

    PROGRESS = 0
    TTC = 1
    LANE_KEEPING = 2
    HISTORY_COMFORT = 3
    TWO_FRAME_EXTENDED_COMFORT = 4
