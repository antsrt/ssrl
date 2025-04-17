from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path


class AliengoUtils:
    """Aliengo robot helper utilities."""

    # constants
    BODY_LENGTH = 0.363
    BODY_WIDTH = 0.22

    # leg offsets from center of body to hip
    LEG_OFFSET_X = BODY_LENGTH / 2
    LEG_OFFSET_Y = BODY_WIDTH / 2

    # leg link lengths
    HIP_LENGTH = 0.083
    THIGH_LENGTH = 0.25
    THIGH_OFFSET = 0.0
    CALF_LENGTH = 0.25

    # joint ranges
    HIP_RANGE = 0.802851  # 46 deg
    THIGH_RANGE = 0.802851  # 46 deg
    CALF_RANGE = 1.74533  # 100 deg

    # motors
    MOTOR_KP = 100.0
    MOTOR_KD = 2.0

    # environment target pose
    STAND_HEIGHT = 0.4

    STANDING_FOOT_POSITIONS = jp.array([
        0.1815, -0.1300, -0.4000,
        0.1815, 0.1300, -0.4000,
        -0.1815, -0.1300, -0.4000,
        -0.1815, 0.1300, -0.4000
    ])

    STANDING_JOINT_ANGLES_FR = jp.array([-0.01014303, 0.7180088, -1.4360176])
    STANDING_JOINT_ANGLES_FL = jp.array([0.01014303, 0.7180088, -1.4360176])
    STANDING_JOINT_ANGLES_RR = jp.array([-0.01014303, 0.7180088, -1.4360176])
    STANDING_JOINT_ANGLES_RL = jp.array([0.01014303, 0.7180088, -1.4360176])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_FR,
        STANDING_JOINT_ANGLES_FL,
        STANDING_JOINT_ANGLES_RR,
        STANDING_JOINT_ANGLES_RL
    ])

    JOINT_LIMIT_PAD = 0.1
    """constant: the amount to pad the joint limits"""

    LOWER_JOINT_LIMITS = jp.array([-0.873, -0.524, -2.775]) + JOINT_LIMIT_PAD
    """constant: the lower joint angle limits for a leg, obtained from
    unitree_legged_sdk/include/aliengo_const.h, and offset by JOINT_LIMIT_PAD"""

    UPPER_JOINT_LIMITS = jp.array([1.047, 3.927, -0.611]) - JOINT_LIMIT_PAD
    """constant: the upper joint angle limits for a leg, obtained from
    unitree_legged_sdk/include/aliengo_const.h, and offset by JOINT_LIMIT_PAD"""

    MOTOR_TORQUE_LIMIT = jp.tile(jp.array([33.5, 33.5, 40.0]), 4)
    """constant: the torque limit for the motors"""

    CACHE_PATH = epath.resource_path('brax') / 'robots/aliengo/.cache'

    @classmethod
    def standing_foot_positions(cls):
        """Returns the foot positions for the robot in a standing posture.

        Returns:
            A numpy array of shape (12,) containing the foot positions in the
            body frame, organized as [FR, FL, RR, RL], where each foot position
            is [x, y, z].
        """
        return cls.STANDING_FOOT_POSITIONS

    @classmethod
    def forward_kinematics(cls, leg: str, q: jp.ndarray) -> jp.ndarray:
        """Computes the forward kinematics for a specific leg.

        Args:
            leg: String identifier for the leg ('FR', 'FL', 'RR', or 'RL')
            q: Array of shape (3,) containing the joint angles [hip, thigh, calf]

        Returns:
            Array of shape (3,) containing the foot position in the body frame
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')
            
        # Determine leg offsets based on leg position
        if leg == 'FR':
            leg_offset_x = cls.LEG_OFFSET_X
            leg_offset_y = -cls.LEG_OFFSET_Y
        elif leg == 'FL':
            leg_offset_x = cls.LEG_OFFSET_X
            leg_offset_y = cls.LEG_OFFSET_Y
        elif leg == 'RR':
            leg_offset_x = -cls.LEG_OFFSET_X
            leg_offset_y = -cls.LEG_OFFSET_Y
        else:  # RL
            leg_offset_x = -cls.LEG_OFFSET_X
            leg_offset_y = cls.LEG_OFFSET_Y
            
        # Apply stand pose offset
        q_offset = jp.array([0., -0.6, 1.2])
        q_adj = q + q_offset
        
        hip_angle = q_adj[0]
        thigh_angle = q_adj[1]
        calf_angle = q_adj[2]
        
        # Compute position in leg frame
        x = cls.THIGH_LENGTH * jp.sin(thigh_angle) + cls.CALF_LENGTH * jp.sin(thigh_angle + calf_angle)
        y = 0.0
        z = -cls.THIGH_LENGTH * jp.cos(thigh_angle) - cls.CALF_LENGTH * jp.cos(thigh_angle + calf_angle)
        
        # Apply hip rotation
        cos_hip = jp.cos(hip_angle)
        sin_hip = jp.sin(hip_angle)
        
        x_rot = x * cos_hip
        y_rot = x * sin_hip
        
        # Add leg offsets to get position in body frame
        x_world = x_rot + leg_offset_x
        y_world = y_rot + leg_offset_y
        z_world = z
        
        return jp.array([x_world, y_world, z_world])

    @classmethod
    def inverse_kinematics(cls, leg: str, p: jp.ndarray) -> jp.ndarray:
        """Computes the inverse kinematics for a specific leg.

        Args:
            leg: String identifier for the leg ('FR', 'FL', 'RR', or 'RL')
            p: Array of shape (3,) containing the desired foot position in the body frame

        Returns:
            Array of shape (3,) containing the joint angles [hip, thigh, calf]
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')
            
        # Determine leg offsets based on leg position
        if leg == 'FR':
            leg_offset_x = cls.LEG_OFFSET_X
            leg_offset_y = -cls.LEG_OFFSET_Y
        elif leg == 'FL':
            leg_offset_x = cls.LEG_OFFSET_X
            leg_offset_y = cls.LEG_OFFSET_Y
        elif leg == 'RR':
            leg_offset_x = -cls.LEG_OFFSET_X
            leg_offset_y = -cls.LEG_OFFSET_Y
        else:  # RL
            leg_offset_x = -cls.LEG_OFFSET_X
            leg_offset_y = cls.LEG_OFFSET_Y
        
        # Get foot position relative to the hip
        x_rel = p[0] - leg_offset_x
        y_rel = p[1] - leg_offset_y
        z_rel = p[2]
        
        # Compute hip angle
        hip_angle = jp.arctan2(y_rel, x_rel)
        
        # Get leg length in the x-z plane (after hip rotation)
        leg_length = jp.sqrt(x_rel**2 + y_rel**2)
        
        # Apply inverse hip rotation to get foot position in leg frame
        x_leg = leg_length
        z_leg = z_rel
        
        # Compute leg length in leg frame
        L = jp.sqrt(x_leg**2 + z_leg**2)
        
        # Check if the foot position is reachable
        cos_knee = (L**2 - cls.THIGH_LENGTH**2 - cls.CALF_LENGTH**2) / (2 * cls.THIGH_LENGTH * cls.CALF_LENGTH)
        cos_knee = jp.clip(cos_knee, -1.0, 1.0)
        
        # Compute joint angles
        calf_angle = jp.arccos(cos_knee)
        thigh_angle = jp.arctan2(x_leg, -z_leg) - jp.arctan2(
            cls.CALF_LENGTH * jp.sin(calf_angle),
            cls.THIGH_LENGTH + cls.CALF_LENGTH * jp.cos(calf_angle)
        )
        
        # Adjust for stand pose offset
        q_offset = jp.array([0., -0.6, 1.2])
        
        # Organize the result
        q = jp.array([hip_angle, thigh_angle, calf_angle])
        
        return q - q_offset

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Aliengo."""

        if used_cached:
            sys = AliengoUtils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/aliengo/xml/aliengo.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Aliengo."""

        if used_cached:
            sys = AliengoUtils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/aliengo/xml/aliengo_approx.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Aliengo to avoid reloading the xml file."""
        sys = AliengoUtils.get_system()
        Path(AliengoUtils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(AliengoUtils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Aliengo."""
        try:
            with open(AliengoUtils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = AliengoUtils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Aliengo."""
        if approx_system:
            path = AliengoUtils.CACHE_PATH / 'aliengo_approx_system.pkl'
        else:
            path = AliengoUtils.CACHE_PATH / 'aliengo_system.pkl'
        return path

    @staticmethod
    def forward_kinematics_all_legs(q: jp.ndarray) -> jp.ndarray:
        """Returns the positions of the feet in the body frame centered on the
           trunk, given the joint angles; (12,)

        Arguments:
            q (jp.ndarray): the joint angles of all legs; (12,)
        """
        p = jp.concatenate([
            AliengoUtils.forward_kinematics('FR', q[0:3]),
            AliengoUtils.forward_kinematics('FL', q[3:6]),
            AliengoUtils.forward_kinematics('RR', q[6:9]),
            AliengoUtils.forward_kinematics('RL', q[9:12]),
        ])
        return p

    @staticmethod
    def inverse_kinematics_all_legs(p: jp.ndarray) -> jp.ndarray:
        """Returns the joint angles of all legs given the positions of the feet
           in the body frame centered on the trunk; (12,)

        Arguments:
            p (jp.ndarray): the positions of the feet in the body frame; (12,)
        """
        q = jp.concatenate([
            AliengoUtils.inverse_kinematics('FR', p[0:3]),
            AliengoUtils.inverse_kinematics('FL', p[3:6]),
            AliengoUtils.inverse_kinematics('RR', p[6:9]),
            AliengoUtils.inverse_kinematics('RL', p[9:12]),
        ])
        return q

    @staticmethod
    def jacobian(leg: str, q: jp.ndarray) -> jp.ndarray:
        """get the jacobian of the leg

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.ndarray): the joint angles of a leg; (3,)

        Returns:
            jp.ndarray: the jacobian of the leg, (3, 3)
        """

        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        d = jax.lax.select(leg in ['FR', 'RR'],
                           -AliengoUtils.THIGH_OFFSET,
                           AliengoUtils.THIGH_OFFSET)
        length = AliengoUtils.THIGH_LENGTH

        q1 = q[0]
        q2 = q[1]
        q3 = q[2]

        J00 = 0.
        J01 = -length*(jp.cos(q2 + q3) + jp.cos(q2))
        J02 = -length*jp.cos(q2 + q3)
        J10 = (
            length*jp.cos(q1)*jp.cos(q2)
            - d*jp.sin(q1)
            + length*jp.cos(q1)*jp.cos(q2)*jp.cos(q3)
            - length*jp.cos(q1)*jp.sin(q2)*jp.sin(q3)
        )
        J11 = -length*jp.sin(q1)*(jp.sin(q2 + q3) + jp.sin(q2))
        J12 = -length*jp.sin(q2 + q3)*jp.sin(q1)
        J20 = (
            d*jp.cos(q1)
            + length*jp.cos(q2)*jp.sin(q1)
            + length*jp.cos(q2)*jp.cos(q3)*jp.sin(q1)
            - length*jp.sin(q1)*jp.sin(q2)*jp.sin(q3)
        )
        J21 = length*jp.cos(q1)*(jp.sin(q2 + q3) + jp.sin(q2))
        J22 = length*jp.sin(q2 + q3)*jp.cos(q1)

        J = jp.stack([
            jp.stack([J00, J01, J02], axis=0),
            jp.stack([J10, J11, J12], axis=0),
            jp.stack([J20, J21, J22], axis=0)
        ], axis=0)

        return J

    @staticmethod
    def foot_vel(leg: str, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the linear velocity of the foot in the body frame; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.jp.ndarray): the joint angles of a leg; (3,)
            qd (jp.jp.ndarray): the joint speeds of a leg; (3,)
        """
        J = AliengoUtils.jacobian(leg, q)
        vel = jp.matmul(J, qd)
        return vel

    @staticmethod
    def foot_vel_all_legs(q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the linear velocities of all feet in the body frame; (12,)

        Arguments:
            q (jp.ndarray): the joint angles of all legs; (12,)
            qd (jp.ndarray): the joint speeds of all legs; (12,)
        """
        vel = jp.concatenate([
            AliengoUtils.foot_vel('FR', q[0:3], qd[0:3]),
            AliengoUtils.foot_vel('FL', q[3:6], qd[3:6]),
            AliengoUtils.foot_vel('RR', q[6:9], qd[6:9]),
            AliengoUtils.foot_vel('RL', q[9:12], qd[9:12]),
        ])
        return vel

    @staticmethod
    def standing_foot_positions() -> jp.ndarray:
        """Returns the positions of the feet in the body frame when the robot
        is standing; (12,)"""
        return AliengoUtils.STANDING_FOOT_POSITIONS


if __name__ == '__main__':
    q = jp.array([0.1, 0.2, 0.3])
    qd = jp.array([0.1, 0.2, 0.3])
    p = AliengoUtils.forward_kinematics('FR', q)
    pd = AliengoUtils.foot_vel('FR', q, qd)
    J = AliengoUtils.jacobian('FR', q)
    print(p)
    print(pd)
    print(J)
    print(AliengoUtils.standing_foot_positions())
    print(AliengoUtils.inverse_kinematics_all_legs(
        AliengoUtils.standing_foot_positions())) 