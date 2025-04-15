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
    def forward_kinematics(cls, q: jp.ndarray):
        """Computes the forward kinematics for all joints.

        Args:
            q: An array of shape (12,) containing the joint angles organized as
              [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
               RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf],
              where abduction is positive for right legs and negative for left legs.

        Returns:
            A numpy array of shape (4, 3) containing the foot positions in the body
            frame, organized as [FR, FL, RR, RL].
        """
        q_offset = jp.array([0., -0.6, 1.2] * 4)  # Offset for standing pose
        q = q + q_offset
        
        # Extract joint angles for each leg
        hip_angles = q[0::3]
        thigh_angles = q[1::3]
        calf_angles = q[2::3]
        
        # Compute positions for each leg
        x = cls.THIGH_LENGTH * jp.sin(thigh_angles) + cls.CALF_LENGTH * jp.sin(thigh_angles + calf_angles)
        y = jp.zeros_like(x)
        z = -cls.THIGH_LENGTH * jp.cos(thigh_angles) - cls.CALF_LENGTH * jp.cos(thigh_angles + calf_angles)
        
        # Apply hip rotation and leg offsets
        cos_hip = jp.cos(hip_angles)
        sin_hip = jp.sin(hip_angles)
        
        leg_offsets_x = jp.array([cls.LEG_OFFSET_X, cls.LEG_OFFSET_X, -cls.LEG_OFFSET_X, -cls.LEG_OFFSET_X])
        leg_offsets_y = jp.array([-cls.LEG_OFFSET_Y, cls.LEG_OFFSET_Y, -cls.LEG_OFFSET_Y, cls.LEG_OFFSET_Y])
        
        x_rot = x * cos_hip
        y_rot = x * sin_hip
        
        x_world = x_rot + leg_offsets_x
        y_world = y_rot + leg_offsets_y
        z_world = z
        
        # Organize the result
        foot_pos = jp.stack([x_world, y_world, z_world], axis=1)
        return foot_pos

    @classmethod
    def inverse_kinematics(cls, foot_world_pos: jp.ndarray):
        """Computes the inverse kinematics for all feet.

        Args:
            foot_world_pos: An array of shape (4, 3) containing the desired foot
              positions in the body frame, organized as [FR, FL, RR, RL].

        Returns:
            A numpy array of shape (12,) containing the joint angles organized as
              [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
               RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf].
        """
        # Get foot positions relative to the hip
        leg_offsets_x = jp.array([cls.LEG_OFFSET_X, cls.LEG_OFFSET_X, -cls.LEG_OFFSET_X, -cls.LEG_OFFSET_X])
        leg_offsets_y = jp.array([-cls.LEG_OFFSET_Y, cls.LEG_OFFSET_Y, -cls.LEG_OFFSET_Y, cls.LEG_OFFSET_Y])
        
        x_rel = foot_world_pos[:, 0] - leg_offsets_x
        y_rel = foot_world_pos[:, 1] - leg_offsets_y
        z_rel = foot_world_pos[:, 2]
        
        # Compute hip angle
        hip_angles = jp.arctan2(y_rel, x_rel)
        
        # Get leg length in the x-z plane (after hip rotation)
        leg_length = jp.sqrt(x_rel**2 + y_rel**2)
        
        # Apply inverse hip rotation to get foot positions in leg frame
        x_leg = leg_length
        z_leg = z_rel
        
        # Compute leg length in leg frame
        L = jp.sqrt(x_leg**2 + z_leg**2)
        
        # Check if the foot position is reachable
        cos_knee = (L**2 - cls.THIGH_LENGTH**2 - cls.CALF_LENGTH**2) / (2 * cls.THIGH_LENGTH * cls.CALF_LENGTH)
        cos_knee = jp.clip(cos_knee, -1.0, 1.0)
        
        # Compute joint angles
        calf_angles = jp.arccos(cos_knee)
        thigh_angles = jp.arctan2(x_leg, -z_leg) - jp.arctan2(
            cls.CALF_LENGTH * jp.sin(calf_angles),
            cls.THIGH_LENGTH + cls.CALF_LENGTH * jp.cos(calf_angles)
        )
        
        # Adjust for stand pose offset
        q_offset = jp.array([0., -0.6, 1.2] * 4)
        
        # Organize the result
        q = jp.zeros((12,))
        q = q.at[0::3].set(hip_angles)
        q = q.at[1::3].set(thigh_angles)
        q = q.at[2::3].set(calf_angles)
        
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