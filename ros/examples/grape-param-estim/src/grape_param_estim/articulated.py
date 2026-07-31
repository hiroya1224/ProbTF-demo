"""Pure-Python aggregate model of the articulated Grape controller geometry."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from grape_param_estim.system import GrapeGeometry, VehicleParameters


def _rotation_x(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


@dataclass(frozen=True)
class _LinkInertia:
    mass: float
    com: np.ndarray
    inertia: np.ndarray


def _link(mass, com, inertia) -> _LinkInertia:
    matrix = np.asarray(inertia, dtype=float)
    if matrix.shape == (6,):
        ixx, ixy, ixz, iyy, iyz, izz = matrix
        matrix = np.asarray(
            ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))
        )
    return _LinkInertia(float(mass), np.asarray(com, dtype=float), matrix)


_MAIN = _link(
    1.03359725029313,
    (-0.00462482034298556, -7.72723613950892e-05,
     0.0137950803533556),
    (0.0016139879539522, -5.366768549397e-07,
     -5.95600212021384e-07, 0.00160428977142187,
     9.13377581199397e-08, 0.00305929998385753),
)
_FC = _link(
    0.0134869851240831,
    (0.0187002711962543, 0.00169857186981875,
     -0.00124990288529885),
    (1.81746685536929e-06, 3.52111681460635e-09,
     2.41443574658866e-11, 1.81309403254674e-06,
     -1.82051260509328e-10, 3.61332371658314e-06),
)
_GIMBAL = (
    _link(
        0.308825311849622,
        (-0.0061543571832175, 2.02344799955032e-06,
         0.00258056279461697),
        (6.23139432038947e-05, -6.80218913011022e-09,
         -2.71244742730139e-06, 6.42832899579805e-05,
         -5.73083772407336e-09, 6.89107151163971e-05),
    ),
    _link(
        0.308825311849623,
        (-0.0061543571832175, 2.02344799955111e-06,
         0.00258056279461695),
        (6.23139432038948e-05, -6.80218913009929e-09,
         -2.7124474273014e-06, 6.42832899579805e-05,
         -5.73083772407336e-09, 6.89107151163972e-05),
    ),
    _link(
        0.308825311849623,
        (-0.00615435718321899, 2.0234479995473e-06,
         0.002580562794617),
        (6.23139432038948e-05, -6.80218913009913e-09,
         -2.71244742730139e-06, 6.42832899579805e-05,
         -5.73083772407073e-09, 6.89107151163971e-05),
    ),
    _link(
        0.308825311849623,
        (-0.00615435718321755, 2.02344799954969e-06,
         0.00258056279461696),
        (6.23139432038948e-05, -6.80218913011076e-09,
         -2.7124474273014e-06, 6.42832899579806e-05,
         -5.73083772407164e-09, 6.89107151163971e-05),
    ),
)
_THRUST = (
    _link(
        0.0172928285355672,
        (-3.90447923992454e-09, 2.30010885643358e-08,
         0.00419839111910028),
        (0.000132814047308871, -6.27243278095286e-06,
         -3.04213085195885e-13, 1.02207952913888e-06,
         -3.00688010181912e-13, 0.000133587732946345),
    ),
    _link(
        0.0172932093865153,
        (-1.30508629458959e-08, 1.85095488362644e-07,
         0.00419843988070346),
        (0.00013281236489956, 6.27225423023414e-06,
         -7.99839179017161e-14, 1.02205641696451e-06,
         3.025237661732e-12, 0.000133586033085267),
    ),
    _link(
        0.0172928313440091,
        (5.83452963720532e-09, -5.91908131602234e-10,
         0.00419839092638537),
        (0.000132814029105212, -6.27243221426146e-06,
         2.95243892069162e-13, 1.02207952004091e-06,
         -1.24906779391424e-14, 0.000133587714747096),
    ),
    _link(
        0.017293238730581,
        (-2.59015725534439e-09, 2.94869426286333e-09,
         0.00419843814070694),
        (0.00013281274906218, 6.27228613960569e-06,
         -1.77493503318e-13, 1.02205918959433e-06,
         1.85598427852344e-13, 0.000133586419954865),
    ),
)


class GrapeArticulatedModel:
    """Aggregate the audited Grape URDF at four measured gimbal angles."""

    arm_origins = np.asarray(
        (
            (-0.22309, -0.22309, 0.0),
            (0.22309, -0.22309, 0.0),
            (0.22309, 0.22309, 0.0),
            (-0.22309, 0.22309, 0.0),
        )
    )
    arm_yaws = np.asarray((-2.3562, -0.7854, 0.7854, 2.3562))
    rotor_directions = np.asarray((-1.0, 1.0, -1.0, 1.0))
    fc_origin = np.asarray(
        (-0.0172999968682441, -0.00110000084294132, 0.05706099896)
    )

    @staticmethod
    def _aggregate(items):
        total_mass = sum(link.mass for link, _origin, _rotation in items)
        centres = [
            origin + rotation @ link.com
            for link, origin, rotation in items
        ]
        centre = sum(
            link.mass * value
            for (link, _origin, _rotation), value in zip(items, centres)
        ) / total_mass
        inertia = np.zeros((3, 3), dtype=float)
        for (link, _origin, rotation), link_centre in zip(items, centres):
            displacement = link_centre - centre
            inertia += rotation @ link.inertia @ rotation.T
            inertia += link.mass * (
                np.dot(displacement, displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        return total_mass, centre, inertia

    def at(
        self, gimbal_angles: Sequence[float]
    ) -> Tuple[VehicleParameters, GrapeGeometry]:
        angles = np.asarray(gimbal_angles, dtype=float)
        if angles.shape != (4,) or not np.all(np.isfinite(angles)):
            raise ValueError("gimbal angles must contain four finite values")
        items = [(_MAIN, np.zeros(3), np.eye(3)),
                 (_FC, self.fc_origin, np.eye(3))]
        # RobotModel includes four fixed virtual rotor-arm inertials.  The
        # root dummy is outside the main-body subtree used for aggregation.
        virtual = _link(1.0e-5, np.zeros(3), (1.0e-6, 0.0, 0.0,
                                             1.0e-6, 0.0, 2.0e-6))
        thrust_origins_main = np.empty((4, 3), dtype=float)
        for rotor, angle in enumerate(angles):
            arm_rotation = _rotation_z(self.arm_yaws[rotor])
            moving_rotation = arm_rotation @ _rotation_x(float(angle))
            arm_origin = self.arm_origins[rotor]
            thrust_origin = arm_origin + moving_rotation @ np.asarray(
                (0.0, 0.0, 0.056)
            )
            thrust_origins_main[rotor] = thrust_origin
            items.append((virtual, arm_origin, arm_rotation))
            items.append((_GIMBAL[rotor], arm_origin, moving_rotation))
            items.append((_THRUST[rotor], thrust_origin, moving_rotation))
        mass, centre, inertia = self._aggregate(items)
        parameters = VehicleParameters(
            mass=mass,
            inertia=inertia,
            # Geometry is already expressed about this instantaneous CoG.
            # VehicleParameters.cog_offset denotes an additional plant-side
            # mismatch relative to the geometry reference, so it is zero for
            # an internally consistent controller snapshot.
            cog_offset=np.zeros(3),
            force_effectiveness=np.ones(4),
            torque_effectiveness=np.ones(4),
            linear_drag=np.zeros(3),
            angular_drag=np.zeros(3),
        )
        geometry = GrapeGeometry(
            rotor_origins=thrust_origins_main - centre,
            arm_yaws=self.arm_yaws,
            rotor_directions=self.rotor_directions,
            moment_force_rate=-0.0181,
            # Origins above already contain the articulated offset.
            thrust_offset=0.0,
        )
        return parameters, geometry
