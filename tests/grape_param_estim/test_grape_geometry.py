import unittest

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.grape_geometry import (
    FIXED_GEOMETRY_EVIDENCE_STATUS,
    FIXED_GEOMETRY_PROFILE_ID,
    FIXED_GEOMETRY_PROFILE_SCHEMA,
    FIXED_GEOMETRY_PROFILE_SHA256,
    allocate_wrench,
    fixed_geometry_profile,
    reconstruct_actuator_wrench,
    validate_fixed_geometry_declaration,
)
from grape_param_estim.plant import FirstOrderActuatorBackend


class GrapeGeometryTests(unittest.TestCase):
    def test_symmetric_vertical_thrust_has_no_horizontal_force(self):
        wrench = reconstruct_actuator_wrench(np.full(4, 5.0), np.zeros(4))
        np.testing.assert_allclose(wrench[:2], np.zeros(2), atol=1.0e-12)
        self.assertAlmostEqual(wrench[2], 20.0, places=12)
        np.testing.assert_allclose(
            wrench[3:],
            [0.0220000168588264, -0.345999937364882, 0.0],
            atol=1.0e-10,
        )

    def test_synthetic_allocator_round_trip(self):
        target = np.array([1.0, -0.7, 22.0, 0.15, -0.12, 0.08])
        thrust, angle, normalized_residual = allocate_wrench(target)
        predicted = reconstruct_actuator_wrench(thrust, angle)
        self.assertLess(normalized_residual, 1.0e-5)
        np.testing.assert_allclose(predicted, target, rtol=0.0, atol=2.0e-5)

    def test_fixed_geometry_requires_an_explicit_hash_bound_declaration(self):
        declaration = {
            "schema": FIXED_GEOMETRY_PROFILE_SCHEMA,
            "profile_id": FIXED_GEOMETRY_PROFILE_ID,
            "profile_sha256": FIXED_GEOMETRY_PROFILE_SHA256,
            "evidence_status": FIXED_GEOMETRY_EVIDENCE_STATUS,
        }
        self.assertEqual(
            validate_fixed_geometry_declaration(declaration),
            declaration,
        )
        self.assertEqual(
            FIXED_GEOMETRY_PROFILE_SHA256,
            stable_hash(fixed_geometry_profile()),
        )
        changed = dict(declaration, profile_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "not bound"):
            validate_fixed_geometry_declaration(changed)

        legacy = FirstOrderActuatorBackend()
        explicit = FirstOrderActuatorBackend(
            FIXED_GEOMETRY_PROFILE_SHA256
        )
        self.assertFalse(legacy.geometry_profile_explicit)
        self.assertTrue(explicit.geometry_profile_explicit)
        with self.assertRaisesRegex(ValueError, "does not match"):
            FirstOrderActuatorBackend("0" * 64)


if __name__ == "__main__":
    unittest.main()
