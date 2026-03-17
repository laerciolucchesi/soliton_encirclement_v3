import math

from controllers import (
    RadialDistanceController,
    TangentialSpacingController,
    WrappedAnglePDController,
)


def test_radial_distance_controller_matches_pd_update():
    ctrl = RadialDistanceController(kp=1.0, kd=0.5, radius_setpoint=10.0)

    first = ctrl.update(measurement=10.5, dt=0.1)
    second = ctrl.update(measurement=10.6, dt=0.1)

    assert math.isclose(first, -0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(second, -1.1, rel_tol=1e-9, abs_tol=1e-9)


def test_wrapped_angle_controller_preserves_wrapped_p_and_shortest_path_d():
    ctrl = WrappedAnglePDController(kp=1.0, kd=0.5, max_abs_output=10.0)

    first = ctrl.update(measurement=(math.pi - 0.01), dt=0.1)
    second = ctrl.update(measurement=(-math.pi + 0.01), dt=0.1)

    expected_first = math.pi - 0.01
    expected_second = (-math.pi + 0.01) + (0.5 * (0.02 / 0.1))

    assert math.isclose(first, expected_first, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(second, expected_second, rel_tol=1e-9, abs_tol=1e-9)


def test_tangential_spacing_controller_keeps_existing_state_dynamics():
    ctrl = TangentialSpacingController(beta_u=7.0, k_e_tau=25.0)

    first = ctrl.update(measurement=0.2, dt=0.1)
    second = ctrl.update(measurement=0.2, dt=0.1)

    assert math.isclose(first.u, 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(first.du_damp, 0.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(first.du_from_error, 5.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(first.delta_u, 0.5, rel_tol=1e-9, abs_tol=1e-9)

    assert math.isclose(second.u, 0.65, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(second.du_damp, -3.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(second.du_from_error, 5.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(second.delta_u, 0.15, rel_tol=1e-9, abs_tol=1e-9)
