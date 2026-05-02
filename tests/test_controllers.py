import math

import pytest

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


def test_tangential_conflict_blend_avoids_hard_winner_switch():
    hard = TangentialSpacingController(beta_u=7.0, k_e_tau=25.0, conflict_blend_width=0.0)
    soft = TangentialSpacingController(beta_u=7.0, k_e_tau=25.0, conflict_blend_width=0.2)

    hard_jump = abs(hard._compose(1.001, -1.0) - hard._compose(0.999, -1.0))
    soft_jump = abs(soft._compose(1.001, -1.0) - soft._compose(0.999, -1.0))

    assert math.isclose(soft._compose(1.0, -1.0), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    assert soft_jump < 0.05
    assert hard_jump > 1.9


def test_tangential_default_composition_mode_is_blend():
    ctrl = TangentialSpacingController(beta_u=7.0, k_e_tau=25.0)
    assert ctrl.composition_mode == "blend"


def test_tangential_sum_mode_is_pure_addition_in_all_regimes():
    ctrl = TangentialSpacingController(
        beta_u=7.0, k_e_tau=25.0, composition_mode="sum",
    )
    # Cooperative regime — coincides with blend
    assert math.isclose(ctrl._compose(2.0, 1.0), 3.0, rel_tol=1e-9, abs_tol=1e-9)
    # Conflict, equal magnitudes — also coincides
    assert math.isclose(ctrl._compose(1.0, -1.0), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    # Conflict, dominant local — sum keeps the algebraic sum (not |dominant|)
    assert math.isclose(ctrl._compose(2.0, -1.0), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    # Conflict, dominant prop — symmetric
    assert math.isclose(ctrl._compose(-1.0, 2.0), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    # Symmetry under negation
    assert math.isclose(
        ctrl._compose(-2.0, 1.0), -ctrl._compose(2.0, -1.0),
        rel_tol=1e-9, abs_tol=1e-9,
    )


def test_tangential_sum_and_blend_diverge_in_conflict_with_unequal_magnitudes():
    blend_ctrl = TangentialSpacingController(
        beta_u=7.0, k_e_tau=25.0,
        composition_mode="blend", conflict_blend_width=0.2,
    )
    sum_ctrl = TangentialSpacingController(
        beta_u=7.0, k_e_tau=25.0, composition_mode="sum",
    )
    # In conflict with clearly dominant local, blend ≈ u_local while sum ≈ u_local + u_prop.
    # tanh(1/0.2) = tanh(5) ≈ 0.9999, so blend stays very close to +2.
    blend_value = blend_ctrl._compose(2.0, -1.0)
    sum_value = sum_ctrl._compose(2.0, -1.0)
    assert blend_value > 1.9
    assert math.isclose(sum_value, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_tangential_invalid_composition_mode_raises():
    with pytest.raises(ValueError, match="composition_mode"):
        TangentialSpacingController(
            beta_u=7.0, k_e_tau=25.0, composition_mode="not_a_mode",
        )
