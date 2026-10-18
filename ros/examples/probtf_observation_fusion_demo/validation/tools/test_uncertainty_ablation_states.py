"""State/K oracles cannot be called directly after permission withdrawal."""
import pytest

from uncertainty_ablation_policy import TruthInjectionRevoked
from uncertainty_ablation_states import intervene_control_result, oracle_K_packet


def test_baseline_is_object_identical_and_does_not_read_truth():
    result = object()
    assert intervene_control_result(result, None, None, 'baseline') is result


@pytest.mark.parametrize('mode', ('oracle_goal', 'oracle_marker', 'true_K',
    'oracle_body', 'oracle_body_goal', 'all_oracle', 'known_landmark', 'unknown'))
def test_direct_state_interventions_are_blocked_before_reading_the_law(mode):
    with pytest.raises(TruthInjectionRevoked, match='revoked.*2026-10-19'):
        intervene_control_result(None, None, None, mode)


@pytest.mark.parametrize('packet', (None, object()))
def test_K_packet_construction_is_blocked_even_without_selected_K(packet):
    with pytest.raises(TruthInjectionRevoked):
        oracle_K_packet(None, None, packet, None)
