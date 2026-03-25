"""Tests for application view state enum."""

from src.core.view_state import ViewState



def test_enum_contains_all_required_states() -> None:
    """Ensure the enum exposes every required state name."""
    expected = {"IDLE", "RECORDING", "PROCESSING", "ERROR"}
    actual = {state.name for state in ViewState}
    assert actual == expected



def test_enum_values_are_unique() -> None:
    """Ensure each state has a unique value."""
    values = [state.value for state in ViewState]
    assert len(values) == len(set(values))



def test_state_comparisons_work() -> None:
    """Validate equality/inequality comparisons used in state logic."""
    assert ViewState.IDLE == ViewState.IDLE
    assert ViewState.RECORDING != ViewState.PROCESSING
    assert ViewState.ERROR != ViewState.IDLE
