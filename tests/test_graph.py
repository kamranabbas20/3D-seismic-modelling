"""The dependency graph: rerun what changed, and nothing else."""

import copy

import pytest

from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import ConfigError
from sim3d.core.graph import (
    STAGES, changed_sections, dependents, explain, section_hashes,
    stale_stages, surviving_stages,
)


def mutate(**changes) -> tuple[ExperimentConfig, ExperimentConfig]:
    before = ExperimentConfig()
    after = copy.deepcopy(before)
    for path, value in changes.items():
        section, _, attribute = path.partition(".")
        target = getattr(after, section)
        if isinstance(getattr(target, attribute, None), dict):
            getattr(target, attribute).update(value)
        else:
            setattr(target, attribute, value)
    return before, after


def test_the_graph_is_topologically_ordered():
    seen = set()
    for stage in STAGES:
        assert set(stage.depends) <= seen, f"{stage.name} depends on a later stage"
        seen.add(stage.name)


def test_every_stage_has_a_reason_to_be_recomputed():
    """A stage with neither inputs nor dependencies could never go stale."""
    for stage in STAGES:
        assert stage.sections or stage.depends, stage.name


def test_an_identical_configuration_invalidates_nothing():
    before = ExperimentConfig()
    assert stale_stages(before, copy.deepcopy(before)) == set()
    assert "Nothing scientific changed" in explain(before, copy.deepcopy(before))


def test_display_settings_are_not_scientific_inputs():
    before, after = mutate(**{"output.directory": "elsewhere"})
    assert changed_sections(section_hashes(before), section_hashes(after)) == set()
    assert stale_stages(before, after) == set()


def test_extending_the_simulation_keeps_the_geology():
    """Requirement 17: changing only the duration must not rebuild geology."""
    before, after = mutate(**{"simulation.duration_days": 3650.0})
    stale = stale_stages(before, after)
    assert "flow" in stale and "gathers" in stale
    assert "geology" not in stale
    assert "wells" not in stale
    assert "acquisition" not in stale


def test_changing_the_wavelet_keeps_the_flow_and_the_rock_physics():
    before, after = mutate(**{"source.frequency": 25.0})
    stale = stale_stages(before, after)
    assert stale == {"gathers", "images", "fourd"}
    survived = surviving_stages(before, after)
    assert {"flow", "rockphysics", "states", "geology"} <= survived


def test_moving_a_well_keeps_the_geology_but_reruns_the_flow():
    before, after = mutate(**{"wells.pattern": "five_spot"})
    stale = stale_stages(before, after)
    assert {"wells", "completions", "controls", "flow", "rockphysics",
            "gathers", "images"} <= stale
    assert "geology" not in stale


def test_modifying_a_fault_reruns_everything_downstream_of_geometry():
    before, after = mutate(**{"geology.parameters": {"throw": 60.0}})
    stale = stale_stages(before, after)
    assert {"geology", "completions", "flow", "rockphysics", "gathers",
            "images", "fourd"} <= stale
    assert "wells" not in stale


def test_changing_the_imaging_condition_keeps_the_gathers():
    before, after = mutate(**{"imaging.imaging_condition": "crosscorrelation"})
    assert stale_stages(before, after) == {"images", "fourd"}
    assert "gathers" in surviving_stages(before, after)


def test_dependents_are_transitive():
    assert "fourd" in dependents("geology")
    assert "images" in dependents("wells")
    assert dependents("fourd") == set()


def test_unknown_stages_are_named():
    with pytest.raises(ConfigError, match="unknown stage"):
        dependents("teleport")


def test_explain_names_the_change_and_the_cost():
    before, after = mutate(**{"imaging.epsilon": 1e-3})
    text = explain(before, after)
    assert "imaging" in text and "expensive" in text and "Still valid" in text
