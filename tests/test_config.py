import numpy as np
import pytest

from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import ConfigError


def test_defaults_build_a_valid_domain_hierarchy():
    config = ExperimentConfig()
    domains = config.domains.build()
    assert domains.geology.n_cells > 0
    assert "Domain hierarchy" in domains.summary()


def test_round_trips_through_yaml(tmp_path):
    config = ExperimentConfig()
    config.project.name = "round_trip"
    config.source.frequency = 17.5
    path = config.save(tmp_path / "experiment.yaml")
    reloaded = ExperimentConfig.load(path)
    assert reloaded.project.name == "round_trip"
    assert reloaded.source.frequency == 17.5
    assert reloaded.content_hash() == config.content_hash()


def test_nested_sections_are_parsed_not_left_as_dicts():
    config = ExperimentConfig.from_dict({
        "source": {"frequency": 12.0},
        "reservoir": {"baseline": {"sw": 0.4}, "scenario": {"time_state": "T2"}},
    })
    assert config.source.frequency == 12.0
    assert config.reservoir.baseline.sw == 0.4
    assert config.reservoir.scenario.time_state == "T2"


def test_a_typo_is_an_error_not_a_silent_default():
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.from_dict({"source": {"freqency": 20.0}})
    with pytest.raises(ConfigError, match="unknown key"):
        ExperimentConfig.from_dict({"solverr": {}})


def test_a_non_mapping_section_is_reported_clearly():
    with pytest.raises(ConfigError, match="must be a mapping"):
        ExperimentConfig.from_dict({"source": [1, 2, 3]})


def test_a_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ExperimentConfig.load(tmp_path / "absent.yaml")


def test_the_hash_covers_the_science_and_nothing_else():
    config = ExperimentConfig()
    original = config.content_hash()

    config.output.directory = "somewhere/else"
    config.output.save_gathers = False
    assert config.content_hash() == original, "output settings must not invalidate a cache"

    config.rock_physics.dry_frame_model = "stiff_sand"
    assert config.content_hash() != original, "a frame model change must invalidate it"


@pytest.mark.parametrize("path", ["examples/configs/demo_small.yaml",
                                  "examples/configs/research_standard.yaml"])
def test_the_shipped_examples_load_and_describe(path):
    config = ExperimentConfig.load(path)
    text = config.describe()
    assert config.project.name in text
    assert config.short_hash in text
    domains = config.domains.build()  # raises if the hierarchy is inconsistent
    assert domains.propagation.n_cells > 0
