"""Scenario persistence: save, load, duplicate, rename, delete (13, 14)."""

import numpy as np
import pytest

from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import ConfigError
from sim3d.io import ScenarioStore, ViewState


@pytest.fixture
def store(tmp_path):
    return ScenarioStore(tmp_path / "scenarios")


def config(name="base", **overrides):
    cfg = ExperimentConfig()
    cfg.project.name = name
    for path, value in overrides.items():
        section, _, attribute = path.partition(".")
        setattr(getattr(cfg, section), attribute, value)
    return cfg


def test_a_saved_scenario_round_trips(store):
    cfg = config("base", **{"simulation.duration_days": 900.0})
    view = ViewState(active_day=365.0, camera={"eye": {"x": 1.0}},
                     visible_layers=["reservoir"])
    store.save("base", cfg, view=view)

    loaded = store.load("base")
    assert loaded.config.simulation.duration_days == 900.0
    assert loaded.config.content_hash() == cfg.content_hash()
    assert loaded.view.active_day == 365.0
    assert loaded.view.camera == {"eye": {"x": 1.0}}


def test_the_three_stores_are_separate_files(store):
    cfg = config()
    store.save("base", cfg,
               simulation={"days": np.arange(4.0)},
               seismic={"image:baseline": np.zeros((2, 2, 2))})
    folder = store.path("base")
    assert {p.name for p in folder.iterdir()} == {
        "scenario.json", "view.json", "simulation.npz", "seismic.npz",
        "manifest.json"}
    # The definition alone is small and readable.
    assert folder.joinpath("scenario.json").read_text(encoding="utf-8").startswith("{")


def test_saving_a_configuration_edit_does_not_discard_results(store):
    """The failure that would cost someone an overnight migration."""
    store.save("base", config(), simulation={"days": np.arange(4.0)})
    store.save("base", config("base", **{"source.frequency": 25.0}))
    reloaded = store.load("base")
    assert reloaded.simulation is not None
    assert reloaded.config.source.frequency == 25.0


def test_results_can_be_cleared_deliberately(store):
    store.save("base", config(), simulation={"days": np.arange(4.0)})
    store.save("base", config(), simulation={})
    assert store.load("base").simulation is None


def test_loading_without_results_is_cheap_and_complete(store):
    store.save("base", config(), simulation={"pressure": np.zeros((3, 4, 4, 4))})
    definition = store.load("base", results=False)
    assert definition.simulation is None
    assert definition.config.project.name == "base"


def test_listing_reports_what_is_stored(store):
    store.save("with results", config("with results"),
               simulation={"days": np.arange(2.0)})
    store.save("definition only", config("definition only"))
    listing = {s.name: s for s in store.list()}
    assert listing["with results"].has_simulation
    assert not listing["definition only"].has_simulation
    assert listing["with results"].config_hash
    assert "results: flow" in listing["with results"].describe()


def test_duplicate_defaults_to_the_definition_alone(store):
    store.save("base", config(), simulation={"days": np.arange(2.0)})
    store.duplicate("base", "variant")
    assert store.load("variant").simulation is None
    assert store.load("variant").config.content_hash() == config().content_hash()

    store.duplicate("base", "full copy", results=True)
    assert store.load("full copy").simulation is not None


def test_duplicate_and_rename_refuse_to_overwrite(store):
    store.save("base", config())
    store.save("other", config("other"))
    with pytest.raises(ConfigError, match="already exists"):
        store.duplicate("base", "other")
    with pytest.raises(ConfigError, match="already exists"):
        store.rename("base", "other")


def test_rename_moves_the_whole_scenario(store):
    store.save("base", config(), simulation={"days": np.arange(2.0)})
    store.rename("base", "renamed")
    assert [s.name for s in store.list()] == ["renamed"]
    assert store.load("renamed").simulation is not None


def test_delete_removes_everything(store):
    store.save("base", config(), simulation={"days": np.arange(2.0)})
    store.delete("base")
    assert store.list() == []
    assert not store.path("base").exists()


def test_missing_scenarios_are_named_not_guessed(store):
    store.save("base", config())
    with pytest.raises(ConfigError, match="no scenario named"):
        store.load("absent")
    with pytest.raises(ConfigError, match="no scenario named"):
        store.delete("absent")


def test_names_are_checked_because_they_are_folder_names(store):
    with pytest.raises(ConfigError, match="folder names"):
        store.save("../escape", config())
    with pytest.raises(ConfigError, match="needs a name"):
        store.save("   ", config())


def test_an_empty_store_lists_nothing(tmp_path):
    assert ScenarioStore(tmp_path / "nothing").list() == []


def test_flow_results_pack_and_unpack(store):
    from sim3d.reservoir.flow import FlowResult, FlowSettings, WellHistory

    history = WellHistory("P1", "producer")
    for day in range(4):
        history.days.append(float(day))
        history.oil_rate.append(1.0)
        history.water_rate.append(0.1)
        history.bhp.append(2.0e7)
        history.control.append("liquid_rate")
    result = FlowResult(days=[0.0, 3.0], pressure=[np.zeros((2, 2, 2))] * 2,
                        water_saturation=[np.full((2, 2, 2), 0.3)] * 2,
                        wells={"P1": history}, settings=FlowSettings(),
                        material_balance_error=1e-12, n_timesteps=4)
    store.save("run", config(), simulation=ScenarioStore.pack_flow(result))
    payload = store.load("run").simulation
    assert np.allclose(payload["days"], [0.0, 3.0])
    assert np.allclose(payload["well:P1:oil_rate"], 1.0)
    assert payload["well:P1:role"][0] == "producer"


def test_images_pack_and_unpack(store):
    class Result:
        def __init__(self, image):
            self.image = image

    images = {"baseline": Result(np.ones((2, 2, 2))),
              "combined": Result(np.zeros((2, 2, 2)))}
    store.save("run", config(), seismic=ScenarioStore.pack_images(images))
    unpacked = ScenarioStore.unpack_images(store.load("run").seismic)
    assert set(unpacked) == {"baseline", "combined"}
    assert np.allclose(unpacked["baseline"], 1.0)
