"""End-to-end pipeline and CLI behaviour."""

import numpy as np
import pytest

from sim3d.cli import build_parser, main
from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import ConfigError, InfeasibleExperiment
from sim3d.experiments.pipeline import STAGES, Pipeline
from sim3d.fourd.scenarios import SCENARIO_NAMES


def tiny_config() -> ExperimentConfig:
    """The smallest configuration that still exercises the whole chain."""
    config = ExperimentConfig()
    config.project.name = "tiny"
    config.domains.geology_bounds = [[0, 1200], [0, 1200], [0, 1400]]
    config.domains.geology_spacing = [50.0, 50.0, 25.0]
    config.domains.propagation_bounds = [[100, 1100], [100, 1100], [400, 1400]]
    config.domains.propagation_spacing = [50.0, 50.0, 50.0]
    config.domains.target_bounds = [[400, 800], [400, 800], [850, 1050]]
    config.geology.template = "flat"
    config.geology.parameters = {"z_reservoir": 900.0, "gross": 100.0}
    config.wells.pattern = "line_drive"
    config.wells.parameters = {
        "centre": [600.0, 600.0], "n_injectors": 1, "n_producers": 1,
        "separation": 500.0, "perforation": [900.0, 1000.0],
    }
    config.reservoir.baseline.temperature = 70.0
    config.reservoir.scenario.pressure = [
        {"well": "I1", "delta_p_bar": 40.0, "radius": [400.0, 400.0, 120.0]}]
    config.reservoir.scenario.water_fronts = [
        {"well": "I1", "target_sw": 0.75, "radius": [200.0, 200.0, 80.0]}]
    config.reservoir.scenario.gas = [
        {"well": "P1", "target_sg": 0.08, "radius": [150.0, 150.0, 60.0]}]
    config.rock_physics.temperature = 70.0
    config.source.frequency = 4.0
    config.solver.record_length = 0.5
    config.solver.pml_nodes = 8
    config.acquisition.centre = [600.0, 600.0]
    config.acquisition.receiver_spacing = 200.0
    config.acquisition.receiver_extent = 200.0
    config.acquisition.source_spacing = 400.0
    config.acquisition.source_line_spacing = 400.0
    config.acquisition.source_extent = 100.0  # one shot at this spacing
    config.acquisition.receiver_depth = 850.0
    config.acquisition.source_depth = 830.0
    return config


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(tiny_config())


def test_the_flow_path_and_the_mechanistic_path_both_produce_four_states():
    """Two ways to reach the four earth models, and both must isolate.

    The flow simulator is the default and gives a physically consistent
    history; the mechanistic generator remains the only way to impose free
    gas, which a two-phase model cannot produce.
    """
    for source in ("flow", "mechanistic"):
        config = tiny_config()
        config.reservoir.source = source
        states = Pipeline(config).reservoir()
        states.check_isolation()
        assert set(dict(states.items())) == set(SCENARIO_NAMES)


def test_stages_run_in_dependency_order_and_cache(pipeline):
    result = pipeline.run(("geology", "reservoir", "rockphysics", "acquisition"))
    assert result.geology is not None
    assert set(result.earth.models) == set(SCENARIO_NAMES)
    first = result.geology
    pipeline.geology()
    assert pipeline.result.geology is first  # rebuilt nothing


def test_unknown_stage_is_refused(pipeline):
    with pytest.raises(ConfigError, match="unknown stage"):
        pipeline.run(("teleport",))
    assert "migrate" in STAGES


def test_the_practical_fmax_is_well_above_the_peak_frequency(pipeline):
    assert pipeline.fmax > 2.0 * pipeline.config.source.frequency


def test_all_four_earth_models_share_one_time_step(pipeline):
    """Gathers on different time axes cannot be differenced."""
    models, dt = pipeline.propagation_models()
    assert set(models) == set(SCENARIO_NAMES)
    assert dt > 0
    # The four models really are different earths - though not in Vmax, which
    # the deep high-velocity unit sets and no reservoir change touches. That
    # is exactly why one shared dt is safe to pin.
    base = models["baseline"].vp
    for name in SCENARIO_NAMES[1:]:
        assert not np.array_equal(models[name].vp, base)
    assert models["saturation_only"].vmin != models["baseline"].vmin


def test_qc_reports_on_every_scenario(pipeline):
    report = pipeline.qc().report()
    for name in SCENARIO_NAMES:
        assert f"[{name}]" in report


def test_the_cost_estimate_counts_every_scenario(pipeline):
    estimate = pipeline.plan()
    assert estimate.n_scenarios == 4
    assert estimate.propagations == estimate.n_sources * 3 * 4


def test_an_over_tight_budget_refuses_with_alternatives():
    config = tiny_config()
    config.budget.max_ram_gb = 1e-6
    with pytest.raises(InfeasibleExperiment, match="reduce Fmax"):
        Pipeline(config).plan()


def test_unimplemented_acquisition_and_imaging_are_named():
    config = tiny_config()
    config.acquisition.type = "streamer"
    with pytest.raises(ConfigError, match="not implemented"):
        Pipeline(config).acquisition()

    config = tiny_config()
    config.imaging.method = "kirchhoff"
    pipe = Pipeline(config)
    pipe.result.gathers["baseline"] = []
    with pytest.raises(ConfigError, match="not implemented"):
        pipe.migrate()


def test_the_migration_model_records_a_deliberate_velocity_error():
    config = tiny_config()
    config.imaging.velocity_scale = 1.05
    pipe = Pipeline(config)
    models, _ = pipe.propagation_models()
    migration = pipe.migration_model(models["baseline"])
    assert migration.vmax == pytest.approx(1.05 * models["baseline"].vmax)
    assert any("deliberate experiment" in n for n in pipe.result.notes)


def test_preview_screens_every_scenario(pipeline):
    previews = pipeline.preview()
    assert set(previews) == set(SCENARIO_NAMES)
    base = previews["baseline"].depth_traces
    assert np.any(previews["combined"].depth_traces != base)
    assert "Not Full 3D Wave Modelling" in previews["baseline"].label


@pytest.mark.slow
def test_the_whole_chain_runs_and_decomposes(pipeline):
    """Forward-model, migrate and decompose four earth models."""
    pipeline.simulate()
    assert set(pipeline.result.gathers) == set(SCENARIO_NAMES)
    for records in pipeline.result.gathers.values():
        assert all(np.all(np.isfinite(r.traces)) for r in records)

    # The four scenarios really did produce different data.
    base = pipeline.result.gathers["baseline"][0].traces
    assert np.max(np.abs(pipeline.result.gathers["combined"][0].traces - base)) > 0

    pipeline.migrate()
    assert set(pipeline.result.images) == set(SCENARIO_NAMES)
    parts = pipeline.decompose()
    seismic = parts["seismic"]["rtm"]
    assert np.allclose(seismic["d_interaction"],
                       seismic["d_combined"] - seismic["d_sum"])
    assert set(parts["seismic"]["nrms"]) == set(SCENARIO_NAMES[1:])


@pytest.mark.slow
def test_the_4d_null_test(tmp_path):
    """Spec section 131: the mandatory system-level validation.

    Make the monitor identical to the baseline, then run *independent*
    forward models, acquisitions and migrations for both. The 4D difference
    must be numerical noise and nothing else.

    This is not a tautology about arithmetic. It is the test that catches
    state leaking between runs - CPML memory variables that were not reset,
    a wavefield store reused across shots, an accumulator carried over from
    the previous scenario - any of which would put a fictitious anomaly in
    every 4D difference the platform ever produced.
    """
    config = tiny_config()
    # The mechanistic generator, because it can be told to perturb nothing at
    # all. The flow simulator cannot: wells that are open move fluid, which
    # is the point of it.
    config.reservoir.source = "mechanistic"
    config.reservoir.scenario.time_state = "T0"   # zero perturbation
    pipe = Pipeline(config)

    states = pipe.reservoir()
    for name in SCENARIO_NAMES[1:]:
        d = states[name].difference(states.baseline)
        assert max(np.max(np.abs(v)) for v in d.values()) == 0.0

    pipe.simulate()
    base_traces = pipe.result.gathers["baseline"][0].traces
    for name in SCENARIO_NAMES[1:]:
        other = pipe.result.gathers[name][0].traces
        assert np.array_equal(other, base_traces), (
            f"the {name} gather differs from the baseline despite an identical "
            f"earth model; solver state is leaking between runs")

    pipe.migrate()
    base_image = pipe.result.images["baseline"].image
    peak = np.max(np.abs(base_image))
    assert peak > 0, "the null test needs a non-trivial image to be meaningful"
    for name in SCENARIO_NAMES[1:]:
        residual = np.max(np.abs(pipe.result.images[name].image - base_image))
        assert residual / peak < 1e-12

    parts = pipe.decompose()
    assert np.max(np.abs(parts["seismic"]["rtm"]["d_combined"])) / peak < 1e-12
    assert all(v == pytest.approx(0.0, abs=1e-9)
               for v in parts["seismic"]["nrms"].values())


def test_the_cli_parser_exposes_every_documented_command():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set(actions[0].choices)
    assert {"describe", "physics", "build", "rockphysics", "qc", "plan",
            "benchmark", "preview", "simulate", "migrate", "decompose",
            "run"} <= commands


def test_the_cli_physics_command_runs(capsys):
    assert main(["physics"]) == 0
    assert "Not Full 3D Wave Modelling" in capsys.readouterr().out


def test_the_cli_describe_command_runs_on_a_shipped_example(capsys):
    assert main(["describe", "examples/configs/demo_small.yaml"]) == 0
    out = capsys.readouterr().out
    assert "demo_small" in out and "Domain hierarchy" in out


def test_the_cli_reports_a_bad_configuration_without_a_traceback(capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("source:\n  freqency: 20\n")
    assert main(["describe", str(bad)]) == 1
    assert "unknown key" in capsys.readouterr().err


# ------------------------------------------------- the sparse synthetic mode
def test_the_synthetic_stage_puts_one_trace_at_each_well():
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    pipe = Pipeline(config)
    synthetics = pipe.synthetic()
    assert set(synthetics) == set(SCENARIO_NAMES)
    base = synthetics["baseline"]
    assert base.names == tuple(w.name for w in pipe.wells())
    assert base.traces.shape[0] == len(pipe.wells())


def test_the_synthetic_traces_index_the_propagation_grid_not_the_target():
    """The layout comes from the target, the indices from the model sampled.

    Conflating the two silently reads the wrong column, which is the kind
    of bug that produces a plausible trace at the wrong place.
    """
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    pipe = Pipeline(config)
    models, _ = pipe.propagation_models()
    grid = models["baseline"].grid
    for site in pipe.synthetic()["baseline"].locations:
        assert 0 <= site.ix < grid.nx and 0 <= site.iy < grid.ny
        assert grid.origin[0] + site.ix * grid.dx == pytest.approx(site.x, abs=grid.dx)


def test_a_flood_moves_the_synthetic_trace():
    """The mode has to see the change it exists to screen for."""
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    pipe = Pipeline(config)
    synthetics = pipe.synthetic()
    difference = synthetics["saturation_only"].traces - synthetics["baseline"].traces
    assert np.abs(difference).max() > 0.0


def test_the_synthetic_costs_far_less_than_the_full_preview_cube():
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    pipe = Pipeline(config)
    pipe.synthetic()
    pipe.preview()
    assert pipe.result.timings["synthetic"] < pipe.result.timings["preview"]


def test_the_decomposition_keeps_the_two_seismic_modes_apart():
    """A reader must be able to tell a trace number from an image number."""
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    pipe = Pipeline(config)
    pipe.synthetic()
    out = pipe.decompose()
    assert "sparse" in out["seismic"]
    assert "rtm" not in out["seismic"]        # nothing was migrated
    assert set(out["seismic"]["sparse_nrms"]) == set(SCENARIO_NAMES[1:])
    per_trace = out["seismic"]["sparse_nrms_per_trace"]["saturation_only"]
    assert set(per_trace) == set(pipe.synthetic()["baseline"].names)


def test_the_trace_layout_can_be_a_lattice_of_exactly_k():
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    config.synthetic.layout = "grid"
    config.synthetic.count = 7
    pipe = Pipeline(config)
    assert pipe.synthetic()["baseline"].n_traces == 7


def test_the_trace_layout_can_be_explicit_points():
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    config.synthetic.layout = "points"
    config.synthetic.points = [[500.0, 500.0], [700.0, 700.0]]
    pipe = Pipeline(config)
    base = pipe.synthetic()["baseline"]
    assert base.names == ("T1", "T2")
    assert base.locations[0].x == 500.0


def test_the_synthetic_cli_command_runs(capsys, tmp_path):
    config = tiny_config()
    config.reservoir.source = "mechanistic"
    path = tmp_path / "tiny.yaml"
    config.save(path)
    assert main(["synthetic", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "Not Full 3D Wave Modelling" in printed
    assert "NRMS against baseline" in printed
