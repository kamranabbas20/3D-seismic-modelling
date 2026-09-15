"""Saving, loading, duplicating and deleting scenarios (requirements 13, 14).

Three stores, deliberately separate:

``scenario.json``
    The complete definition - geology, wells, completions, controls,
    schedules, fluids, simulation settings, rock physics, acquisition and
    imaging.  Human-readable, diffable, and small enough to email.
``simulation.npz``
    Flow results: pressure and saturation through time, and the well
    histories.
``seismic.npz``
    Migrated images and, optionally, the shot gathers.

Keeping them apart is what stops a configuration file becoming a gigabyte.
It also means a scenario can be re-run from its definition alone, that
results can be discarded without losing the setup, and that two scenarios
can be compared on definition alone before either has been simulated.

``view.json`` holds the display state - visible layers, camera, colour
ranges, the selected timestep.  It is saved with the scenario but is not
part of its identity: the content hash covers the science only, so
restoring a camera position never invalidates a result.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core.config import ExperimentConfig
from ..core.errors import ConfigError

MANIFEST = "manifest.json"
DEFINITION = "scenario.json"
VIEW = "view.json"
SIMULATION = "simulation.npz"
SEISMIC = "seismic.npz"

#: Characters allowed in a scenario name, so it is also a safe directory name.
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.")


def _check_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ConfigError("a scenario needs a name")
    bad = set(cleaned) - _SAFE
    if bad:
        raise ConfigError(
            f"scenario name {name!r} contains {sorted(bad)}; names double as "
            f"folder names, so letters, digits, spaces, dot, dash and "
            f"underscore only")
    return cleaned


@dataclass
class ViewState:
    """Display state (requirement 13's optional section).  Never scientific."""

    visible_layers: list = field(default_factory=list)
    visible_wells: list = field(default_factory=list)
    show_faults: bool = True
    show_volume: bool = True
    property_shown: str = "porosity"
    opacity: float = 0.35
    vertical_exaggeration: float = 1.5
    orthographic: bool = False
    camera: dict = field(default_factory=dict)
    active_day: float = 0.0
    colour_range: dict = field(default_factory=dict)
    cursor: list = field(default_factory=list)


@dataclass
class ScenarioInfo:
    """What a listing shows without opening a scenario."""

    name: str
    created: str
    modified: str
    config_hash: str
    has_simulation: bool
    has_seismic: bool
    description: str = ""

    def describe(self) -> str:
        stored = [n for n, present in (("flow", self.has_simulation),
                                       ("seismic", self.has_seismic)) if present]
        return (f"{self.name:28s} [{self.config_hash}] modified {self.modified[:19]}"
                f"  results: {', '.join(stored) if stored else 'none'}")


@dataclass
class Scenario:
    """A loaded scenario: definition, display state, and any stored results."""

    name: str
    config: ExperimentConfig
    view: ViewState = field(default_factory=ViewState)
    simulation: dict | None = None
    seismic: dict | None = None
    info: ScenarioInfo | None = None


class ScenarioStore:
    """A directory of scenarios."""

    def __init__(self, root: str | Path = "scenarios"):
        self.root = Path(root)

    # -- listing ---------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.root / _check_name(name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / DEFINITION).exists()

    def list(self) -> list[ScenarioInfo]:
        """Every scenario in the store, newest first."""
        if not self.root.is_dir():
            return []
        out = []
        for folder in sorted(self.root.iterdir()):
            manifest = folder / MANIFEST
            if not (folder / DEFINITION).exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            out.append(ScenarioInfo(
                name=folder.name,
                created=data.get("created", ""),
                modified=data.get("modified", ""),
                config_hash=data.get("config_hash", ""),
                description=data.get("description", ""),
                has_simulation=(folder / SIMULATION).exists(),
                has_seismic=(folder / SEISMIC).exists(),
            ))
        return sorted(out, key=lambda s: s.modified, reverse=True)

    # -- writing ---------------------------------------------------------
    def save(self, name: str, config: ExperimentConfig, *,
             view: ViewState | None = None, simulation: dict | None = None,
             seismic: dict | None = None, overwrite: bool = True) -> ScenarioInfo:
        """Write a scenario.

        ``simulation`` and ``seismic`` are mappings of name to array; passing
        ``None`` leaves whatever is already stored alone, so saving a
        configuration edit does not silently discard an overnight migration.
        Pass an empty mapping to clear them deliberately.
        """
        folder = self.path(name)
        if folder.exists() and not overwrite:
            raise ConfigError(
                f"scenario {name!r} already exists; use Save As with a new name, "
                f"or pass overwrite=True")
        folder.mkdir(parents=True, exist_ok=True)

        (folder / DEFINITION).write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=False, default=str),
            encoding="utf-8")
        (folder / VIEW).write_text(
            json.dumps(asdict(view or ViewState()), indent=2, default=str),
            encoding="utf-8")

        for payload, filename in ((simulation, SIMULATION), (seismic, SEISMIC)):
            if payload is None:
                continue
            target = folder / filename
            if not payload:
                target.unlink(missing_ok=True)
                continue
            np.savez_compressed(target, **{k: np.asarray(v)
                                           for k, v in payload.items()})

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        previous = {}
        if (folder / MANIFEST).exists():
            try:
                previous = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
        info = ScenarioInfo(
            name=folder.name, created=previous.get("created", now), modified=now,
            config_hash=config.short_hash, description=config.project.description,
            has_simulation=(folder / SIMULATION).exists(),
            has_seismic=(folder / SEISMIC).exists())
        (folder / MANIFEST).write_text(
            json.dumps(asdict(info), indent=2), encoding="utf-8")
        return info

    # -- reading ---------------------------------------------------------
    def load(self, name: str, results: bool = True) -> Scenario:
        """Read a scenario back.

        ``results=False`` reads the definition alone, which is what a
        listing or a comparison of setups needs and is far cheaper than
        pulling gigabytes of arrays off disk.
        """
        folder = self.path(name)
        definition = folder / DEFINITION
        if not definition.exists():
            known = [s.name for s in self.list()]
            raise ConfigError(
                f"no scenario named {name!r} in {self.root}; found {known}")
        config = ExperimentConfig.from_dict(
            json.loads(definition.read_text(encoding="utf-8")))

        view = ViewState()
        if (folder / VIEW).exists():
            try:
                view = ViewState(**json.loads((folder / VIEW).read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                pass

        simulation = seismic = None
        if results:
            simulation = self._load_arrays(folder / SIMULATION)
            seismic = self._load_arrays(folder / SEISMIC)

        info = next((s for s in self.list() if s.name == folder.name), None)
        return Scenario(name=folder.name, config=config, view=view,
                        simulation=simulation, seismic=seismic, info=info)

    @staticmethod
    def _load_arrays(path: Path) -> dict | None:
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    # -- management ------------------------------------------------------
    def duplicate(self, name: str, new_name: str, results: bool = False) -> ScenarioInfo:
        """Copy a scenario under a new name, definition only by default.

        A duplicate is almost always made in order to change something,
        which invalidates the results anyway; copying gigabytes of arrays
        that the next edit will discard is the wrong default. Pass
        ``results=True`` to keep them.
        """
        source = self.path(name)
        if not (source / DEFINITION).exists():
            raise ConfigError(f"no scenario named {name!r} to duplicate")
        target = self.path(new_name)
        if target.exists():
            raise ConfigError(f"scenario {new_name!r} already exists")
        target.mkdir(parents=True)
        for filename in (DEFINITION, VIEW):
            if (source / filename).exists():
                shutil.copy2(source / filename, target / filename)
        if results:
            for filename in (SIMULATION, SEISMIC):
                if (source / filename).exists():
                    shutil.copy2(source / filename, target / filename)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        config = ExperimentConfig.from_dict(
            json.loads((target / DEFINITION).read_text(encoding="utf-8")))
        info = ScenarioInfo(
            name=target.name, created=now, modified=now,
            config_hash=config.short_hash, description=config.project.description,
            has_simulation=(target / SIMULATION).exists(),
            has_seismic=(target / SEISMIC).exists())
        (target / MANIFEST).write_text(json.dumps(asdict(info), indent=2),
                                       encoding="utf-8")
        return info

    def rename(self, name: str, new_name: str) -> ScenarioInfo:
        source, target = self.path(name), self.path(new_name)
        if not (source / DEFINITION).exists():
            raise ConfigError(f"no scenario named {name!r} to rename")
        if target.exists():
            raise ConfigError(f"scenario {new_name!r} already exists")
        source.rename(target)
        return next(s for s in self.list() if s.name == target.name)

    def delete(self, name: str) -> None:
        folder = self.path(name)
        if not (folder / DEFINITION).exists():
            raise ConfigError(f"no scenario named {name!r} to delete")
        shutil.rmtree(folder)

    # -- packing helpers -------------------------------------------------
    @staticmethod
    def pack_flow(result) -> dict[str, np.ndarray]:
        """Flatten a :class:`~sim3d.reservoir.flow.FlowResult` for storage."""
        out: dict[str, np.ndarray] = {
            "days": np.asarray(result.days, dtype=float),
            "pressure": np.asarray(result.pressure, dtype=np.float32),
            "water_saturation": np.asarray(result.water_saturation, dtype=np.float32),
        }
        for name, history in result.wells.items():
            arrays = history.arrays()
            for key, values in arrays.items():
                out[f"well:{name}:{key}"] = values.astype(float)
            out[f"well:{name}:role"] = np.array([history.role])
        return out

    @staticmethod
    def pack_images(images: dict) -> dict[str, np.ndarray]:
        """Flatten migrated images for storage."""
        return {f"image:{name}": np.asarray(result.image, dtype=np.float32)
                for name, result in images.items()}

    @staticmethod
    def unpack_images(payload: dict | None) -> dict[str, np.ndarray]:
        if not payload:
            return {}
        return {key.split(":", 1)[1]: value for key, value in payload.items()
                if key.startswith("image:")}
