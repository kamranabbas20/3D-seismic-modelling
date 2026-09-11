"""Explicit dependency graph over the modelling stages (requirement 17).

The alternative - a single "Run Everything" button, or invalidating every
cached result whenever any setting changes - makes the expensive half of
this application unusable.  Extending a run from five years to ten has no
business rebuilding the geological model, and changing the wavelet has no
business re-running the flow simulation.

Each stage declares two things: the **configuration sections it reads**,
and the **stages it depends on**.  Everything else follows.  Comparing two
configurations section by section gives the set of directly invalidated
stages; the transitive closure over the edges gives the rest.

The graph is the authority, not a comment: :func:`stale_stages` is what the
pipeline and the GUI both call, so a stage that forgets to declare an input
shows up as a wrong answer in a test rather than as a stale cache in front
of a user.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .errors import ConfigError


@dataclass(frozen=True)
class Stage:
    """One node: what it reads, and what it is built on."""

    name: str
    sections: tuple[str, ...] = ()
    depends: tuple[str, ...] = ()
    #: Rough cost, used only to explain to a user what a change will cost.
    cost: str = "cheap"


#: The modelling graph.  Order is topological, so a stage's dependencies
#: always appear before it.
STAGES: tuple[Stage, ...] = (
    Stage("geology", sections=("domains", "geology"), cost="cheap"),
    Stage("wells", sections=("wells",), cost="cheap"),
    Stage("completions", depends=("geology", "wells"), cost="cheap"),
    Stage("controls", sections=("reservoir",), depends=("completions",), cost="cheap"),
    Stage("flow", sections=("reservoir", "simulation"),
          depends=("controls",), cost="moderate"),
    Stage("states", sections=("reservoir", "simulation"),
          depends=("flow",), cost="cheap"),
    Stage("rockphysics", sections=("rock_physics",),
          depends=("states", "geology"), cost="moderate"),
    Stage("acquisition", sections=("acquisition", "domains"), cost="cheap"),
    Stage("gathers", sections=("solver", "source"),
          depends=("rockphysics", "acquisition"), cost="expensive"),
    Stage("images", sections=("imaging",), depends=("gathers",), cost="expensive"),
    # The sparse synthetic reads the imaging section for its trace layout but
    # not the gathers: it never propagates a wavefield, which is exactly why
    # it is cheap and why it depends on the wells that name its traces.
    Stage("synthetic", sections=("synthetic", "source", "solver"),
          depends=("rockphysics", "wells"), cost="cheap"),
    # The synthetic volume needs no acquisition and no wells: it converts the
    # earth model itself, column by column.
    Stage("sim2seis", sections=("sim2seis", "source", "solver"),
          depends=("rockphysics",), cost="moderate"),
    Stage("fourd", depends=("images", "rockphysics"), cost="cheap"),
)

_BY_NAME = {stage.name: stage for stage in STAGES}


def stage(name: str) -> Stage:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ConfigError(
            f"unknown stage {name!r}; the graph has {sorted(_BY_NAME)}") from None


def dependents(name: str) -> set[str]:
    """Every stage that would have to be redone if ``name`` were redone."""
    stage(name)
    out: set[str] = set()
    frontier = {name}
    while frontier:
        current = frontier.pop()
        for candidate in STAGES:
            if current in candidate.depends and candidate.name not in out:
                out.add(candidate.name)
                frontier.add(candidate.name)
    return out


def section_hashes(config) -> dict[str, str]:
    """A digest per configuration section.

    ``output`` is excluded: it holds display and file-location settings,
    and no scientific stage reads it.
    """
    from dataclasses import asdict, fields

    out = {}
    for f in fields(config):
        if f.name in getattr(config, "NON_SCIENTIFIC", ()):
            continue
        value = getattr(config, f.name)
        payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               default=str)
        out[f.name] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return out


def changed_sections(before: dict[str, str], after: dict[str, str]) -> set[str]:
    """Section names whose contents differ between two configurations."""
    return {name for name in set(before) | set(after)
            if before.get(name) != after.get(name)}


def stale_stages(before, after) -> set[str]:
    """Stages invalidated by the difference between two configurations.

    ``before`` and ``after`` may be configurations or section-hash mappings.
    """
    a = before if isinstance(before, dict) else section_hashes(before)
    b = after if isinstance(after, dict) else section_hashes(after)
    changed = changed_sections(a, b)
    direct = {s.name for s in STAGES if set(s.sections) & changed}
    stale = set(direct)
    for name in direct:
        stale |= dependents(name)
    return stale


def surviving_stages(before, after) -> set[str]:
    """Stages whose results remain valid - the useful half of the answer."""
    return {s.name for s in STAGES} - stale_stages(before, after)


def explain(before, after) -> str:
    """Human-readable statement of what a change costs."""
    a = before if isinstance(before, dict) else section_hashes(before)
    b = after if isinstance(after, dict) else section_hashes(after)
    changed = changed_sections(a, b)
    if not changed:
        return "Nothing scientific changed; every cached result stays valid."
    stale = stale_stages(a, b)
    kept = sorted(surviving_stages(a, b), key=lambda n: [s.name for s in STAGES].index(n))
    ordered = [s for s in STAGES if s.name in stale]
    lines = [f"Changed: {', '.join(sorted(changed))}",
             f"Must rerun ({len(ordered)}):"]
    lines += [f"  {s.name:12s} ({s.cost})" for s in ordered]
    lines.append(f"Still valid: {', '.join(kept) if kept else 'nothing'}")
    return "\n".join(lines)
