"""Storage for the source wavefield during RTM (spec section 75).

The complete time-dependent wavefield is never assumed to fit in memory.
A store holds ``n_saved`` pressure snapshots, either in RAM or in a
memory-mapped file on disk, and the caller decides which by setting a
budget rather than by guessing.

Time decimation is the other half of the strategy.  The imaging condition
correlates two wavefields, so its product oscillates at up to twice the
maximum wavefield frequency; sampling it at ``dt_save <= 1/(4 f_max)``
keeps that product unaliased.  :func:`safe_decimation` computes the
largest legal stride, and asking for a coarser one raises instead of
quietly degrading the image.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ..core.errors import ConfigError


def safe_decimation(dt: float, fmax: float) -> int:
    """Largest time-step stride that keeps the imaging condition unaliased.

    The correlation ``S(t) R(t)`` contains frequencies up to ``2 * fmax``,
    so its Nyquist limit is ``dt_save <= 1 / (4 * fmax)``.
    """
    if dt <= 0 or fmax <= 0:
        raise ConfigError(f"dt and fmax must be positive, got {dt}, {fmax}")
    return max(1, int(np.floor(1.0 / (4.0 * fmax * dt))))


class WavefieldStore:
    """A sequence of pressure snapshots, in RAM or memory-mapped on disk.

    Parameters
    ----------
    shape:
        Grid shape of one snapshot.
    n_saved:
        Number of snapshots to hold.
    dtype:
        Snapshot dtype, normally the solver's.
    max_ram_bytes:
        Above this size the store spills to a memory-mapped file.
    directory:
        Where to put the memmap; a temporary directory by default.
    """

    def __init__(self, shape: tuple[int, int, int], n_saved: int, dtype=np.float32,
                 max_ram_bytes: int = 2 * 2**30, directory: str | os.PathLike | None = None):
        if n_saved < 1:
            raise ConfigError(f"n_saved must be >= 1, got {n_saved}")
        self.shape = tuple(int(s) for s in shape)
        self.n_saved = int(n_saved)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.n_saved * self.dtype.itemsize
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None

        full = (self.n_saved, *self.shape)
        if self.nbytes <= max_ram_bytes:
            self.backing = "memory"
            self.data = np.zeros(full, dtype=self.dtype)
        else:
            self.backing = "memmap"
            if directory is None:
                self._tempdir = tempfile.TemporaryDirectory(prefix="sim3d_wavefield_")
                directory = self._tempdir.name
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.path = Path(directory) / f"source_wavefield_{os.getpid()}.dat"
            self.data = np.lib.format.open_memmap(
                self.path, mode="w+", dtype=self.dtype, shape=full
            )

    def __len__(self) -> int:
        return self.n_saved

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index]

    def save(self, index: int, field: np.ndarray) -> None:
        """Copy ``field`` into slot ``index``."""
        self.data[index] = field

    def flush(self) -> None:
        if self.backing == "memmap":
            self.data.flush()

    def close(self) -> None:
        """Release the memmap and delete any temporary file."""
        self.data = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
            self.path = None

    def __enter__(self) -> "WavefieldStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> str:
        return (
            f"wavefield store: {self.n_saved} snapshots of {self.shape}, "
            f"{self.dtype.name}, {self.nbytes / 2**30:.2f} GiB, backing={self.backing}"
            + (f" at {self.path}" if self.path else "")
        )
