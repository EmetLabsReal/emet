"""Append-only certificate store.

Each line is a sealed JSON certificate. The seal is the primary key.
Content-addressed, deterministic, immutable. Same inputs always
produce the same seal, so deduplication is free.

The file is a JSONL log. The spatial index is built lazily in memory.
If the file gets corrupted, every line is independently verifiable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from emet.certificate import EmetCertificate, to_json, from_json
from emet.portrait import PhasePoint, PhasePortrait, Regime


class CertificateStore:
    """Append-only store of certified phase points."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._seals: set[str] | None = None
        self._entries: list[dict[str, Any]] | None = None

    def _load(self) -> None:
        """Load all entries from the JSONL file."""
        if self._entries is not None:
            return
        self._entries = []
        self._seals = set()
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                seal = entry.get("seal", "")
                if seal and seal not in self._seals:
                    self._entries.append(entry)
                    self._seals.add(seal)
            except json.JSONDecodeError:
                continue

    def append(
        self,
        cert: EmetCertificate,
        *,
        domain: str | None = None,
        params: dict[str, Any] | None = None,
        beta: float | None = None,
    ) -> bool:
        """Append a certified point. Returns False if seal already exists."""
        self._load()
        assert self._seals is not None
        assert self._entries is not None

        if cert.seal in self._seals:
            return False

        entry = json.loads(to_json(cert))
        if domain is not None:
            entry["_domain"] = domain
        if params is not None:
            entry["_params"] = params
        if beta is not None:
            entry["_beta"] = beta

        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        with self.path.open("a") as f:
            f.write(line + "\n")

        self._entries.append(entry)
        self._seals.add(cert.seal)
        return True

    def __len__(self) -> int:
        self._load()
        assert self._entries is not None
        return len(self._entries)

    def __iter__(self):
        self._load()
        assert self._entries is not None
        return iter(self._entries)

    def _to_phase_point(self, entry: dict[str, Any]) -> PhasePoint:
        """Convert a stored entry to a PhasePoint."""
        gamma = entry.get("gamma", 0.0)
        lam = entry.get("lambda", entry.get("lambda_", 0.0))
        beta = entry.get("_beta")
        return PhasePoint.from_values(gamma, lam, beta=beta)

    def query_neighborhood(
        self,
        gamma: float,
        lambda_: float,
        radius: float,
    ) -> list[PhasePoint]:
        """All certified points within radius in (gamma, lambda) space."""
        self._load()
        assert self._entries is not None
        r2 = radius * radius
        results = []
        for entry in self._entries:
            g = entry.get("gamma", 0.0)
            l = entry.get("lambda", entry.get("lambda_", 0.0))
            if (g - gamma) ** 2 + (l - lambda_) ** 2 <= r2:
                results.append(self._to_phase_point(entry))
        return results

    def query_trajectory(
        self,
        domain: str,
        param_name: str,
    ) -> list[tuple[float, PhasePoint]]:
        """All points from a domain, ordered by named parameter."""
        self._load()
        assert self._entries is not None
        trajectory = []
        for entry in self._entries:
            if entry.get("_domain") != domain:
                continue
            params = entry.get("_params") or {}
            val = params.get(param_name)
            if val is None:
                continue
            trajectory.append((float(val), self._to_phase_point(entry)))
        trajectory.sort(key=lambda t: t[0])
        return trajectory

    def portrait(self) -> PhasePortrait:
        """The full portrait from all stored certificates."""
        self._load()
        assert self._entries is not None
        points = [self._to_phase_point(e) for e in self._entries]
        return PhasePortrait(points)

    def seals(self) -> set[str]:
        """All stored seals."""
        self._load()
        assert self._seals is not None
        return set(self._seals)
