"""Tests for the certificate store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import emet
from emet.certificate import certify
from emet.portrait import Regime
from emet.store import CertificateStore


def _make_cert(g_squared: float = 8.0, beta: float | None = 8.0):
    """Helper: build Yang-Mills, decide, certify."""
    from emet.domains.yang_mills import build_plaquette_blocks
    H, ret, omit, _ = build_plaquette_blocks(g_squared=g_squared)
    report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    return certify(
        H, ret, omit, report,
        domain="yang_mills",
        params={"g_squared": g_squared},
        beta=beta,
    )


class TestCertificateStore:

    def test_append_and_len(self):
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "certs.jsonl")
            cert = _make_cert(g_squared=8.0)
            assert store.append(cert, domain="yang_mills",
                                params={"g_squared": 8.0}, beta=8.0)
            assert len(store) == 1

    def test_dedup_by_seal(self):
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "certs.jsonl")
            cert = _make_cert(g_squared=8.0)
            assert store.append(cert)
            assert not store.append(cert)
            assert len(store) == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "certs.jsonl"
            store1 = CertificateStore(path)
            store1.append(_make_cert(g_squared=8.0))
            store1.append(_make_cert(g_squared=16.0))
            # Reload from disk
            store2 = CertificateStore(path)
            assert len(store2) == 2

    def test_portrait(self):
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "certs.jsonl")
            store.append(_make_cert(8.0), domain="yang_mills",
                         params={"g_squared": 8.0}, beta=8.0)
            store.append(_make_cert(16.0), domain="yang_mills",
                         params={"g_squared": 16.0}, beta=16.0)
            portrait = store.portrait()
            assert len(portrait.points) == 2

    def test_query_neighborhood(self):
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "certs.jsonl")
            c1 = _make_cert(8.0)
            c2 = _make_cert(64.0)
            store.append(c1, beta=8.0)
            store.append(c2, beta=64.0)
            near = store.query_neighborhood(c1.gamma, c1.lambda_, radius=1.0)
            assert len(near) >= 1

    def test_query_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "certs.jsonl")
            for g2 in [4.0, 8.0, 16.0, 32.0]:
                store.append(
                    _make_cert(g2), domain="yang_mills",
                    params={"g_squared": g2}, beta=g2,
                )
            traj = store.query_trajectory("yang_mills", "g_squared")
            assert len(traj) == 4
            g2_vals = [t[0] for t in traj]
            assert g2_vals == sorted(g2_vals)

    def test_seals(self):
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "certs.jsonl")
            c = _make_cert(8.0)
            store.append(c)
            assert c.seal in store.seals()
