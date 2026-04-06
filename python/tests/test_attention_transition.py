"""Low-rank attention transition: chi phase transition at n = k + O(sqrt(k)).

1. Full-rank random keys: chi > 1 for 50/50 partition at all n.
2. Low-rank keys: transition from chi < 1 to chi > 1 near n = k.
3. Transition location scales with effective rank k, not ambient d.
4. Variance of H_eff peaks near the transition.
"""

import numpy as np
import pytest


def _make_gram(K: np.ndarray, eta: float) -> np.ndarray:
    n = K.shape[0]
    H = K @ K.T + eta * np.eye(n)
    return 0.5 * (H + H.T)


def _chi(H: np.ndarray, retained: list[int], omitted: list[int]) -> float:
    P, Q = sorted(retained), sorted(omitted)
    H_PQ = H[np.ix_(P, Q)]
    H_QQ = H[np.ix_(Q, Q)]
    lam = np.linalg.norm(H_PQ, ord=2)
    gam = np.linalg.svd(H_QQ, compute_uv=False)[-1]
    if gam <= 1e-15:
        return float("inf")
    return (lam / gam) ** 2


def _pca_partition(H: np.ndarray, n_retain: int):
    diag = np.abs(np.diag(H))
    order = np.argsort(-diag)
    ret = sorted(order[:n_retain].tolist())
    omi = sorted(order[n_retain:].tolist())
    return ret, omi


def _low_rank_keys(n, d, k, sigma_s, sigma_n, rng):
    U = rng.standard_normal((n, k))
    U, _, _ = np.linalg.svd(U, full_matrices=False)
    V_raw = rng.standard_normal((d, k))
    V_orth, _, _ = np.linalg.svd(V_raw, full_matrices=False)
    V = V_orth.T
    S = sigma_s * np.eye(k)
    return U @ S @ V + sigma_n * rng.standard_normal((n, d))


class TestFullRankSupercritical:
    """Full-rank random keys: chi > 1 everywhere with 50/50 split."""

    def test_all_supercritical(self):
        rng = np.random.default_rng(42)
        d = 16
        eta = 1.0
        for n in [d, 2 * d, 4 * d]:
            chis = []
            for _ in range(20):
                K = rng.standard_normal((n, d))
                H = _make_gram(K, eta)
                half = n // 2
                ret = list(range(half))
                omi = list(range(half, n))
                chis.append(_chi(H, ret, omi))
            assert np.mean(chis) > 1.0, f"Expected supercritical at n={n}"


class TestLowRankTransition:
    """Low-rank keys: sharp transition controlled by k."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.d = 64
        self.eta = 1.0
        self.sigma_s = 5.0
        self.sigma_n = 0.1
        self.n_trials = 30

    def _subcritical_fraction(self, k, n, seed=42):
        rng = np.random.default_rng(seed)
        sub = 0
        for _ in range(self.n_trials):
            K = _low_rank_keys(n, self.d, k, self.sigma_s, self.sigma_n, rng)
            H = _make_gram(K, self.eta)
            n_ret = min(k, n - 1)
            ret, omi = _pca_partition(H, n_ret)
            if _chi(H, ret, omi) < 1:
                sub += 1
        return sub / self.n_trials

    def test_subcritical_below_threshold(self):
        """At n = k + 1, majority of trials should be subcritical."""
        for k in [4, 8, 16]:
            frac = self._subcritical_fraction(k, k + 1)
            assert frac > 0.5, f"k={k}: expected >50% subcritical at n=k+1, got {frac:.0%}"

    def test_supercritical_above_threshold(self):
        """At n = 2k, should be overwhelmingly supercritical."""
        for k in [4, 8, 16]:
            frac = self._subcritical_fraction(k, 2 * k)
            assert frac < 0.1, f"k={k}: expected <10% subcritical at n=2k, got {frac:.0%}"

    def test_transition_scales_with_k(self):
        """Transition n* should scale linearly with k."""
        transitions = {}
        for k in [4, 8, 16]:
            prev_frac = 1.0
            for n in range(k + 1, 4 * k):
                frac = self._subcritical_fraction(k, n)
                if prev_frac >= 0.5 and frac < 0.5:
                    transitions[k] = n
                    break
                prev_frac = frac

        assert len(transitions) >= 2, "Need at least 2 transition points"

        # n*/k should be in [1.0, 2.0] for all k
        for k, n_star in transitions.items():
            ratio = n_star / k
            assert 1.0 <= ratio <= 2.0, (
                f"k={k}: n*/k={ratio:.2f} outside [1.0, 2.0]"
            )

        # Ratios should be roughly similar (transition scales linearly)
        ratios = [n_star / k for k, n_star in sorted(transitions.items())]
        spread = max(ratios) - min(ratios)
        assert spread < 0.5, (
            f"Transition ratios too spread: {ratios} (spread={spread:.2f})"
        )


class TestVariancePeak:
    """Variance of H_eff peaks near the transition."""

    def test_peak_near_transition(self):
        d, k = 64, 8
        eta = 1.0
        sigma_s, sigma_n = 5.0, 0.1
        n_trials = 40
        rng = np.random.default_rng(2026)

        variances = {}
        for n in [k + 1, k + 3, k + 5, 2 * k, 4 * k]:
            h_effs = []
            for _ in range(n_trials):
                K = _low_rank_keys(n, d, k, sigma_s, sigma_n, rng)
                H = _make_gram(K, eta)
                n_ret = min(k, n - 1)
                ret, omi = _pca_partition(H, n_ret)
                P, Q = sorted(ret), sorted(omi)
                H_PP = H[np.ix_(P, P)]
                H_QQ = H[np.ix_(Q, Q)]
                H_PQ = H[np.ix_(P, Q)]
                try:
                    R = H_PQ @ np.linalg.inv(H_QQ) @ H_PQ.T
                    h_effs.append((H_PP - R).flatten())
                except np.linalg.LinAlgError:
                    pass

            if len(h_effs) >= 2:
                flat = np.array(h_effs)
                variances[n] = np.mean(np.var(flat, axis=0))

        # The variance at n near the transition (k+3 to k+5)
        # should be larger than variance far above threshold (4k)
        near = max(variances.get(k + 3, 0), variances.get(k + 5, 0))
        far = variances.get(4 * k, 0)
        assert near > far, (
            f"Variance near transition ({near:.4f}) should exceed "
            f"variance far above ({far:.4f})"
        )
