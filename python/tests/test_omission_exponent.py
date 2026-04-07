"""Tests for the quantitative claims in the omission exponent paper.

Every assert here corresponds to a numbered result in paper/src/omission.tex.
If any of these fail, the paper's claims are wrong.
"""

import numpy as np
import pytest
import emet
from emet.domains.yang_mills import build_plaquette_blocks
from emet.domains.kahan import certified_subcritical


# ---------------------------------------------------------------------------
# Helpers (same as examples/)
# ---------------------------------------------------------------------------

def radial_hamiltonian(n_grid, r_max, Z=1.0, l=0, mass=1.0):
    dr = r_max / (n_grid + 1)
    r = np.array([(i + 1) * dr for i in range(n_grid)])
    kinetic = np.zeros((n_grid, n_grid))
    coeff = 1.0 / (2.0 * mass * dr**2)
    for i in range(n_grid):
        kinetic[i, i] = 2.0 * coeff
        if i > 0:
            kinetic[i, i - 1] = -coeff
        if i < n_grid - 1:
            kinetic[i, i + 1] = -coeff
    potential = np.diag(l * (l + 1) / (2.0 * mass * r**2) - Z / r)
    H = kinetic + potential
    return 0.5 * (H + H.T), r


def transition_rate_matrix(eigs, vecs, r_vec):
    n = len(eigs)
    R_eig = vecs.T @ np.diag(r_vec) @ vecs
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = R_eig[i, i]
        for j in range(n):
            if i != j:
                gap = abs(eigs[i] - eigs[j])
                if gap > 1e-12:
                    T[i, j] = R_eig[i, j] ** 2 / gap
    return 0.5 * (T + T.T)


def compute_return_norm(H, retained, omitted):
    H_PQ = H[np.ix_(retained, omitted)]
    H_QQ = H[np.ix_(omitted, omitted)]
    H_QP = H[np.ix_(omitted, retained)]
    R = H_PQ @ np.linalg.inv(H_QQ) @ H_QP
    return np.linalg.norm(R, ord=2)


def extract_slope(x_vals, y_vals, last_n=3):
    """Log-log slope from last_n points."""
    x = np.log(np.array(x_vals[-last_n:]))
    y = np.log(np.array(y_vals[-last_n:]))
    return np.polyfit(x, y, 1)[0]


N_GRID = 400


# ---------------------------------------------------------------------------
# Proposition 2.1: Yang-Mills chi ~ (g^2)^{-4}
# ---------------------------------------------------------------------------

class TestYangMillsChiExponent:

    @pytest.fixture(scope="class")
    def ym_data(self):
        g2_list = [4.0, 8.0, 16.0, 32.0, 64.0]
        chis = []
        for g2 in g2_list:
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
            r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            chis.append(r["advanced_metrics"]["chi"])
        return g2_list, chis

    def test_chi_exponent_is_minus_four(self, ym_data):
        slope = extract_slope(*ym_data)
        assert abs(slope - (-4.0)) < 0.05, f"YM chi exponent {slope:.4f}, expected -4"

    def test_all_subcritical(self, ym_data):
        for chi in ym_data[1]:
            assert chi < 1

    def test_all_kahan_certified(self, ym_data):
        for g2 in [8.0, 16.0, 32.0, 64.0]:
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
            r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            m = r["advanced_metrics"]
            k = certified_subcritical(m["chi"], m["gamma"], m["lambda"])
            assert k["certified"], f"YM g^2={g2} not Kahan-certified"


# ---------------------------------------------------------------------------
# Proposition 2.2: Yang-Mills omission exponent = 2
# ---------------------------------------------------------------------------

class TestYangMillsOmissionExponent:

    def test_omission_exponent_is_two(self):
        g2_list = [8.0, 16.0, 32.0, 64.0]
        ratios = []
        for g2 in g2_list:
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
            r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            m = r["advanced_metrics"]
            ratios.append(m["lambda"] / m["gamma"])
        slope = extract_slope(g2_list, ratios)
        assert abs(slope - (-2.0)) < 0.02, f"YM omission exponent {-slope:.4f}, expected 2"


# ---------------------------------------------------------------------------
# Proposition 3.1: Hydrogen chi ~ Z^{-6}
# ---------------------------------------------------------------------------

class TestHydrogenChiExponent:

    @pytest.fixture(scope="class")
    def h_data(self):
        Z_list = [15.0, 20.0, 30.0, 50.0]
        chis = []
        for Z in Z_list:
            r_max = 80.0 / Z
            H, r_vec = radial_hamiltonian(N_GRID, r_max, Z=Z, l=0)
            eigs, vecs = np.linalg.eigh(H)
            T = transition_rate_matrix(eigs, vecs, r_vec)
            r = emet.decide_dense_matrix(T, retained=[0], omitted=list(range(1, N_GRID)))
            chis.append(r["advanced_metrics"]["chi"])
        return Z_list, chis

    def test_chi_exponent_is_minus_six(self, h_data):
        slope = extract_slope(*h_data)
        assert abs(slope - (-6.0)) < 0.1, f"H chi exponent {slope:.4f}, expected -6"

    def test_all_kahan_certified(self, h_data):
        for Z in [15.0, 20.0, 30.0, 50.0]:
            r_max = 80.0 / Z
            H, r_vec = radial_hamiltonian(N_GRID, r_max, Z=Z, l=0)
            eigs, vecs = np.linalg.eigh(H)
            T = transition_rate_matrix(eigs, vecs, r_vec)
            r = emet.decide_dense_matrix(T, retained=[0], omitted=list(range(1, N_GRID)))
            m = r["advanced_metrics"]
            k = certified_subcritical(m["chi"], m["gamma"], m["lambda"])
            assert k["certified"], f"H Z={Z} not Kahan-certified"


# ---------------------------------------------------------------------------
# Proposition 3.1: Hydrogen omission exponent = 3
# ---------------------------------------------------------------------------

class TestHydrogenOmissionExponent:

    def test_omission_exponent_is_three(self):
        Z_list = [15.0, 20.0, 30.0, 50.0]
        ratios = []
        for Z in Z_list:
            r_max = 80.0 / Z
            H, r_vec = radial_hamiltonian(N_GRID, r_max, Z=Z, l=0)
            eigs, vecs = np.linalg.eigh(H)
            T = transition_rate_matrix(eigs, vecs, r_vec)
            r = emet.decide_dense_matrix(T, retained=[0], omitted=list(range(1, N_GRID)))
            m = r["advanced_metrics"]
            ratios.append(m["lambda"] / m["gamma"])
        slope = extract_slope(Z_list, ratios)
        assert abs(slope - (-3.0)) < 0.05, f"H omission exponent {-slope:.4f}, expected 3"


# ---------------------------------------------------------------------------
# Theorem 8.1: ||R|| = gamma * chi
# ---------------------------------------------------------------------------

class TestReturnBound:

    def test_ym_return_bounded_by_gamma_chi(self):
        for g2 in [8.0, 16.0, 32.0, 64.0]:
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
            R_norm = compute_return_norm(H, ret, omit)
            r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            m = r["advanced_metrics"]
            gam_chi = m["gamma"] * m["chi"]
            assert R_norm <= gam_chi * 1.001, f"YM g^2={g2}: ||R||={R_norm} > gamma*chi={gam_chi}"
            rel_err = abs(R_norm - gam_chi) / gam_chi
            assert rel_err < 1e-3, f"YM g^2={g2}: bound not tight, err={rel_err}"

    def test_h_return_equals_gamma_chi(self):
        for Z in [20.0, 30.0, 50.0]:
            r_max = 80.0 / Z
            H, r_vec = radial_hamiltonian(N_GRID, r_max, Z=Z, l=0)
            eigs, vecs = np.linalg.eigh(H)
            T = transition_rate_matrix(eigs, vecs, r_vec)
            ret, omit = [0], list(range(1, N_GRID))
            R_norm = compute_return_norm(T, ret, omit)
            r = emet.decide_dense_matrix(T, retained=ret, omitted=omit)
            m = r["advanced_metrics"]
            gam_chi = m["gamma"] * m["chi"]
            rel_err = abs(R_norm - gam_chi) / gam_chi
            assert rel_err < 0.05, f"H Z={Z}: ||R||={R_norm}, gamma*chi={gam_chi}, err={rel_err}"


# ---------------------------------------------------------------------------
# Proposition 8.2: Return exponent = gap exponent + chi exponent
# ---------------------------------------------------------------------------

class TestReturnExponent:

    def test_ym_return_exponent_is_minus_three(self):
        g2_list = [8.0, 16.0, 32.0, 64.0]
        R_norms = []
        for g2 in g2_list:
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
            R_norms.append(compute_return_norm(H, ret, omit))
        slope = extract_slope(g2_list, R_norms)
        assert abs(slope - (-3.0)) < 0.02, f"YM return exponent {slope:.4f}, expected -3"

    def test_h_return_exponent_is_minus_seven(self):
        Z_list = [15.0, 20.0, 30.0, 50.0]
        R_norms = []
        for Z in Z_list:
            r_max = 80.0 / Z
            H, r_vec = radial_hamiltonian(N_GRID, r_max, Z=Z, l=0)
            eigs, vecs = np.linalg.eigh(H)
            T = transition_rate_matrix(eigs, vecs, r_vec)
            R_norms.append(compute_return_norm(T, [0], list(range(1, N_GRID))))
        slope = extract_slope(Z_list, R_norms)
        assert abs(slope - (-7.0)) < 0.1, f"H return exponent {slope:.4f}, expected -7"


# ---------------------------------------------------------------------------
# Theorem 5.1: Factorization chi(g^2, j_cut) = f(g^2) * h(j_cut)
# ---------------------------------------------------------------------------

class TestFactorization:

    def test_scale_profile_is_universal(self):
        """Ratio chi(g2, j_cut) / chi(g2, j_cut=1) converges across g2."""
        j_cuts = [1.0, 1.5, 2.0, 2.5, 3.0]
        profiles = {}
        for g2 in [8.0, 16.0, 32.0]:
            chis = []
            for j_cut in j_cuts:
                H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=j_cut)
                r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
                chis.append(r["advanced_metrics"]["chi"])
            # normalize by j_cut=1
            profiles[g2] = [c / chis[0] for c in chis]

        # profiles at g2=16 and g2=32 should agree to 3 digits
        for i in range(len(j_cuts)):
            diff = abs(profiles[16.0][i] - profiles[32.0][i])
            assert diff < 1e-3, f"j_cut={j_cuts[i]}: profile differs by {diff}"


# ---------------------------------------------------------------------------
# Definition 4.1: Condensation regimes
# ---------------------------------------------------------------------------

class TestCondensation:

    def test_hydrogen_ground_state_subcritical(self):
        H, r_vec = radial_hamiltonian(N_GRID, 80.0, Z=1.0, l=0)
        eigs, vecs = np.linalg.eigh(H)
        T = transition_rate_matrix(eigs, vecs, r_vec)
        r = emet.decide_dense_matrix(T, retained=[0], omitted=list(range(1, N_GRID)))
        assert r["advanced_metrics"]["chi"] < 1

    def test_hydrogen_n2_supercritical(self):
        H, r_vec = radial_hamiltonian(N_GRID, 80.0, Z=1.0, l=0)
        eigs, vecs = np.linalg.eigh(H)
        T = transition_rate_matrix(eigs, vecs, r_vec)
        r = emet.decide_dense_matrix(T, retained=[0, 1], omitted=list(range(2, N_GRID)))
        assert r["advanced_metrics"]["chi"] > 1

    def test_ym_all_cuts_subcritical(self):
        """Yang-Mills is subcritical at every j_cut (confinement)."""
        for j_cut in [0.5, 1.0, 1.5, 2.0, 3.0]:
            H, ret, omit, _ = build_plaquette_blocks(16.0, j_max=10, j_cut=j_cut)
            r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            assert r["advanced_metrics"]["chi"] < 1, f"YM j_cut={j_cut} not subcritical"


# ---------------------------------------------------------------------------
# Theorem (cave §10): n > 2d implies chi > 1 for rank-d Gram matrix
# ---------------------------------------------------------------------------

class TestCaveBound:

    @pytest.mark.parametrize("n,d", [(256, 64), (512, 64), (128, 32)])
    def test_supercritical_past_2d(self, n, d):
        """Random rank-d Gram matrix with n > 2d: chi > 1 under half-split."""
        assert n > 2 * d
        K = np.random.randn(n, d)
        eta = 0.1
        H = K @ K.T + eta * np.eye(n)
        mid = n // 2
        ret = list(range(mid))
        omit = list(range(mid, n))
        r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        chi = r["advanced_metrics"]["chi"]
        assert chi > 1, f"n={n}, d={d}: chi={chi}, expected > 1"

    @pytest.mark.parametrize("n,d", [(64, 64), (100, 64)])
    def test_subcritical_below_2d(self, n, d):
        """Random rank-d Gram matrix with n <= 2d: chi can be < 1."""
        assert n <= 2 * d
        K = np.random.randn(n, d)
        eta = 0.1
        H = K @ K.T + eta * np.eye(n)
        mid = n // 2
        ret = list(range(mid))
        omit = list(range(mid, n))
        r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        # Below 2d, chi is not guaranteed > 1. Just verify it's finite.
        chi = r["advanced_metrics"]["chi"]
        assert chi is not None and np.isfinite(chi)
