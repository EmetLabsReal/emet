"""Two-cavity glymphatic coupling: Schumann EM cavity ↔ neural oscillator.

The Earth-ionosphere cavity has Schumann resonances at 7.83, 14.3,
20.8 Hz etc.  The brain has neural oscillation bands (theta 4-8 Hz,
alpha 8-13 Hz, beta 13-30 Hz) driven by thalamocortical loops and
hemodynamic standing waves (Bentov's aorta-brain mechanism at ~7 Hz).

The coupling is NOT between two EM cavities — the cranial shell is
too small for EM resonance at Schumann frequencies.  It's between
the Schumann EM field and the neural mechanical/electrical oscillator,
mediated by the atmospheric electric field penetrating the skull.

Block structure:
  H = [[H_schumann,         α·V_coupling],
       [α·V_coupling^T,     H_neural    ]]

H_schumann: thin-shell Laplacian (angular + radial), eigenvalues in Hz².
H_neural: diagonal matrix with entries f_i² for each neural band,
where f_i are measured EEG frequencies.
Partition retained: Schumann fundamentals. Omitted: everything else.

The lapse α ∈ [0,1] represents glymphatic channel openness:
  α ≈ 0.1: normal waking (skull + dura attenuation)
  α ≈ 0.3: focused meditation (Bentov hemodynamic resonance)
  α ≈ 0.6: NREM sleep (glymphatic channels dilate ~60%)
  α ≈ 0.0: deep focus / flow state (fully decoupled)

The sweep finds α_c where χ crosses 1: the entrainment threshold.
"""

from __future__ import annotations

import numpy as np


# Earth-ionosphere cavity
R_EARTH = 6.371e6       # m
C_LIGHT = 2.998e8       # m/s
DELTA_IONOSPHERE = 80e3  # m

# Neural oscillation frequencies (Hz) — measured EEG bands
NEURAL_FREQS_HZ = [
    4.0,    # theta low
    6.0,    # theta mid
    7.83,   # theta high — matches Schumann n=1
    8.5,    # alpha low
    10.0,   # alpha mid
    12.0,   # alpha high
    14.3,   # beta low — matches Schumann n=2
    20.0,   # beta mid
    25.0,   # beta high
    40.0,   # gamma
]

NEURAL_LABELS = [
    "theta_4Hz", "theta_6Hz", "theta_7.83Hz",
    "alpha_8.5Hz", "alpha_10Hz", "alpha_12Hz",
    "beta_14.3Hz", "beta_20Hz", "beta_25Hz",
    "gamma_40Hz",
]

# Schumann frequencies for n=1..6 (Hz)
SCHUMANN_FREQS_HZ = [
    (C_LIGHT / (2.0 * np.pi * R_EARTH)) * np.sqrt(n * (n + 1))
    for n in range(1, 7)
]
# ≈ [7.83, 14.1, 20.3, 26.4, 32.4, 38.3]


def _schumann_block(
    n_angular: int = 6,
    n_radial: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Schumann cavity operator in Hz² units.

    Same physics as schumann.py but with frequency-squared scaling
    so eigenvalues are directly in Hz².
    """
    eps = DELTA_IONOSPHERE / R_EARTH
    freq_scale = (C_LIGHT / (2.0 * np.pi * R_EARTH)) ** 2

    N = n_angular * n_radial
    H = np.zeros((N, N))
    labels = []

    for n in range(n_angular):
        for k in range(1, n_radial + 1):
            labels.append(f"sch(n={n},k={k})")

    for n in range(n_angular):
        ang = n * (n + 1)
        for k in range(1, n_radial + 1):
            i = n * n_radial + (k - 1)
            H[i, i] = freq_scale * (ang + (k * np.pi / eps) ** 2)

            for kp in range(k + 1, n_radial + 1):
                j = n * n_radial + (kp - 1)
                if (k - kp) % 2 == 0:
                    continue
                km = k - kp
                ks = k + kp
                overlap = -8.0 * k * kp / (np.pi**2 * km**2 * ks**2)
                if abs(overlap) > 1e-15:
                    coupling = freq_scale * ang * 2.0 * eps * overlap
                    H[i, j] = coupling
                    H[j, i] = coupling

    return 0.5 * (H + H.T), labels


def _neural_block(
    freqs_hz: list[float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Neural oscillator Hamiltonian: diagonal, entries = f_i² (Hz²).

    Each neural band is a harmonic mode at its measured frequency.
    Off-diagonal: nearest-neighbor coupling between adjacent bands
    (thalamocortical cross-frequency coupling, ~10% of geometric mean).
    """
    if freqs_hz is None:
        freqs_hz = list(NEURAL_FREQS_HZ)

    n = len(freqs_hz)
    H = np.zeros((n, n))

    # Diagonal: f² in Hz²
    for i, f in enumerate(freqs_hz):
        H[i, i] = f ** 2

    # Off-diagonal: cross-frequency coupling between adjacent bands
    # Empirical: ~10% of geometric mean of adjacent mode frequencies²
    for i in range(n - 1):
        coupling = 0.1 * np.sqrt(freqs_hz[i] * freqs_hz[i + 1])
        H[i, i + 1] = coupling
        H[i + 1, i] = coupling

    return 0.5 * (H + H.T), list(NEURAL_LABELS[:n])


def _coupling_block(
    n_angular: int,
    n_radial: int,
    n_neural: int,
    coupling_strength: float,
) -> np.ndarray:
    """Coupling between Schumann fundamentals and neural modes.

    Each Schumann fundamental (n=1..n_angular-1, k=1) couples to
    the neural mode nearest in frequency.  Coupling strength is
    proportional to frequency overlap (Lorentzian with width 2 Hz).
    """
    N_sch = n_angular * n_radial
    V = np.zeros((N_sch, n_neural))

    neural_f = NEURAL_FREQS_HZ[:n_neural]
    width = 2.0  # Hz — resonance bandwidth

    for n in range(1, n_angular):
        f_sch = SCHUMANN_FREQS_HZ[n - 1] if n - 1 < len(SCHUMANN_FREQS_HZ) else 0.0
        i_sch = n * n_radial + 0  # fundamental radial mode (k=1)

        for j, f_neural in enumerate(neural_f):
            # Lorentzian overlap: peaks when frequencies match
            detuning = f_sch - f_neural
            overlap = width**2 / (detuning**2 + width**2)

            if overlap > 0.01:
                # Coupling in Hz² units: geometric mean of the two
                # mode frequencies, weighted by overlap
                V[i_sch, j] = coupling_strength * np.sqrt(f_sch * f_neural) * overlap

    return V


def build_glymphatic_blocks(
    alpha: float = 0.0,
    n_angular: int = 6,
    n_radial: int = 3,
    n_neural: int | None = None,
    coupling_strength: float = 1.0,
) -> tuple[np.ndarray, list[int], list[int], list[str]]:
    """Build the Schumann-neural two-system Hermitian operator.

    Parameters
    ----------
    alpha : float
        Lapse (glymphatic openness) in [0, 1].
    n_angular : int
        Schumann angular mode count.
    n_radial : int
        Schumann radial mode count.
    n_neural : int or None
        Number of neural frequency bands (default: all 10).
    coupling_strength : float
        Base coupling amplitude.

    Returns
    -------
    H : np.ndarray
    retained : list[int]
        Schumann fundamental modes (k=1 for each angular n).
    omitted : list[int]
        Everything else (Schumann overtones + all neural modes).
    labels : list[str]
    """
    if n_neural is None:
        n_neural = len(NEURAL_FREQS_HZ)

    H_sch, lab_sch = _schumann_block(n_angular, n_radial)
    H_neural, lab_neural = _neural_block(NEURAL_FREQS_HZ[:n_neural])
    V = _coupling_block(n_angular, n_radial, n_neural, coupling_strength)

    N_sch = H_sch.shape[0]
    N_neural = H_neural.shape[0]
    N = N_sch + N_neural

    H = np.zeros((N, N))
    H[:N_sch, :N_sch] = H_sch
    H[N_sch:, N_sch:] = H_neural
    H[:N_sch, N_sch:] = alpha * V
    H[N_sch:, :N_sch] = alpha * V.T
    H = 0.5 * (H + H.T)

    labels = [f"earth:{l}" for l in lab_sch] + [f"neural:{l}" for l in lab_neural]

    # Retained: Schumann fundamentals (k=1) for each angular mode
    retained = [n * n_radial for n in range(n_angular)]
    omitted = [i for i in range(N) if i not in retained]

    return H, retained, omitted, labels
