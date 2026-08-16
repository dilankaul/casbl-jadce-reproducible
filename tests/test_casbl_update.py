import numpy as np
from casbl_jadce.algorithms.casbl import stable_gamma_update, casbl
from casbl_jadce.algorithms.sbl import sbl
from casbl_jadce.system_model import simulate_communication


def test_stable_root_matches_direct_root():
    eta = np.array([0.01, 0.2, 2.0, 20.0])
    phi = np.array([1e-8, 0.05, 0.5, 3.0])
    stable = stable_gamma_update(eta, phi)
    direct = (np.sqrt(1.0 + 4.0 * phi * eta) - 1.0) / (2.0 * phi)
    assert np.allclose(stable, direct, rtol=1e-7, atol=1e-10)


def test_nonpositive_phi_falls_back_to_eta():
    eta = np.array([0.2, 0.3, 0.4])
    phi = np.array([0.0, -1.0, 0.2])
    result = stable_gamma_update(eta, phi)
    assert result[0] == eta[0]
    assert result[1] == eta[1]
    assert 0 <= result[2] <= eta[2]


def test_alpha_zero_matches_sbl():
    rng = np.random.default_rng(4)
    N, M, L = 20, 3, 10
    a = np.zeros(N, dtype=bool); a[[1, 7, 13]] = True
    r = simulate_communication(rng, a, M=M, L=L, snr_db=15)
    C = np.eye(N)
    ca = casbl(r.Theta, r.Y, r.noise_var, C, alpha=0.0, beta=0.1, max_iter=25, tol=1e-8)
    sb = sbl(r.Theta, r.Y, r.noise_var, max_iter=25, tol=1e-8)
    assert np.allclose(ca.gamma, sb.gamma, atol=1e-10, rtol=1e-9)
    assert np.allclose(ca.mu, sb.mu, atol=1e-10, rtol=1e-9)
