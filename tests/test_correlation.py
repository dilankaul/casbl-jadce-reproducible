import numpy as np
from casbl_jadce.correlation import build_C, build_Omega, interaction_phi


def test_omega_and_fast_phi_match_explicit_expression():
    loc = np.array([[0., 0.], [1., 0.], [30., 0.]])
    C = build_C(loc, rho=7, U=20)
    alpha, beta, M = 2.5, 0.1, 4
    gamma = np.array([0.2, 0.4, 0.1])
    Omega = build_Omega(C, alpha, beta)
    assert np.allclose(interaction_phi(C, gamma, alpha, beta, M), (Omega @ gamma) / M)
    assert C[0, 2] == 0.0
    assert np.allclose(np.diag(C), 1.0)
