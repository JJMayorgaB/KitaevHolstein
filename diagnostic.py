import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.special import factorial, eval_genlaguerre
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')


def disp_matrix(M_ph, alpha):
    N      = M_ph + 1
    alpha2 = alpha * alpha
    gauss  = np.exp(-0.5 * alpha2)
    facts  = np.array([float(factorial(k, exact=True)) for k in range(N)])
    D = np.zeros((N, N))
    for mp in range(N):
        for n in range(N):
            if mp >= n:
                k    = mp - n
                pref = gauss * (alpha**k) * np.sqrt(facts[n] / facts[mp])
                D[mp, n] = pref * eval_genlaguerre(n, k, alpha2)
            else:
                k    = n - mp
                pref = gauss * ((-alpha)**k) * np.sqrt(facts[mp] / facts[n])
                D[mp, n] = pref * eval_genlaguerre(mp, k, alpha2)
    return D


def MofLam_sweet_exact(g1, g2, omega0=1.0, n_sigma=4, floor=15, M_cap=55):
    lam_max = max(abs(g1), abs(g2)) / omega0
    M = int(np.ceil(lam_max**2 + n_sigma*lam_max))
    M = max(M, floor)
    return min(M, M_cap)


def build_LF_hamiltonians_sweet(M, t, Delta, omega1, omega2, g1, g2):
    nph  = M + 1
    n_ph = nph * nph
    lam1 = g1 / omega1
    lam2 = g2 / omega2

    D1p = disp_matrix(M,  lam1)
    D1m = disp_matrix(M, -lam1)
    D2p = disp_matrix(M,  lam2)
    D2m = disp_matrix(M, -lam2)

    ph_E = np.array([m1*omega1 + m2*omega2
                     for m1 in range(nph) for m2 in range(nph)])

    T_hop  = t     * np.kron(D1p, D2m)
    T_pair = Delta * np.kron(D1m, D2m)

    H_e = np.zeros((2*n_ph, 2*n_ph))
    H_e[:n_ph, :n_ph] = np.diag(ph_E)
    H_e[n_ph:, n_ph:] = np.diag(ph_E)
    H_e[:n_ph, n_ph:] = T_pair
    H_e[n_ph:, :n_ph] = T_pair.T

    H_o = np.zeros((2*n_ph, 2*n_ph))
    H_o[:n_ph, :n_ph] = np.diag(ph_E)
    H_o[n_ph:, n_ph:] = np.diag(ph_E)
    H_o[:n_ph, n_ph:] = T_hop
    H_o[n_ph:, :n_ph] = T_hop.T

    return H_e, H_o, nph, n_ph


def diagonalize_LF_sweet(M, t, Delta, omega1, omega2, g1, g2):
    H_e, H_o, nph, n_ph = build_LF_hamiltonians_sweet(
        M, t, Delta, omega1, omega2, g1, g2)
    ev_e, evec_e = eigh(H_e)
    ev_o, evec_o = eigh(H_o)
    return ev_e, evec_e, ev_o, evec_o, nph, n_ph


def exact_cat_bimodal(g1, g2, evec_ex, nph, n_ph, omega0=1.0, sign=+1, sector='even'):
    lam1 = g1 / omega0
    lam2 = g2 / omega0

    phi1 = evec_ex[:n_ph]
    phi2 = evec_ex[n_ph:]

    D1p = disp_matrix(nph-1, +0.5*lam1); D1m = disp_matrix(nph-1, -0.5*lam1)
    D2p = disp_matrix(nph-1, +0.5*lam2); D2m = disp_matrix(nph-1, -0.5*lam2)

    if sector == 'even':
        Dp = np.kron(D1p, D2p)
        Dm = np.kron(D1m, D2m)
    else:
        Dp = np.kron(D1p, D2m)
        Dm = np.kron(D1m, D2p)

    cat = Dp @ phi2 + (-sign) * (Dm @ phi1)
    cat /= np.linalg.norm(cat)
    return cat


def pad_state(psi_ph, nph_old, nph_new):
    pm = psi_ph.reshape(nph_old, nph_old)
    pm_new = np.zeros((nph_new, nph_new), dtype=complex)
    pm_new[:nph_old, :nph_old] = pm
    return pm_new.reshape(-1)


# ══════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO — espectro de Schmidt de rho1 en un solo θ
# ══════════════════════════════════════════════════════════════════
omega0  = 1.0
cv_val  = 1.0
Lam_fix = 3*np.sqrt(2)
th      = 1.047
nph_wig = 85

g1 = Lam_fix*np.cos(th)
g2 = Lam_fix*np.sin(th)
M  = min(MofLam_sweet_exact(g1, g2, omega0=omega0, M_cap=45) + 25, 55)

ev_e, evec_e, ev_o, evec_o, nph, n_ph = diagonalize_LF_sweet(
    M, cv_val, cv_val, omega0, omega0, g1, g2)
psi_ph = exact_cat_bimodal(g1, g2, evec_e[:, 0], nph, n_ph, omega0,
                           sign=+1, sector='even')

psi_ph = pad_state(psi_ph, nph, nph_wig)
pm = psi_ph.reshape(nph_wig, nph_wig)
rho1 = pm @ pm.conj().T

ev = np.sort(np.linalg.eigvalsh(rho1).real)[::-1]
print("M =", M, " nph =", nph)
print("traza rho1 =", ev.sum())
print("10 mayores autovalores de rho1:")
for i, v in enumerate(ev[:10]):
    print(f"  lam_{i} = {v:.6e}")
print("componentes > 1e-12:", np.sum(ev > 1e-12))
print("componentes > 1e-8: ", np.sum(ev > 1e-8))
print("componentes > 1e-4: ", np.sum(ev > 1e-4))