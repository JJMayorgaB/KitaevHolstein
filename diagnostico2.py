import os
import numpy as np
from scipy.special import factorial, eval_genlaguerre
from scipy.linalg import eigh, expm
import warnings
warnings.filterwarnings('ignore')


def disp_matrix_lag(M_ph, alpha):
    """Versión Laguerre (la actual)."""
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


def disp_matrix_expm(M_ph, alpha):
    """D(α) = exp[α(a† − a)] por exponenciación del generador."""
    N = M_ph + 1
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    return expm(alpha*(a.T - a))


# ── selector global: cambia aquí para probar una u otra ──
disp_matrix = disp_matrix_expm     # o disp_matrix_lag


def build_LF_hamiltonians_sweet(M, t, Delta, omega1, omega2, g1, g2):
    nph  = M + 1
    n_ph = nph * nph
    lam1 = g1 / omega1
    lam2 = g2 / omega2

    D1p = disp_matrix(M,  lam1); D1m = disp_matrix(M, -lam1)
    D2p = disp_matrix(M,  lam2); D2m = disp_matrix(M, -lam2)

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


def pad_state(psi_ph, nph_old, nph_new):
    pm = psi_ph.reshape(nph_old, nph_old)
    pm_new = np.zeros((nph_new, nph_new), dtype=complex)
    pm_new[:nph_old, :nph_old] = pm
    return pm_new.reshape(-1)


# ══════════════════════════════════════════════════════════════════
omega0  = 1.0
cv_val  = 1.0
Lam_fix = 3*np.sqrt(2)
th      = 1.047
M       = 75
nph_f   = 100
sign    = +1

g1 = Lam_fix*np.cos(th); g2 = Lam_fix*np.sin(th)
lam1 = g1/omega0;        lam2 = g2/omega0
b1   = 0.5*lam1;         b2   = 0.5*lam2

# ══════════════════════════════════════════════════════════════════
#  TEST DE UNITARIEDAD — ambas versiones, varias dimensiones
# ══════════════════════════════════════════════════════════════════
print("=== unitariedad ||D^T D - I|| ===")
print(f"{'alpha':>8}  {'N':>5}  {'Laguerre':>14}  {'expm':>14}")
for a, tag in [(b1,'b1'), (b2,'b2'), (lam1,'lam1'), (lam2,'lam2')]:
    for N in [M+1, nph_f]:
        I = np.eye(N)
        DL = disp_matrix_lag(N-1, a)
        DE = disp_matrix_expm(N-1, a)
        eL = np.linalg.norm(DL.T @ DL - I)
        eE = np.linalg.norm(DE.T @ DE - I)
        print(f"{a:8.4f}  {N:5d}  {eL:14.6e}  {eE:14.6e}   ({tag})")

# ══════════════════════════════════════════════════════════════════
#  ESPECTRO DE SCHMIDT — con cada versión
# ══════════════════════════════════════════════════════════════════
for name, dm in [("Laguerre", disp_matrix_lag), ("expm", disp_matrix_expm)]:
    globals()['disp_matrix'] = dm

    H_e, H_o, nph, n_ph = build_LF_hamiltonians_sweet(
        M, cv_val, cv_val, omega0, omega0, g1, g2)
    ev_e, evec_e = eigh(H_e)
    vec = evec_e[:, 0]

    p1 = vec[:n_ph].reshape(nph, nph)
    peso_fuera = 0.5 - abs(p1[0,0])**2

    phi1 = pad_state(vec[:n_ph], nph, nph_f)
    phi2 = pad_state(vec[n_ph:], nph, nph_f)
    P1 = phi1.reshape(nph_f, nph_f); P2 = phi2.reshape(nph_f, nph_f)

    D1p = dm(nph_f-1, +b1); D1m = dm(nph_f-1, -b1)
    D2p = dm(nph_f-1, +b2); D2m = dm(nph_f-1, -b2)

    catm = (D1p @ P2 @ D2p.T) + (-sign)*(D1m @ P1 @ D2m.T)
    catm /= np.linalg.norm(catm)

    rho1 = catm @ catm.conj().T
    ev = np.sort(np.linalg.eigvalsh(rho1).real)[::-1]
    s = np.sqrt(ev[ev > 1e-12]); lam = s**2

    print(f"\n=== {name} ===")
    print(f"  peso fuera del vacio (phi1) = {peso_fuera:.6e}")
    print(f"  traza rho1 = {ev.sum():.12f}")
    for i, v in enumerate(ev[:6]):
        print(f"  lam_{i} = {v:.6e}")
    print(f"  E_N = {2*np.log2(np.sum(s)):.4f}   S = {-np.sum(lam*np.log2(lam)):.4f}")
    print(f"  rango efectivo (>1e-4): {np.sum(ev > 1e-4)}")