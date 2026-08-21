import os
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


def coherent_vec(N, alpha):
    n = np.arange(N)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, N)))])
    logc = -0.5*abs(alpha)**2 + n*np.log(alpha + 0j) - 0.5*logfact
    v = np.exp(logc)
    return v / np.linalg.norm(v)


# ══════════════════════════════════════════════════════════════════
#  Estado exacto vs gato de coherentes ideal
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

# --- estado exacto ---
H_e, H_o, nph, n_ph = build_LF_hamiltonians_sweet(
    M, cv_val, cv_val, omega0, omega0, g1, g2)
ev_e, evec_e = eigh(H_e)
vec = evec_e[:, 0]

# ── diagnóstico de truncación: peso cerca del borde de Fock ──
p1 = vec[:n_ph].reshape(nph, nph)
p2 = vec[n_ph:].reshape(nph, nph)
for tag, p in [("phi1", p1), ("phi2", p2)]:
    print(f"{tag}: peso total = {np.sum(np.abs(p)**2):.6e}")
    print(f"      ultimas 5 filas/cols = "
          f"{np.sum(np.abs(p[-5:,:])**2) + np.sum(np.abs(p[:,-5:])**2):.6e}")
    print(f"      poblacion modo 1 en n=nph-1: {np.sum(np.abs(p[-1,:])**2):.6e}")
    print(f"      poblacion modo 2 en n=nph-1: {np.sum(np.abs(p[:,-1])**2):.6e}")

phi1 = pad_state(vec[:n_ph],  nph, nph_f)

phi2 = pad_state(vec[n_ph:],  nph, nph_f)
P1 = phi1.reshape(nph_f, nph_f); P2 = phi2.reshape(nph_f, nph_f)

D1p = disp_matrix(nph_f-1, +b1); D1m = disp_matrix(nph_f-1, -b1)
D2p = disp_matrix(nph_f-1, +b2); D2m = disp_matrix(nph_f-1, -b2)

cat_ex = (D1p @ P2 @ D2p.T) + (-sign)*(D1m @ P1 @ D2m.T)
cat_ex /= np.linalg.norm(cat_ex)

# --- gato de coherentes ideal ---
u_p = coherent_vec(nph_f, +b1); v_p = coherent_vec(nph_f, +b2)
u_m = coherent_vec(nph_f, -b1); v_m = coherent_vec(nph_f, -b2)
cat_id = np.outer(u_p, v_p) + (-sign)*np.outer(u_m, v_m)
cat_id /= np.linalg.norm(cat_id)

# --- fidelidad ---
F = abs(np.vdot(cat_id.reshape(-1), cat_ex.reshape(-1)))**2
print(f"beta1 = {b1:.4f}   beta2 = {b2:.4f}")
print(f"\nF = |<Cat_coh|Cat_exact>|^2 = {F:.8f}")
print(f"1 - F                       = {1-F:.6e}")

# --- espectros ---
for tag, cm in [("exacto", cat_ex), ("coherente ideal", cat_id)]:
    rho1 = cm @ cm.conj().T
    ev = np.sort(np.linalg.eigvalsh(rho1).real)[::-1]
    s = np.sqrt(ev[ev > 1e-12]); lam = s**2
    print(f"\n--- {tag} ---")
    for i, v in enumerate(ev[:4]):
        print(f"  lam_{i} = {v:.6e}")
    print(f"  lam_2 + lam_3 = {ev[2]+ev[3]:.6e}")
    print(f"  E_N = {2*np.log2(np.sum(s)):.4f}   S = {-np.sum(lam*np.log2(lam)):.4f}")