import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.special import factorial, eval_genlaguerre
from scipy.linalg import eigh
from joblib import Parallel, delayed
import qutip as qt
import warnings
warnings.filterwarnings('ignore')


def disp_matrix(M_ph, alpha):
    """Elementos de matriz <m'|D(alpha)|m> en base de Fock, alpha real."""
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


def build_LF_hamiltonian_sweet(M, t, Delta, omega1, omega2, g1, g2, sector='even'):
    nph  = M + 1
    n_ph = nph * nph
    lam1 = g1 / omega1; lam2 = g2 / omega2

    ph_E = np.array([m1*omega1 + m2*omega2
                     for m1 in range(nph) for m2 in range(nph)])

    if sector == 'even':
        T = Delta * np.kron(disp_matrix(M, -lam1), disp_matrix(M, -lam2))
    else:
        T = t     * np.kron(disp_matrix(M,  lam1), disp_matrix(M, -lam2))

    H = np.zeros((2*n_ph, 2*n_ph))
    H[:n_ph, :n_ph] = np.diag(ph_E)
    H[n_ph:, n_ph:] = np.diag(ph_E)
    H[:n_ph, n_ph:] = T
    H[n_ph:, :n_ph] = T.T
    return H, nph, n_ph


def diagonalize_LF_sweet(M, t, Delta, omega1, omega2, g1, g2, sector='even'):
    """Construye y diagonaliza SOLO el sector pedido."""
    H, nph, n_ph = build_LF_hamiltonian_sweet(M, t, Delta, omega1, omega2, g1, g2, sector=sector)
    ev, evec = eigh(H)
    return ev, evec, nph, n_ph


def pad_state(psi_ph, nph_old, nph_new):
    pm = psi_ph.reshape(nph_old, nph_old)
    pm_new = np.zeros((nph_new, nph_new), dtype=complex)
    pm_new[:nph_old, :nph_old] = pm
    return pm_new.reshape(-1)


def exact_cat_bimodal(g1, g2, evec_ex, nph, n_ph, omega0=1.0, sign=+1,
                      sector='even', nph_out=None):
    """Gato bimodal exacto en base de sitio. Padding ANTES de desplazar."""
    lam1 = g1 / omega0
    lam2 = g2 / omega0

    phi1 = evec_ex[:n_ph]
    phi2 = evec_ex[n_ph:]

    if nph_out is not None and nph_out > nph:
        phi1 = pad_state(phi1, nph, nph_out)
        phi2 = pad_state(phi2, nph, nph_out)
        nph  = nph_out

    D1p = disp_matrix(nph-1, +0.5*lam1); D1m = disp_matrix(nph-1, -0.5*lam1)
    D2p = disp_matrix(nph-1, +0.5*lam2); D2m = disp_matrix(nph-1, -0.5*lam2)

    P1 = phi1.reshape(nph, nph)
    P2 = phi2.reshape(nph, nph)

    if sector == 'even':
        catm = (D1p @ P2 @ D2p.T) + (-sign)*(D1m @ P1 @ D2m.T)
    else:
        catm = (D1p @ P2 @ D2m.T) + (-sign)*(D1m @ P1 @ D2p.T)

    cat = catm.reshape(-1)
    cat /= np.linalg.norm(cat)
    return cat, nph


def entanglement_measures_pure(rho1, tol=1e-12):
    """E_N, S y rango efectivo del espectro de Schmidt de ρ1 (estado PURO)."""
    ev = np.linalg.eigvalsh(rho1).real
    ev = ev[ev > tol]
    E_N = 2.0 * np.log2(np.sum(np.sqrt(ev)))
    S   = -np.sum(ev * np.log2(ev))
    rank_eff = int(np.sum(ev > 1e-4))
    return E_N, S, rank_eff


def wigner_dm_negativity(rho, xvec, pvec):
    """W(α) de una matriz densidad de un modo + su negatividad.

    Usa qt.wigner (algoritmo de Clenshaw, compilado): no construye D(α),
    evitando la pérdida de precisión de la fórmula de Laguerre con
    factoriales explícitos, que producía ∫W hasta 1.14 en Λ grande.
    """
    W = qt.wigner(qt.Qobj(rho), xvec, pvec)

    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check


def marginal_negativities_exact(psi_ph, nph, xvec, pvec):
    pm = psi_ph.reshape(nph, nph)
    rho1 = pm @ pm.conj().T
    rho2 = pm.T @ pm.conj()

    E_N, S, rank_eff = entanglement_measures_pure(rho1)

    d1, n1 = wigner_dm_negativity(rho1, xvec, pvec)
    d2, n2 = wigner_dm_negativity(rho2, xvec, pvec)
    return d1, d2, n1, n2, E_N, S, rank_eff


def wigner_grid(Lam, n_x=301, n_p=251, x_pad=5.0, p_half=4.0):
    """Malla adaptativa (misma que el variacional)."""
    x_max = 0.5*Lam + x_pad
    return (np.linspace(-x_max, x_max, n_x),
            np.linspace(-p_half, p_half, n_p))

def M_of_Lam(Lam, floor=75):
    """M = 75 a partir de Λ = 2√2; por debajo escala con el tamaño del estado."""
    if Lam >= 2*np.sqrt(2):
        return floor
    M = int(np.ceil(Lam**2 + 4*Lam)) + 10
    return int(min(M, floor))


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN — mapa (Λ, θ) de δ1, δ2, E_N, S (EXACTO en base de sitio)
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    omega0   = 1.0
    cv_val   = 1.0
    sector   = 'even'
    sign     = +1
    nph_wig  = 85
    N_JOBS   = 101

    nLam = 32
    nth  = 32
    Lam_arr   = np.linspace(0.05, 6.0, nLam)
    theta_arr = np.linspace(0.0, 2.0*np.pi, nth)

    OUTDIR = (f'cat_state_information_resources_exact_omega0{omega0:.1f}' f'_nL_{nLam}_nth_{nth}')
    os.makedirs(OUTDIR, exist_ok=True)

    def _one_point(iL, ith):
        Lam = Lam_arr[iL]; th = theta_arr[ith]
        g1 = Lam*np.cos(th); g2 = Lam*np.sin(th)

        M = M_of_Lam(Lam)
        ev, evec, nph, n_ph = diagonalize_LF_sweet( M, cv_val, cv_val, omega0, omega0, g1, g2, sector=sector)
        gap = ev[1] - ev[0]

        psi_ph, nph_f = exact_cat_bimodal(g1, g2, evec[:, 0], nph, n_ph, omega0, sign=sign, sector=sector, nph_out=nph_wig)

        xvec, pvec = wigner_grid(Lam)
        d1, d2, n1, n2, E_N, S, rk = marginal_negativities_exact(psi_ph, nph_f, xvec, pvec)

        return iL, ith, d1, d2, E_N, S, n1, n2, rk, gap

    tasks = [(iL, ith) for iL in range(nLam) for ith in range(nth)]
    print(f'Puntos: {len(tasks)}   ({nLam} Λ × {nth} θ)', flush=True)

    _res = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(_one_point)(iL, ith) for iL, ith in tasks)

    d1_map  = np.full((nLam, nth), np.nan)
    d2_map  = np.full((nLam, nth), np.nan)
    EN_map  = np.full((nLam, nth), np.nan)
    S_map   = np.full((nLam, nth), np.nan)
    n1_map  = np.full((nLam, nth), np.nan)
    n2_map  = np.full((nLam, nth), np.nan)
    rk_map  = np.full((nLam, nth), np.nan)
    gap_map = np.full((nLam, nth), np.nan)

    for iL, ith, d1, d2, E_N, S, n1, n2, rk, gap in _res:
        d1_map[iL, ith] = d1; d2_map[iL, ith] = d2
        EN_map[iL, ith] = E_N; S_map[iL, ith]  = S
        n1_map[iL, ith] = n1; n2_map[iL, ith]  = n2
        rk_map[iL, ith] = rk; gap_map[iL, ith] = gap

    hdr = (f'mapa exacto (Lambda, theta) | filas = Lambda ({nLam}), ' f'columnas = theta ({nth}) | sector={sector}, sign={sign:+d}, ' f'nph_wig={nph_wig}')

    np.savetxt(f'{OUTDIR}/coord_Lam.txt',   Lam_arr,   fmt='%.8e', header='Lambda')
    np.savetxt(f'{OUTDIR}/coord_theta.txt', theta_arr, fmt='%.8e', header='theta')
    np.savetxt(f'{OUTDIR}/delta1.txt', d1_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/delta2.txt', d2_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/EN.txt',     EN_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/S.txt',      S_map,  fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/norm1.txt',  n1_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/norm2.txt',  n2_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/rank_eff.txt', rk_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/gap.txt',      gap_map, fmt='%.8e', header=hdr)

    print(f'Guardado en {OUTDIR}/')
    print(f'  ∫W1 ∈ [{np.nanmin(n1_map):.6f}, {np.nanmax(n1_map):.6f}]')
    print(f'  ∫W2 ∈ [{np.nanmin(n2_map):.6f}, {np.nanmax(n2_map):.6f}]')
    print(f'  puntos con |∫W-1| > 1e-3 : {int(np.sum(np.abs(n1_map-1) > 1e-3))} de {n1_map.size}')
    print(f'  E_N ∈ [{np.nanmin(EN_map):.4f}, {np.nanmax(EN_map):.4f}]   '
          f'S ∈ [{np.nanmin(S_map):.4f}, {np.nanmax(S_map):.4f}]')
    print(f'  rank_eff ∈ [{int(np.nanmin(rk_map))}, {int(np.nanmax(rk_map))}]')