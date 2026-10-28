import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import expm_multiply
from joblib import Parallel, delayed
import qutip as qt
import warnings
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from variational_P_AP_cat import minimize_cat          # funcional cerrado de (P+AP)S + pulido + variación mínima

omega0 = 1.0

#  PARÁMETROS ÓPTIMOS DE (P+AP)S, detuning cero
#  El funcional solo depende de Λ: se minimiza una vez por Λ (no por cada θ), en continuación
#  creciente en Λ (pulido de Newton + variación mínima, como en variational_P_AP_cat.py).
def paps_params(Lam_arr, cv):
    """Columnas: η, r, κ, E  (una fila por Λ)."""
    P = np.zeros((len(Lam_arr), 4))
    x_warm = None
    for i, Lam in enumerate(Lam_arr):
        E, x = minimize_cat(Lam, cv, x_warm=x_warm)
        P[i, :3] = x; P[i, 3] = E
        x_warm = x
    return P


def paps_overlap(Lam, eta, r, kap):
    """o = <φ↑|D(Λ)|φ↓> cerrado (marco de laboratorio); p± = (1 ± o)/2."""
    F = np.exp(-0.5*(Lam - eta)**2*np.exp(2.0*r))
    return (2.0*kap + (1.0 + kap*kap)*F)/(1.0 + kap*kap + 2.0*kap*F)


def cat_bimodal(lam1, lam2, eta, r, N, sector='even', sign=+1, r_thresh=1e-5):
    """Gato de dos modos con (η, r) óptimos de (P+AP)S (η hace el papel de α del D S).
    κ no aparece: el gato normalizado de (P+AP)S tiene la misma forma que el de D S."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    if Lam < 1e-12:
        # Λ=0: sin acoplamiento, estado trivial
        vac = qt.tensor(qt.basis(N, 0), qt.basis(N, 0))
        return vac, (0.0, 0.0, 0.0, 0.0)

    gam1 = lam1 * (0.5 - eta/(2.0*Lam))
    gam2 = lam2 * (0.5 - eta/(2.0*Lam))

    a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
    a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
    vac = qt.tensor(qt.basis(N, 0), qt.basis(N, 0))

    if abs(r) >= r_thresh:
        c1  = (lam1/Lam)**2
        c2  = (lam2/Lam)**2
        c12 = 2.0*lam1*lam2/Lam**2
        sgn_tm = +1.0 if sector == 'even' else -1.0     # par: S_c (modo A_e), impar: S_r (modo a_o)
        G = c1*(a1**2 - a1.dag()**2) + c2*(a2**2 - a2.dag()**2) \
            + sgn_tm * c12 * (a1*a2 - a1.dag()*a2.dag())
        # S|0,0> = exp[(r/2) G]|0,0> sin construir la exponencial densa (mismo resultado que G.expm())
        v = expm_multiply((r/2.0) * sps.csr_matrix(G.full()), vac.full().ravel())
        base = qt.Qobj(v.reshape(-1, 1), dims=vac.dims)
    else:
        base = vac

    D1 = lambda g: qt.tensor(qt.displace(N, g), qt.qeye(N))
    D2 = lambda g: qt.tensor(qt.qeye(N), qt.displace(N, g))

    if sector == 'even':
        lobeA = D1(+gam1) * D2(+gam2) * base
        lobeB = D1(-gam1) * D2(-gam2) * base
    else:
        lobeA = D1(+gam1) * D2(-gam2) * base
        lobeB = D1(-gam1) * D2(+gam2) * base

    psi = (lobeA + sign * lobeB).unit()
    return psi, (eta, r, gam1, gam2)


def reduced_dm_mode(psi, mode):
    rho = psi * psi.dag()
    return rho.ptrace(0 if mode == 1 else 1)


def wigner_negativity_dm(rho1, xvec, pvec):
    W = qt.wigner(rho1, xvec, pvec)
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check


def marginal_negativities(psi, xvec, pvec):
    rho1 = reduced_dm_mode(psi, 1)
    rho2 = reduced_dm_mode(psi, 2)
    d1, n1 = wigner_negativity_dm(rho1, xvec, pvec)
    d2, n2 = wigner_negativity_dm(rho2, xvec, pvec)
    return d1, d2, n1, n2


def entanglement_measures_var(psi, N, tol=1e-12):
    """E_N y S del espectro de Schmidt (estado bipartito PURO)."""
    pm = psi.full().reshape(N, N)
    s  = np.linalg.svd(pm, compute_uv=False)
    s  = s[s > np.sqrt(tol)]
    E_N = 2.0*np.log2(np.sum(s))
    lam = s**2
    S   = -np.sum(lam*np.log2(lam))
    return E_N, S


def wigner_grid(Lam, n_x=301, n_p=251, x_pad=5.0, p_half=4.0):
    """Malla adaptativa"""
    x_max = 0.5*Lam + x_pad
    return (np.linspace(-x_max, x_max, n_x), np.linspace(-p_half, p_half, n_p))


#  EJECUCIÓN — mapa (Λ, θ) de δ1, δ2, E_N, S (variacional (P+AP)S, ramas Cat+ y Cat−)
if __name__ == "__main__":

    CV_LIST  = [1.0, 0.5, 1.5]   # J_α/ω0; cada valor en su subcarpeta cv{...}
    sector   = 'odd'
    SIGNS    = (+1, -1)          # ramas |Cat_+> (sign=+1) y |Cat_-> (sign=-1), mismos (η, r, κ)
    N_FOCK   = 40
    r_thresh = 1e-5
    N_JOBS   = 128

    nLam = 300
    nth  = 300
    Lam_arr   = np.linspace(0.05, 6.0, nLam)
    theta_arr = np.linspace(0.0, 2.0*np.pi, nth)

    FILES = ('paps_params', 'coord_Lam', 'coord_theta', 'delta1', 'delta2', 'EN', 'S', 'norm1', 'norm2')

    def outdir(sign, cv_val):
        return os.path.join(HERE, 'Figures', 'cat', 'data',
                            f'cat_state_information_resources_PAPS_sign{sign:+d}'
                            f'_omega0{omega0:.1f}_nL_{nLam}_nth_{nth}', f'cv{cv_val}')

    def _one_point(PAR, iL, ith):
        Lam = Lam_arr[iL]; th = theta_arr[ith]
        lam1 = Lam*np.cos(th); lam2 = Lam*np.sin(th)

        xvec, pvec = wigner_grid(Lam)
        out = []
        for sign in SIGNS:
            psi, pars = cat_bimodal(lam1, lam2, PAR[iL, 0], PAR[iL, 1], N_FOCK,
                                    sector=sector, sign=sign, r_thresh=r_thresh)
            E_N, S = entanglement_measures_var(psi, N_FOCK)
            d1, d2, n1, n2 = marginal_negativities(psi, xvec, pvec)
            out.append((d1, d2, E_N, S, n1, n2))

        return iL, ith, out

    for cv_val in CV_LIST:
        # si los mapas de esta J_α/ω0 ya están completos (ambas ramas), no se recalculan
        if all(os.path.exists(os.path.join(outdir(s, cv_val), f + '.txt')) for s in SIGNS for f in FILES):
            print(f'J_α/ω0 = {cv_val}: ya calculado, se omite', flush=True)
            continue
        for s in SIGNS:
            os.makedirs(outdir(s, cv_val), exist_ok=True)

        # 1) (η, r, κ) óptimos por Λ (serial, unos segundos); los mismos para ambas ramas
        PAR = paps_params(Lam_arr, cv_val)
        o_arr = paps_overlap(Lam_arr, PAR[:, 0], PAR[:, 1], PAR[:, 2])
        for sign in SIGNS:
            np.savetxt(os.path.join(outdir(sign, cv_val), 'paps_params.txt'),
                       np.column_stack([Lam_arr, PAR[:, 0], PAR[:, 1], PAR[:, 2], PAR[:, 3],
                                        0.5*(1.0 + o_arr), 0.5*(1.0 - o_arr)]),
                       fmt='%.10e',
                       header=(f'(P+AP)S, J/omega0={cv_val} | Lambda  eta  r  kappa  E  '
                               f'p_cat(sign=+1)=(1+o)/2  p_cat(sign=-1)=(1-o)/2,  o=<phi_up|D(Lambda)|phi_dn>'))
        print(f'J_α/ω0 = {cv_val}: parámetros (P+AP)S listos ({nLam} valores de Λ)', flush=True)

        tasks = [(iL, ith) for iL in range(nLam) for ith in range(nth)]
        print(f'J_α/ω0 = {cv_val}:  puntos {len(tasks)}   ({nLam} Λ × {nth} θ) × ramas {SIGNS}', flush=True)

        _res = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(_one_point)(PAR, iL, ith) for iL, ith in tasks)

        for k_s, sign in enumerate(SIGNS):
            d1_map = np.full((nLam, nth), np.nan)
            d2_map = np.full((nLam, nth), np.nan)
            EN_map = np.full((nLam, nth), np.nan)
            S_map  = np.full((nLam, nth), np.nan)
            n1_map = np.full((nLam, nth), np.nan)
            n2_map = np.full((nLam, nth), np.nan)

            for iL, ith, out in _res:
                d1, d2, E_N, S, n1, n2 = out[k_s]
                d1_map[iL, ith] = d1; d2_map[iL, ith] = d2
                EN_map[iL, ith] = E_N; S_map[iL, ith]  = S
                n1_map[iL, ith] = n1; n2_map[iL, ith]  = n2

            hdr = (f'mapa variacional (P+AP)S (Lambda, theta) | filas = Lambda ({nLam}), '
                   f'columnas = theta ({nth}) | sector={sector}, sign={sign:+d}, J/omega0={cv_val}, '
                   f'N_FOCK={N_FOCK}, r_thresh={r_thresh}')
            D = outdir(sign, cv_val)

            np.savetxt(f'{D}/coord_Lam.txt',   Lam_arr,   fmt='%.8e', header='Lambda')
            np.savetxt(f'{D}/coord_theta.txt', theta_arr, fmt='%.8e', header='theta')
            np.savetxt(f'{D}/delta1.txt', d1_map, fmt='%.8e', header=hdr)
            np.savetxt(f'{D}/delta2.txt', d2_map, fmt='%.8e', header=hdr)
            np.savetxt(f'{D}/EN.txt',     EN_map, fmt='%.8e', header=hdr)
            np.savetxt(f'{D}/S.txt',      S_map,  fmt='%.8e', header=hdr)
            np.savetxt(f'{D}/norm1.txt',  n1_map, fmt='%.8e', header=hdr)
            np.savetxt(f'{D}/norm2.txt',  n2_map, fmt='%.8e', header=hdr)

            print(f'Guardado en {D}/')
            print(f'  min ∫W1 = {np.nanmin(n1_map):.4f}   min ∫W2 = {np.nanmin(n2_map):.4f}', flush=True)
