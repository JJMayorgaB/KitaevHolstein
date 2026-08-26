import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import qutip as qt
import warnings
warnings.filterwarnings('ignore')

omega0 = 1.0

# ══════════════════════════════════════════════════════════════════
#  FUNCIONAL Y MINIMIZACIÓN DEL GATO (2 params), detuning cero
# ══════════════════════════════════════════════════════════════════
def energy_grad_cat(x, Lam, s):
    a, r = x
    d = (Lam - a)
    A = d*d
    B = np.exp(2.0*r)
    F = np.exp(-0.5*A*B)
    E = omega0*(0.25*a*a + np.sinh(r)**2) - s*F
    dE_da = 0.5*omega0*a          - s*F*B*d
    dE_dr = omega0*np.sinh(2.0*r) + s*A*B*F
    return E, np.array([dE_da, dE_dr])


def _minimize_cat(lam1, lam2, s, x_warm=None):
    Lam = np.sqrt(lam1**2 + lam2**2)
    seeds = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = [(-Lam-1, Lam+1), (-1.0, 0.0)]
    lo = [b[0] for b in bnd]; hi = [b[1] for b in bnd]
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0 = np.clip(np.array(sd, float), lo, hi)
        res = minimize(energy_grad_cat, x0, args=(Lam, s), jac=True,
                       method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


def cat_bimodal(lam1, lam2, cv, N, sector='even', sign=+1, r_thresh=1e-5):
    Lam = np.sqrt(lam1**2 + lam2**2)
    if Lam < 1e-12:
        # Λ=0: sin acoplamiento, estado trivial
        vac = qt.tensor(qt.basis(N, 0), qt.basis(N, 0))
        return vac, (0.0, 0.0, 0.0, 0.0)

    E, x = _minimize_cat(lam1, lam2, cv)
    alpha, r = x[0], x[1]
    gam1 = lam1 * (0.5 - alpha/(2.0*Lam))
    gam2 = lam2 * (0.5 - alpha/(2.0*Lam))

    a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
    a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
    vac = qt.tensor(qt.basis(N, 0), qt.basis(N, 0))

    if abs(r) >= r_thresh:
        c1  = (lam1/Lam)**2
        c2  = (lam2/Lam)**2
        c12 = 2.0*lam1*lam2/Lam**2
        sgn_tm = +1.0 if sector == 'odd' else -1.0
        G = c1*(a1**2 - a1.dag()**2) + c2*(a2**2 - a2.dag()**2) \
            + sgn_tm * c12 * (a1*a2 - a1.dag()*a2.dag())
        S = ((r/2.0) * G).expm()
        base = S * vac
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
    return psi, (alpha, r, gam1, gam2)


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
    return (np.linspace(-x_max, x_max, n_x),
            np.linspace(-p_half, p_half, n_p))


#  EJECUCIÓN — mapa (Λ, θ) de δ1, δ2, E_N, S (variacional)
if __name__ == "__main__":

    cv_val   = 1.0
    sector   = 'even'
    sign     = +1
    N_FOCK   = 50
    r_thresh = 1e-5
    N_JOBS   = 120

    nLam = 32
    nth  = 32
    Lam_arr   = np.linspace(0.05, 6.0, nLam)
    theta_arr = np.linspace(0.0, 2.0*np.pi, nth)

    OUTDIR = f'cat_state_information_resources_omega0{omega0:.1f}_nL_{nLam}_nth_{nth}'
    TMPDIR = os.path.join(OUTDIR, '_tmp')
    os.makedirs(TMPDIR, exist_ok=True)

    def _one_point(iL, ith):
        Lam = Lam_arr[iL]; th = theta_arr[ith]
        lam1 = Lam*np.cos(th); lam2 = Lam*np.sin(th)

        psi, pars = cat_bimodal(lam1, lam2, cv_val, N_FOCK, sector=sector, sign=sign, r_thresh=r_thresh)
        E_N, S = entanglement_measures_var(psi, N_FOCK)

        xvec, pvec = wigner_grid(Lam)
        d1, d2, n1, n2 = marginal_negativities(psi, xvec, pvec)

        np.savetxt(os.path.join(TMPDIR, f'{iL:03d}_{ith:03d}.txt'), np.array([[Lam, th, d1, d2, E_N, S, n1, n2]]), fmt='%.8e')
        return iL, ith, d1, d2, E_N, S, n1, n2

    tasks = [(iL, ith) for iL in range(nLam) for ith in range(nth)]
    print(f'Puntos: {len(tasks)}   ({nLam} Λ × {nth} θ)', flush=True)

    _res = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_one_point)(iL, ith) for iL, ith in tasks)

    d1_map = np.full((nLam, nth), np.nan)
    d2_map = np.full((nLam, nth), np.nan)
    EN_map = np.full((nLam, nth), np.nan)
    S_map  = np.full((nLam, nth), np.nan)
    n1_map = np.full((nLam, nth), np.nan)
    n2_map = np.full((nLam, nth), np.nan)

    for iL, ith, d1, d2, E_N, S, n1, n2 in _res:
        d1_map[iL, ith] = d1; d2_map[iL, ith] = d2
        EN_map[iL, ith] = E_N; S_map[iL, ith]  = S
        n1_map[iL, ith] = n1; n2_map[iL, ith]  = n2

    hdr = (f'mapa variacional (Lambda, theta) | filas = Lambda ({nLam}), '
           f'columnas = theta ({nth}) | sector={sector}, sign={sign:+d}, '
           f'N_FOCK={N_FOCK}, r_thresh={r_thresh}')

    np.savetxt(f'{OUTDIR}/coord_Lam.txt',   Lam_arr,   fmt='%.8e', header='Lambda')
    np.savetxt(f'{OUTDIR}/coord_theta.txt', theta_arr, fmt='%.8e', header='theta')
    np.savetxt(f'{OUTDIR}/delta1.txt', d1_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/delta2.txt', d2_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/EN.txt',     EN_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/S.txt',      S_map,  fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/norm1.txt',  n1_map, fmt='%.8e', header=hdr)
    np.savetxt(f'{OUTDIR}/norm2.txt',  n2_map, fmt='%.8e', header=hdr)

    print(f'Guardado en {OUTDIR}/')
    print(f'  min ∫W1 = {np.nanmin(n1_map):.4f}   min ∫W2 = {np.nanmin(n2_map):.4f}')

#vim test writing 
