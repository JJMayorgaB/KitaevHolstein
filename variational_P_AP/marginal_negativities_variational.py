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
from marginal_negativities import (disp_matrix, entanglement_measures_pure,
                                   wigner_dm_negativity)
from variational_P_AP_cat import minimize_cat          # funcional cerrado de (P+AP)S + pulido de Newton

# ══════════════════════════════════════════════════════════════════
#  Contraparte variacional (P+AP)S de marginal_negativities.py:
#  δ1, δ2, E_N y S vs θ a Λ fijo, con la MISMA base de sitio (nph_wig), la MISMA malla
#  de Wigner y las MISMAS medidas que el exacto.
# ══════════════════════════════════════════════════════════════════


def var_cat_bimodal(lam1, lam2, eta, r, nph, sector='odd', sign=+1):
    """Gato (P+AP)S en base de sitio, vector idx = m1*nph + m2 (como exact_cat_bimodal).
    Lóbulos D1(±γ1) D2(±γ2) S(r)|0,0>,  γ_i = λ_i(1/2 − η/(2Λ)),  en el impar el lóbulo +
    va con (+γ1, −γ2).  S(r) = exp[(r/2)(b² − b†²)],  b = (λ1 â1 + λ2 â2)/Λ (par, modo A_e)
    o b = (λ1 â1 − λ2 â2)/Λ (impar, modo a_o).  κ no aparece: el gato normalizado de
    (P+AP)S tiene la misma forma que el de D S.
    Desplazamientos con disp_matrix (elementos de matriz exactos), como en exact_cat_bimodal."""
    Lam  = np.sqrt(lam1**2 + lam2**2)
    gam1 = lam1 * (0.5 - eta/(2.0*Lam))
    gam2 = lam2 * (0.5 - eta/(2.0*Lam))
    sgn_tm = +1.0 if sector == 'even' else -1.0

    a  = sps.diags(np.sqrt(np.arange(1, nph, dtype=float)), 1, format='csr')
    I  = sps.identity(nph, format='csr')
    a1 = sps.kron(a, I, format='csr')
    a2 = sps.kron(I, a, format='csr')
    b  = (lam1*a1 + sgn_tm*lam2*a2) / Lam
    bd = b.T.tocsr()

    vac = np.zeros(nph*nph); vac[0] = 1.0
    base = expm_multiply((0.5*r) * (b @ b - bd @ bd), vac) if r != 0.0 else vac
    P = base.reshape(nph, nph)

    D1p = disp_matrix(nph-1, +gam1); D1m = disp_matrix(nph-1, -gam1)
    D2p = disp_matrix(nph-1, +gam2); D2m = disp_matrix(nph-1, -gam2)
    if sector == 'even':
        catm = (D1p @ P @ D2p.T) + sign*(D1m @ P @ D2m.T)
    else:
        catm = (D1p @ P @ D2m.T) + sign*(D1m @ P @ D2p.T)

    cat = catm.reshape(-1)
    return cat / np.linalg.norm(cat)


def wigner_dm_negativity_fast(rho, xvec, pvec):
    """La misma W(α), α = x + ip, y la misma integración que wigner_dm_negativity
    (marginal_negativities.py), evaluada con qutip.wigner: con g = 2 qutip usa
    α = g(x + ip)/2 = x + ip y la normalización ∫W dx dp = 1."""
    W = qt.wigner(qt.Qobj(rho), xvec, pvec, g=2.0)
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check


def marginal_negativities_var(psi_ph, nph, xvec, pvec, fast=True):
    """δ1, δ2 marginales + E_N, S y rango efectivo (igual que marginal_negativities_exact)."""
    pm = psi_ph.reshape(nph, nph)
    rho1 = pm @ pm.conj().T
    rho2 = pm.T @ pm.conj()
    E_N, S, rank_eff = entanglement_measures_pure(rho1)
    if fast:
        d1, n1 = wigner_dm_negativity_fast(rho1, xvec, pvec)
        d2, n2 = wigner_dm_negativity_fast(rho2, xvec, pvec)
    else:
        d1, n1 = wigner_dm_negativity(rho1, xvec, pvec, n_jobs=1)
        d2, n2 = wigner_dm_negativity(rho2, xvec, pvec, n_jobs=1)
    return d1, d2, n1, n2, E_N, S, rank_eff


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN — δ1, δ2, E_N y S vs θ (Λ fijo, VARIACIONAL (P+AP)S, ramas Cat+ y Cat−)
#    θ ∈ (0, π);  λ1 = Λcosθ, λ2 = Λsinθ
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    omega0   = 1.0
    cv_val   = 1.0
    LAM_LIST = [2*np.sqrt(2), 3*np.sqrt(2)]      # los mismos Λ del exacto
    sector   = 'odd'
    SIGNS    = (+1, -1)           # ramas |Cat_+> (sign=+1) y |Cat_-> (sign=-1), mismos (η, r, κ)
    nph_wig  = 85                 # la misma base de sitio del exacto
    N_JOBS   = 128
    FAST_WIGNER = True            # False: wigner_dm_negativity del exacto (~10 min por θ en lugar de ~1 s)

    nth       = 301               # más denso que el exacto (64): las curvas son baratas
    theta_arr = np.linspace(0.0, np.pi, nth)

    xvec = np.linspace(-4.2, 4.2, 251)           # la misma malla del exacto
    pvec = np.linspace(-3.0, 3.0, 251)

    OUTDIR = os.path.join(HERE, 'Figures', 'cat', 'data')
    os.makedirs(OUTDIR, exist_ok=True)

    def _one_theta(Lam_fix, eta, r, k, th):
        lam1 = Lam_fix * np.cos(th)
        lam2 = Lam_fix * np.sin(th)
        out = []
        for sign in SIGNS:
            psi_ph = var_cat_bimodal(lam1, lam2, eta, r, nph_wig, sector=sector, sign=sign)
            out.append(marginal_negativities_var(psi_ph, nph_wig, xvec, pvec, fast=FAST_WIGNER))
        return k, out

    for Lam_fix in LAM_LIST:
        # el funcional solo depende de Λ: una minimización para todo el corte
        E, (eta, r, kap) = minimize_cat(Lam_fix, cv_val)
        F = np.exp(-0.5*(Lam_fix - eta)**2*np.exp(2.0*r))
        o = (2.0*kap + (1.0 + kap*kap)*F)/(1.0 + kap*kap + 2.0*kap*F)
        print(f'Λ = {Lam_fix:.6f}:  η = {eta:.8f}  r = {r:.8f}  κ = {kap:.8f}  '
              f'E = {E:.10f}  p_cat(+) = {0.5*(1.0 + o):.6f}  p_cat(−) = {0.5*(1.0 - o):.6f}', flush=True)

        _res = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_one_theta)(Lam_fix, eta, r, k, th) for k, th in enumerate(theta_arr))

        for k_s, sign in enumerate(SIGNS):
            p_cat = 0.5*(1.0 + sign*o)
            _out = np.full((nth, 8), np.nan)
            _out[:, 0] = theta_arr
            for k, out in _res:
                _out[k, 1:] = out[k_s]

            fname = os.path.join(OUTDIR, f'marginal_negativity_vs_theta_PAPS_sign{sign:+d}_{Lam_fix}.txt')
            np.savetxt(fname, _out, fmt='%.8e',
                       header=(f'(P+AP)S, sector={sector}, sign={sign:+d}, Lambda={Lam_fix}, '
                               f'J/omega0={cv_val}, nph={nph_wig} | eta={eta:.10e} r={r:.10e} '
                               f'kappa={kap:.10e} E={E:.10e} p_cat={p_cat:.10e}\n'
                               'theta  delta1  delta2  norm1  norm2  E_N  S  rank_eff'))
            print(f'Guardado: {fname}')
            print(f'  min ∫W1 = {np.nanmin(_out[:, 3]):.6f}   min ∫W2 = {np.nanmin(_out[:, 4]):.6f}')

    print("Barrido marginal θ (P+AP)S listo.")
