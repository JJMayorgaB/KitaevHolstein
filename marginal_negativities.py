import numpy as np
from scipy.special import factorial, eval_genlaguerre
from scipy.optimize import minimize
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from joblib import Parallel, delayed
import contextlib, joblib
from tqdm.auto import tqdm
import qutip as qt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import warnings
from math import comb as _C
from matplotlib.ticker import PercentFormatter
warnings.filterwarnings('ignore')
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
import matplotlib.colors as colors
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


def get_M_CM(lam1, lam2, M_max=12):
    """Truncación adaptativa. El desplazamiento efectivo en los modos
    relativo/CoM es Λ = sqrt(λ₁²+λ₂²), por eso los umbrales se ajustan a Λ."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    if   Lam < 0.5: return 5
    elif Lam < 1.5: return 9
    else:           return M_max


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

def disp_matrix_complex(M_ph, alpha):
    """Elementos de matriz <m'|D(alpha)|n> en base de Fock, alpha complejo.
    D(alpha) = exp(alpha a^dag - alpha* a)
    """
    N       = M_ph + 1
    alpha   = complex(alpha)
    alpha_c = np.conj(alpha)
    abs2    = (alpha * alpha_c).real
    gauss   = np.exp(-0.5 * abs2)
    facts   = np.array([float(factorial(k, exact=True)) for k in range(N)])

    D = np.zeros((N, N), dtype=complex)
    for mp in range(N):
        for n in range(N):
            if mp >= n:
                k    = mp - n
                pref = gauss * (alpha**k) * np.sqrt(facts[n] / facts[mp])
                D[mp, n] = pref * eval_genlaguerre(n, k, abs2)
            else:
                k    = n - mp
                pref = gauss * ((-alpha_c)**k) * np.sqrt(facts[mp] / facts[n])
                D[mp, n] = pref * eval_genlaguerre(mp, k, abs2)
    return D


def MofLam_sweet_exact(g1, g2, omega0=1.0, n_sigma=4, floor=15, M_cap=55):
    lam_max = max(abs(g1), abs(g2)) / omega0
    M = int(np.ceil(lam_max**2 + n_sigma*lam_max)) 
    M = max(M, floor)          # piso para λ pequeño
    return min(M, M_cap)

def build_LF_hamiltonians_sweet(M, t, Delta, omega1, omega2, g1, g2):
    """
    Hamiltoniano LF en base de SITIO, en el SWEET SPOT (ε̃₁=ε̃₂=0).
    En el sweet spot ε_i = g_i²/ω_i, así que ε_i_eff = ε_i - g_i²/ω_i = 0
    y los bloques electrónicos no llevan offset diagonal.
    """
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

    # Sector par (ε̃₁=ε̃₂=0 → ambos bloques con solo energía fotónica)
    H_e = np.zeros((2*n_ph, 2*n_ph))
    H_e[:n_ph, :n_ph] = np.diag(ph_E)
    H_e[n_ph:, n_ph:] = np.diag(ph_E)
    H_e[:n_ph, n_ph:] = T_pair
    H_e[n_ph:, :n_ph] = T_pair.T

    # Sector impar (ε̃₁=ε̃₂=0)
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

def wigner_joint_cut_exact(psi_ph, nph, uvec, vvec, plane='real', verbose=True, tag=''):
    """
    Wigner conjunta de dos modos W(α1,α2) en un corte 2D, para un
    vector de Fock de dos modos (dim nph², idx = m1*nph + m2).
      plane='real': Re(α1)=u vs Re(α2)=v   (Im=0)
      plane='imag': Im(α1)=u vs Im(α2)=v   (Re=0)
    W = (2/π)² Σ_{n1,n2}(-1)^{n1+n2}|<n1 n2|D1†(α1)⊗D2†(α2)|ψ>|²
    Con ψ como matriz [n1,n2]: D1† por la izquierda, D2†^T por la derecha.
    """
    psi_mat = psi_ph.reshape(nph, nph)          # [m1, m2]
    par = (-1.0) ** np.arange(nph)
    Par = np.outer(par, par)

    def _Ddag(val):
        a = (val + 0j) if plane == 'real' else (1j*val)
        return disp_matrix_complex(nph-1, -a)   # D†(α) = D(-α)

    Ddag_u = [_Ddag(u) for u in uvec]
    Ddag_vT = [_Ddag(v).T for v in vvec]

    pref = (2.0/np.pi)**2
    nu = len(uvec)
    step10 = max(1, nu // 10)

    W = np.zeros((len(vvec), len(uvec)))
    for iu in range(nu):
        left = Ddag_u[iu] @ psi_mat             # [n1', m2]
        for iv in range(len(vvec)):
            tpsi = left @ Ddag_vT[iv]           # [n1', n2']
            W[iv, iu] = pref * np.sum(Par * (np.abs(tpsi)**2))
        if verbose and ((iu+1) % step10 == 0 or (iu+1) == nu):
            print(f"      [{tag}] {100.0*(iu+1)/nu:5.1f}%", flush=True)
    return W


def wigner_joint_and_cuts_exact(psi_ph, nph, uvec, vvec, plane='real', verbose=True, tag=''):
    W = wigner_joint_cut_exact(psi_ph, nph, uvec, vvec, plane=plane, verbose=verbose, tag=tag)
    iv0 = np.argmin(np.abs(vvec))
    iu0 = np.argmin(np.abs(uvec))
    return W, W[iv0, :], W[:, iu0]

def reduced_dm_mode_exact(psi_ph, nph, mode):
    """Matriz densidad reducida de un modo, desde el vector de Fock de dos modos.
       psi_ph: vector dim nph² (idx = m1*nph + m2).
       mode=1 → ρ1 = Tr_2 (contrae m2);  mode=2 → ρ2 = Tr_1 (contrae m1).
       Devuelve matriz nph×nph."""
    psi_mat = psi_ph.reshape(nph, nph)          # [m1, m2]
    if mode == 1:
        rho = psi_mat @ psi_mat.conj().T        # ρ1[m1,m1'] = Σ_m2 ψ[m1,m2]ψ*[m1',m2]
    else:
        rho = psi_mat.T @ psi_mat.conj()        # ρ2[m2,m2'] = Σ_m1 ψ[m1,m2]ψ*[m1,m2']
    return rho


def wigner_dm_negativity(rho, xvec, pvec, verbose=False, n_jobs=-1):
    """Wigner de un modo de una matriz densidad ρ (nph×nph) + su negatividad.
       W(α) = (2/π) Σ_n (-1)^n <n|D†(α) ρ D(α)|n>
            = (2/π) Σ_n (-1)^n [D†(α) ρ D(α)]_{nn}
       Vectorizado por punto: para cada α, calcula D(-α) (=D†(α)) y la forma
       cuadrática. Reusa disp_matrix_complex."""
    N = rho.shape[0]
    parity = (-1.0) ** np.arange(N)

    def _row(p):
        row = np.zeros(len(xvec))
        for ix, x in enumerate(xvec):
            alpha = (x + 1j*p)
            Dd = disp_matrix_complex(N-1, -alpha)      # D†(α) = D(-α)
            # [D† ρ D]_{nn} = (D† ρ D)diag ; D = Dd†
            M_ = Dd @ rho @ Dd.conj().T
            row[ix] = (2.0/np.pi) * np.sum(parity * np.diag(M_).real)
        return row

    if n_jobs == 1:
        W = np.array([_row(p) for p in pvec])
    else:
        import joblib
        if verbose:
            print(f"   Wigner reducida: {len(pvec)} filas en paralelo ...", flush=True)
        W = np.array(joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_row)(p) for p in pvec))

    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check

def pad_state(psi_ph, nph_old, nph_new):
    """Expande el estado de dos modos de nph_old² a nph_new² con ceros en Fock altos."""
    pm = psi_ph.reshape(nph_old, nph_old)
    pm_new = np.zeros((nph_new, nph_new), dtype=complex)
    pm_new[:nph_old, :nph_old] = pm
    return pm_new.reshape(-1)


def marginal_negativities_exact(psi_ph, nph, xvec, pvec, nph_wig=56, n_jobs=-1):
    """δ1, δ2 marginales. Hace padding del estado a nph_wig para Wigner fiel
       (D(α) unitaria en el borde de la malla) sin encarecer la diagonalización."""
    if nph_wig > nph:
        psi_ph = pad_state(psi_ph, nph, nph_wig)
        nph = nph_wig
    pm = psi_ph.reshape(nph, nph)
    rho1 = pm @ pm.conj().T
    rho2 = pm.T @ pm.conj()
    d1, n1 = wigner_dm_negativity(rho1, xvec, pvec, n_jobs=n_jobs)
    d2, n2 = wigner_dm_negativity(rho2, xvec, pvec, n_jobs=n_jobs)
    return d1, d2, n1, n2


def exact_cat_bimodal(g1, g2, evec_ex, nph, n_ph, omega0=1.0, sign=+1, sector='even'):
    lam1 = g1 / omega0
    lam2 = g2 / omega0

    phi1 = evec_ex[:n_ph]
    phi2 = evec_ex[n_ph:]

    D1p = disp_matrix(M, +0.5*lam1); D1m = disp_matrix(M, -0.5*lam1)
    D2p = disp_matrix(M, +0.5*lam2); D2m = disp_matrix(M, -0.5*lam2)

    if sector == 'even':
        Dp = np.kron(D1p, D2p)   # φ2: (+λ1/2, +λ2/2)
        Dm = np.kron(D1m, D2m)   # φ1: (-λ1/2, -λ2/2)
    else:  # 'odd'
        Dp = np.kron(D1p, D2m)   # φ2: (+λ1/2, -λ2/2)
        Dm = np.kron(D1m, D2p)   # φ1: (-λ1/2, +λ2/2)

    cat = Dp @ phi2 + (-sign) * (Dm @ phi1)
    cat /= np.linalg.norm(cat)
    return cat

# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN — negatividades marginales δ1, δ2 vs θ (Λ fijo, EXACTO sitio)
#    θ ∈ (0, π);  λ1 = Λcosθ, λ2 = Λsinθ  →  g_i = λ_i·ω0
# ══════════════════════════════════════════════════════════════════
omega0   = 1.0
cv_val   = 1.0
Lam_fix  = 3*np.sqrt(2)       # Λ fijo
sector   = 'even'
sign     = +1
nph_wig  = 85              

nth      = 1
theta_arr = np.linspace(0.0, np.pi, nth)

xvec = np.linspace(-4.2, 4.2, 251)
pvec = np.linspace(-3.0, 3.0, 251)

d1_arr = np.full(nth, np.nan); d2_arr = np.full(nth, np.nan)
n1_arr = np.full(nth, np.nan); n2_arr = np.full(nth, np.nan)

_step10 = max(1, nth // 10)
for k, th in enumerate(theta_arr):
    g1 = Lam_fix * np.cos(th)
    g2 = Lam_fix * np.sin(th)
    M = min(MofLam_sweet_exact(g1, g2, omega0=omega0, M_cap=45) + 25, 55)   # M de diagonalización (barato)
    ev_e, evec_e, ev_o, evec_o, nph, n_ph = diagonalize_LF_sweet(M, cv_val, cv_val, omega0, omega0, g1, g2)
    vec = evec_e[:, 0] if sector == 'even' else evec_o[:, 0]
    psi_ph = exact_cat_bimodal(g1, g2, vec, nph, n_ph, omega0, sign=sign, sector=sector)

    # marginal con padding a nph_wig para Wigner fiel
    d1_arr[k], d2_arr[k], n1_arr[k], n2_arr[k] = marginal_negativities_exact(psi_ph, nph, xvec, pvec, nph_wig=nph_wig)
    if k == 0:
        print(f" θ={th:5.3f}  δ1={d1_arr[k]:.4f} δ2={d2_arr[k]:.4f} " f"(∫W1={n1_arr[k]:.3f}, ∫W2={n2_arr[k]:.3f})", flush=True)
    if (k+1) % _step10 == 0 or (k+1) == nth:
        print(f"  {100.0*(k+1)/nth:5.1f}%  θ={th:5.3f}  " f"δ1={d1_arr[k]:.4f} δ2={d2_arr[k]:.4f} " f"(∫W1={n1_arr[k]:.3f}, ∫W2={n2_arr[k]:.3f})", flush=True)

print("Barrido marginal θ (exacto) listo.")

os.makedirs('Resultados_aniso', exist_ok=True)
_out = np.column_stack([theta_arr, d1_arr, d2_arr, n1_arr, n2_arr])
np.savetxt(f'Resultados_aniso/marginal_negativity_vs_theta_exact_{Lam_fix}.txt', _out, fmt='%.8e', header='theta  delta1  delta2  norm1  norm2')
print(f'Guardado: marginal_negativity_vs_theta_exact_{Lam_fix}.txt')