# -*- coding: utf-8 -*-
"""
Estados gato en detuning cero con el ansatz variacional con trisqueezing.

En detuning cero (δ_α = 0) el bloque es
    H = [[ ω0 b†b ,  s D(Λ) ],
         [ s D(Λ)† , ω0 b†b ]],      s = t (impar) o Δ (par),
y la simetría Ŝ = s_x e^{iπ b†b} fija (notas):
    |φ↑> = D(η/2) S(r) T(τ)|0>,   |φ↓> = Π|φ↑> = D(−η/2) S(r) T(−τ)|0>,
    E(η, r, τ) = ω0 <m>↑ − s <φ↑|D(Λ)|φ↓>.
Con τ = 0 se recupera el funcional del gato E(η, r) del código original.

Todo depende solo de Λ: se usa el corte θ = π/4 (λ₁ = λ₂ = Λ/√2).

Corridas (MODE = 'all'):
    1) τ = 0 fijo (prueba: reproduce las figuras originales, sufijo _tau0)
    2) trisqueezing (τ libre)
Todas las figuras se guardan en Resultados_variational_trisqueezing_cat/.
La parte EXACTA (diagonalización, gato exacto, solapamiento exacto con cortes
escalonados y el suavizado de las curvas exactas) se mantiene igual que en el
código original.
"""

import io
import os
import shutil
import warnings
from math import comb as _C

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import factorial, eval_genlaguerre
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
import scipy.sparse as sps
from scipy.sparse.linalg import expm_multiply
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
MODE    = 'all'                 # 'tau0' | 'trisq' | 'all'  ('all' corre primero τ = 0)
N_JOBS  = -1

omega0  = 1.0
cv_list = [0.5, 1.0, 1.5]       # J_α/ω0 = s/ω0
TAU_MAX = 0.08                  # cota de |τ| (convergencia de T(τ)|0>, ver tests)
RHO     = 1.0                   # corte θ = π/4

# mallas en Λ (las mismas de las figuras originales)
LAM_PARAMS     = np.linspace(0.01, 4.0*np.sqrt(2), 200)        # parámetros del gato
LAM_INFID_TWIN = np.linspace(0.01, 5.0, 200)                   # infidelidad 1D, eje doble (+ 2x2)
LAM_INFID      = np.linspace(0.01, 5.0, 300)                   # infidelidad 1D, un eje
LAM_CUT        = np.sqrt(2.0)*np.linspace(0.01, 4.0, 300)      # p±, β̃, solapamientos

# negatividad δ_W (cv = 1)
CV_NEG      = 1.0
LAM_NEG_EX  = np.linspace(0.01, 7.0, 200)
# malla común (exacto, DS, DST) en coordenadas de α, Δx = Δp = 0.02.
# Convergencia verificada contra x∈[−10,10], p∈[−7,7] (1501×1051): |Δδ| ≤ 6e-5, ∫W = 1.
X_NEG_EX    = np.linspace(-8.0, 8.0, 801)
P_NEG_EX    = np.linspace(-5.5, 5.5, 551)
LAM_NEG_VAR = np.linspace(0.01, 7.0, 300)
X_NEG_VAR   = X_NEG_EX
P_NEG_VAR   = P_NEG_EX

# Wigner exacto vs variacional (λ₁ = λ₂ = λ_val)
CV_WIG     = 1.0
WIG_POINTS = [(3.0, (7.0, 7.0)), (2.0, (9.0, 9.0))]
WIG_GRID   = np.linspace(-4.2, 4.2, 301)

# Wigner conjunta de dos modos
CV_JOINT     = 1.0
LAM_JOINT    = 3.0*np.sqrt(2)
THETA_JOINT  = np.pi/4
N_FOCK_JOINT = 50
SECTOR_JOINT = 'even'
SIGN_JOINT   = -1
UV_JOINT     = np.linspace(-3.5, 3.5, 201)

CV_2X2 = 1.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Resultados_variational_trisqueezing_cat')

# ══════════════════════════════════════════════════════════════════
#  ESTILO (el original; si no hay LaTeX, mathtext con fuente cm)
# ══════════════════════════════════════════════════════════════════
_BASE_RC = {
    'font.size': 26,
    'axes.labelsize': 26,
    'axes.titlesize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
}


def _latex_available():
    if shutil.which('latex') is None:
        return False
    try:
        with plt.rc_context({'text.usetex': True}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r'$\alpha$')
            fig.savefig(io.BytesIO(), format='png')
        plt.close('all')
        return True
    except Exception:
        plt.close('all')
        return False


def setup_style():
    HAS_LATEX = _latex_available()
    rc = dict(_BASE_RC)
    rc.update({
        'text.usetex': HAS_LATEX,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
    })
    plt.rcParams.update(rc)
    print('[estilo] ' + ('LaTeX disponible: usetex' if HAS_LATEX
                         else 'LaTeX no disponible: mathtext (cm)'))
    return HAS_LATEX


def _trapz(y, x, axis=-1):
    f = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return f(y, x, axis=axis)


# ══════════════════════════════════════════════════════════════════
#  FOCK (funciones originales)
# ══════════════════════════════════════════════════════════════════
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
    """Elementos de matriz <m'|D(alpha)|n>, alpha complejo."""
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


def squeezed_vacuum(M, r):
    """S(r)|0>, convención S(r)=exp[r/2(â²−â†²)]."""
    vec  = np.zeros(M + 1)
    pref = 1.0 / np.sqrt(np.cosh(r))
    th   = np.tanh(r)
    for k in range(M // 2 + 1):
        if 2*k <= M:
            vec[2*k] = pref * ((-th)**k) * np.sqrt(float(_C(2*k, k))) / (2.0**k)
    return vec


def _Lambda(lam1, lam2):
    return np.sqrt(lam1**2 + lam2**2)


def _Mof_Lam(Lam):
    return int(min(170, max(50, np.ceil(4.0*Lam*Lam) + 25)))


def _lam12_from_Lam_rho(Lam, rho):
    lam2 = Lam / np.sqrt(1.0 + rho**2)
    lam1 = rho * lam2
    return lam1, lam2


# ══════════════════════════════════════════════════════════════════
#  OPERADORES DEL ANSATZ (construidos a mano, como en variational_trisqueezing.py)
# ══════════════════════════════════════════════════════════════════
class FockOps:
    """G1 = b†−b (D), G2 = (b²−b†²)/2 (S), G3 = b†³−b³ (T);
    exp(xG) = V exp(−i x w) V†, con H = iG = V w V†."""

    def __init__(self, M):
        self.M    = M
        self.dim  = M + 1
        self.nvec = np.arange(self.dim, dtype=float)
        b  = np.diag(np.sqrt(np.arange(1, self.dim, dtype=float)), 1)
        bd = b.T
        self.b = b
        gens = {1: bd - b, 2: 0.5*(b @ b - bd @ bd), 3: bd @ bd @ bd - b @ b @ b}
        self.eig = {}
        for k, G in gens.items():
            w, V = eigh(1j*G)
            self.eig[k] = (w, V, V.conj().T)

    def apply(self, k, x, v):
        if x == 0.0:
            return v
        w, V, Vh = self.eig[k]
        return (V @ (np.exp(-1j*x*w) * (Vh @ v))).real

    def state(self, eta, r, tau):
        """|τ, r, η> = D(η) S(r) T(τ)|0>."""
        v = np.zeros(self.dim)
        v[0] = 1.0
        v = self.apply(3, tau, v)
        v = self.apply(2, r, v)
        v = self.apply(1, eta, v)
        return v


_OPS_CACHE = {}


def get_ops(M):
    if M not in _OPS_CACHE:
        _OPS_CACHE[M] = FockOps(M)
    return _OPS_CACHE[M]


# ══════════════════════════════════════════════════════════════════
#  FUNCIONAL DEL GATO
# ══════════════════════════════════════════════════════════════════
def energy_grad_cat(x, Lam, s):
    """Funcional analítico original E(a, r) (τ = 0). Solo para los tests."""
    a, r = x
    d = (Lam - a)
    A = d*d
    B = np.exp(2.0*r)
    F = np.exp(-0.5*A*B)
    E = omega0*(0.25*a*a + np.sinh(r)**2) - s*F
    dE_da = 0.5*omega0*a       - s*F*B*d
    dE_dr = omega0*np.sinh(2.0*r) + s*A*B*F
    return E, np.array([dE_da, dE_dr])


def _minimize_cat(lam1, lam2, s, x_warm=None):
    """Minimizador analítico original (τ = 0). Solo para los tests."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    seeds = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = [(-Lam-1, Lam+1), (-1.0, 0.0)]
    lo = [b[0] for b in bnd]; hi = [b[1] for b in bnd]
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0 = np.clip(np.array(sd, float), lo, hi)
        res = minimize(energy_grad_cat, x0, args=(Lam, s), jac=True, method='L-BFGS-B',
                       bounds=bnd, options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


def cat_quantities(x, Lam, s, ops, fix_tau0):
    """E, F = <φ↑|D(Λ)|φ↓>, |φ↑>, |φ↓> con x = (η, r[, τ])."""
    eta, r = x[0], x[1]
    tau = 0.0 if fix_tau0 else x[2]
    pu = ops.state(+0.5*eta, r, +tau)
    pd = ops.state(-0.5*eta, r, -tau)
    m  = pu @ (ops.nvec*pu)
    F  = pu @ ops.apply(1, Lam, pd)
    E  = omega0*m - s*F
    return E, F, pu, pd


def energy_cat_num(x, Lam, s, ops, fix_tau0):
    return cat_quantities(x, Lam, s, ops, fix_tau0)[0]


def minimize_cat_num(Lam, s, M, fix_tau0, x_warm=None):
    """Devuelve (E, (η, r, τ)). Mismas semillas y cotas que _minimize_cat."""
    ops  = get_ops(M)
    base = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    bnd  = [(-Lam - 1.0, Lam + 1.0), (-1.0, 0.0)]
    if fix_tau0:
        seeds = list(base)
        n_par = 2
    else:
        seeds = [sd + (0.0,) for sd in base]
        seeds += [sd + (tt,) for sd in [(Lam, -0.25), (0.5*Lam, -0.1)] for tt in (0.02, -0.02)]
        bnd  += [(-TAU_MAX, TAU_MAX)]
        n_par = 3
    if x_warm is not None:
        seeds = [tuple(x_warm[:n_par])] + seeds
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0  = np.clip(np.asarray(sd, float), lo, hi)
        res = minimize(energy_cat_num, x0, args=(Lam, s, ops, fix_tau0), method='L-BFGS-B',
                       bounds=bnd, options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    if fix_tau0:
        bestx = np.concatenate([bestx, [0.0]])
    return bestE, bestx


def variational_cat_vector(Lam, x, sign, M):
    """Gato variacional (modo b, marco original, centrado):
    D(β)S(r)T(−τ)|0> + sign·D(−β)S(r)T(τ)|0>, β = (Λ−η)/2, normalizado."""
    ops = get_ops(M)
    eta, r, tau = x
    beta = 0.5*(Lam - eta)
    psi  = ops.state(beta, r, -tau) + sign*ops.state(-beta, r, tau)
    return psi/np.linalg.norm(psi)


# ══════════════════════════════════════════════════════════════════
#  EXACTO Y FIDELIDAD DEL GATO (parte exacta igual a la original)
# ══════════════════════════════════════════════════════════════════
def E_cat_exact(lam1, lam2, s, M=None):
    """GS exacto en detuning cero. Devuelve (E, evec, M)."""
    Lam = _Lambda(lam1, lam2)
    if M is None:
        M = _Mof_Lam(Lam)
    n1 = M + 1
    DL = disp_matrix(M, Lam)
    n  = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = n
    H[n1:, n1:] = n
    H[:n1, n1:] = s*DL
    H[n1:, :n1] = s*DL.T
    ev, V = eigh(H)
    return ev[0], V[:, 0], M


def _cat_overlap(lam1, lam2, x, evec_ex, M, sign=+1):
    """(|fid|, o_ex, o_var) para la rama sign del gato.
    Parte exacta idéntica a la original (cortes escalonados para o_ex).
    Parte variacional: lóbulos |τ, r, ±η/2> del ansatz con trisqueezing."""
    LAM_CUTS = np.linspace(1.5, 4.0, 20)
    CUTOFFS  = np.linspace(0.999, 0.995, 20)

    Lam = _Lambda(lam1, lam2)
    n1  = M + 1
    ops = get_ops(M)

    phi1 = evec_ex[:n1]
    phi2 = evec_ex[n1:]

    eta, r, tau = x
    g_pa = ops.state(+0.5*eta, r, +tau)     # |τ, r, η/2>   (rama ↑)
    g_ma = ops.state(-0.5*eta, r, -tau)     # |−τ, r, −η/2> (rama ↓)

    DpL2 = disp_matrix(M, +0.5*Lam)
    DmL2 = disp_matrix(M, -0.5*Lam)

    cat_var = DpL2 @ g_ma + sign * (DmL2 @ g_pa)
    cat_var /= np.linalg.norm(cat_var)

    cat_ex = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    cat_ex /= np.linalg.norm(cat_ex)

    fid_overlap = np.vdot(cat_var, cat_ex)

    o_var = g_pa @ ops.apply(1, Lam, g_ma)  # <φ↑|D(Λ)|φ↓>

    n_passed = int(np.searchsorted(LAM_CUTS, Lam, side='right'))
    if n_passed == 0:
        o_ex = 2.0 * np.vdot(phi1, disp_matrix(M, +Lam) @ phi2).real
    else:
        c = CUTOFFS[min(n_passed - 1, len(CUTOFFS) - 1)]

        def _Mc(v):
            cs = np.cumsum(np.abs(v)**2) / max(np.vdot(v, v).real, 1e-30)
            return int(np.searchsorted(cs, c)) + 1

        Mc = min(max(_Mc(phi1), _Mc(phi2)), n1)
        o_ex = 2.0 * np.vdot(phi1[:Mc], disp_matrix(Mc - 1, +Lam) @ phi2[:Mc]).real

    return abs(fid_overlap), o_ex, o_var


def exact_cat_state(lam1, lam2, evec_ex, M, sign=+1):
    Lam = _Lambda(lam1, lam2)
    n1  = M + 1
    phi1 = evec_ex[:n1]
    phi2 = evec_ex[n1:]
    DpL2 = disp_matrix(M, +0.5*Lam)
    DmL2 = disp_matrix(M, -0.5*Lam)
    cat_ex = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    return cat_ex / np.linalg.norm(cat_ex)


# ══════════════════════════════════════════════════════════════════
#  BARRIDO EN Λ (θ = π/4)
# ══════════════════════════════════════════════════════════════════
def cat_sweep(cv, Lam_arr, fix_tau0):
    n = len(Lam_arr)
    out = {k: np.full(n, np.nan) for k in
           ('Lam', 'eta', 'r', 'tau', 'Evar', 'Eex', 'infid_p', 'infid_m',
            'o_ex', 'o_var', 'leak')}
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = _lam12_from_Lam_rho(Lam, RHO)
        M = _Mof_Lam(Lam)
        E, x = minimize_cat_num(Lam, cv, M, fix_tau0, warm)
        warm = x
        Ee, vec, _ = E_cat_exact(lam1, lam2, cv, M=M)
        ov_p, o_ex, o_var = _cat_overlap(lam1, lam2, x, vec, M, sign=+1)
        ov_m, _, _        = _cat_overlap(lam1, lam2, x, vec, M, sign=-1)
        _, _, pu, pd = cat_quantities(x, Lam, cv, get_ops(M), False)
        out['Lam'][k] = Lam
        out['eta'][k], out['r'][k], out['tau'][k] = x
        out['Evar'][k], out['Eex'][k] = E, Ee
        out['infid_p'][k] = 1.0 - ov_p**2
        out['infid_m'][k] = 1.0 - ov_m**2
        out['o_ex'][k], out['o_var'][k] = o_ex, o_var
        out['leak'][k] = max(np.sum(pu[-5:]**2), np.sum(pd[-5:]**2))
    return out


GRIDS = {'params': LAM_PARAMS, 'twin': LAM_INFID_TWIN,
         'single': LAM_INFID, 'cut': LAM_CUT}


def compute_sweeps(fix_tau0):
    _OPS_CACHE.clear()
    keys = [(g, cv) for g in GRIDS for cv in cv_list]
    res  = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(cat_sweep)(cv, GRIDS[g], fix_tau0) for g, cv in keys)
    return dict(zip(keys, res))


# ══════════════════════════════════════════════════════════════════
#  WIGNER (funciones originales, numpy)
# ══════════════════════════════════════════════════════════════════
def wigner_cat(psi, xvec, pvec, verbose=True, n_jobs=-1):
    """W(α) = (2/π) Σ_n (−1)^n |<n|D(−α)|ψ>|², α = x + i p.
    Usa W(x,−p) = W(x,p) (lóbulos en el eje real, coeficientes reales)."""
    N = len(psi)
    lnf = np.zeros(N); lnf[1:] = np.cumsum(np.log(np.arange(1, N)))
    parity = (-1.0) ** np.arange(N)
    mp_idx = np.arange(N)[:, None]
    n_idx  = np.arange(N)[None, :]
    K       = np.abs(mp_idx - n_idx)
    upper   = mp_idx >= n_idx
    deg     = np.where(upper, n_idx, mp_idx)
    lnmin   = np.minimum(lnf[mp_idx], lnf[n_idx])
    lnmax   = np.maximum(lnf[mp_idx], lnf[n_idx])
    sqrtfac = np.exp(0.5 * (lnmin - lnmax))

    def _row(p):
        row = np.zeros(len(xvec))
        for ix, x in enumerate(xvec):
            alpha = -(x + 1j*p)
            ac    = np.conj(alpha)
            abs2  = (alpha * ac).real
            gauss = np.exp(-0.5 * abs2)
            L     = eval_genlaguerre(deg, K, abs2)
            phase = np.where(upper, alpha**K, (-ac)**K)
            D     = gauss * phase * sqrtfac * L
            c     = D @ psi
            row[ix] = (2.0/np.pi) * np.sum(parity * np.abs(c)**2)
        return row

    pvec = np.asarray(pvec)
    idx_pos = np.where(pvec >= 0.0)[0]
    p_pos   = pvec[idx_pos]
    if n_jobs == 1:
        rows_pos = np.array([_row(p) for p in p_pos])
    else:
        rows_pos = np.array(Parallel(n_jobs=n_jobs)(delayed(_row)(p) for p in p_pos))

    W = np.zeros((len(pvec), len(xvec)))
    pos_lookup = {round(pp, 12): rows_pos[k] for k, pp in enumerate(p_pos)}
    for ip, p in enumerate(pvec):
        key = round(abs(p), 12)
        W[ip, :] = pos_lookup[key] if key in pos_lookup else _row(p)
    return W


def wigner_pure_padded(psi, xvec, pvec):
    """W(α) = (2/π) Σ_n (−1)^n |<n|D(−α)|ψ>|², α = x + i p, para un estado puro.
    El estado se rellena con ceros hasta N_pad ≥ (R + √<n>_ψ + 6)², R = máx|α| de la malla,
    para que D(−α) truncado sea exacto en toda la malla aun cuando ψ tiene lóbulos lejos
    del origen. D(−α) ∝ D(−x) D(−i p) (la fase no afecta |·|²); cada factor se exponencia
    vía la diagonalización de su generador."""
    R = np.sqrt(np.max(np.abs(xvec))**2 + np.max(np.abs(pvec))**2)
    n_mean = np.sum(np.arange(len(psi)) * np.abs(psi)**2) / np.sum(np.abs(psi)**2)
    N = max(len(psi), int(np.ceil((R + np.sqrt(n_mean) + 6.0)**2)))
    v = np.zeros(N); v[:len(psi)] = psi
    b  = np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1)
    w1, V1 = eigh(1j*(b.T - b))          # D(x)  = V1 e^{−i x w1} V1†
    w2, V2 = eigh(b.T + b)               # D(iy) = V2 e^{+i y w2} V2ᵀ
    V1h = V1.conj().T
    par = ((-1.0)**np.arange(N))[:, None]
    ph_x = np.exp(1j*np.outer(w1, np.asarray(xvec)))      # D(−x): e^{+i x w1}
    W = np.zeros((len(pvec), len(xvec)))
    for ip, y in enumerate(pvec):
        u  = V2 @ (np.exp(-1j*y*w2) * (V2.T @ v))         # D(−i y)|ψ>
        Dx = V1 @ (ph_x * (V1h @ u)[:, None])             # D(−x) D(−i y)|ψ>, todas las x
        W[ip] = (2.0/np.pi) * np.sum(par * np.abs(Dx)**2, axis=0)
    return W


def wigner_and_cuts_padded(psi, xvec, pvec):
    W = wigner_pure_padded(psi, xvec, pvec)
    ip0 = np.argmin(np.abs(pvec))
    ix0 = np.argmin(np.abs(xvec))
    return W, W[ip0, :], W[:, ix0]


def wigner_and_cuts_num(psi, xvec, pvec, verbose=True):
    W = wigner_cat(psi, xvec, pvec, verbose=verbose)
    ip0 = np.argmin(np.abs(pvec))
    ix0 = np.argmin(np.abs(xvec))
    return W, W[ip0, :], W[:, ix0]


def wigner_negativity(W, xvec, pvec):
    """δ = ½(∫|W| − 1) y ∫W (convención 2/π, coordenadas de α)."""
    norm_check = _trapz(_trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (_trapz(_trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check


def _neg_point(Lam, sign, xvec, pvec, cv_val):
    """δ y ∫W del gato EXACTO (rama sign)."""
    lam1 = lam2 = Lam / np.sqrt(2.0)
    M = _Mof_Lam(Lam)
    _, vec, _ = E_cat_exact(lam1, lam2, cv_val, M=M)
    psi = exact_cat_state(lam1, lam2, vec, M, sign=sign)
    W = wigner_pure_padded(psi, xvec, pvec)
    return wigner_negativity(W, xvec, pvec)


def _neg_point_variational(Lam, sign, xvec, pvec, cv_val, fix_tau0):
    """δ y ∫W del gato VARIACIONAL (rama sign), parámetros recalculados en Λ."""
    M = _Mof_Lam(Lam)
    _, x = minimize_cat_num(Lam, cv_val, M, fix_tau0)
    psi = variational_cat_vector(Lam, x, sign, M)
    W = wigner_pure_padded(psi, xvec, pvec)
    return wigner_negativity(W, xvec, pvec)


def compute_negativity_exact():
    _OPS_CACHE.clear()
    tasks = [(L, s) for L in LAM_NEG_EX for s in (+1, -1)]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_neg_point)(L, s, X_NEG_EX, P_NEG_EX, CV_NEG) for L, s in tasks)
    res = np.array(res).reshape(len(LAM_NEG_EX), 2, 2)
    return dict(Lam=LAM_NEG_EX, dp=res[:, 0, 0], dm=res[:, 1, 0],
                np_=res[:, 0, 1], nm=res[:, 1, 1])


def compute_negativity_var(fix_tau0):
    _OPS_CACHE.clear()
    tasks = [(L, s) for L in LAM_NEG_VAR for s in (+1, -1)]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_neg_point_variational)(L, s, X_NEG_VAR, P_NEG_VAR, CV_NEG, fix_tau0)
        for L, s in tasks)
    res = np.array(res).reshape(len(LAM_NEG_VAR), 2, 2)
    return dict(Lam=LAM_NEG_VAR, dp=res[:, 0, 0], dm=res[:, 1, 0],
                np_=res[:, 0, 1], nm=res[:, 1, 1])


# ══════════════════════════════════════════════════════════════════
#  GATO BIMODAL (dos cavidades) CON TRISQUEEZING
# ══════════════════════════════════════════════════════════════════
def cat_bimodal(lam1, lam2, cv, N, sector, sign, fix_tau0):
    """Gato de dos modos en la base de sitio (vector, idx = m1*N + m2).
    Modo colectivo acoplado (notas): b_e = c â1 + s â2 (par), b_o = c â1 − s â2 (impar),
    c = λ1/Λ, s = λ2/Λ. Lóbulos: D_b(±β) S_b(r) T_b(∓τ)|0,0>, β = (Λ−η)/2,
    es decir, desplazamientos (γ1, ±γ2) = (cβ, ±sβ) en las cavidades.
    Devuelve (psi, (η, r, τ, γ1, γ2))."""
    Lam = np.sqrt(lam1**2 + lam2**2)
    _, x = minimize_cat_num(Lam, cv, _Mof_Lam(Lam), fix_tau0)
    eta, r, tau = x
    beta = 0.5*(Lam - eta)
    c, s = lam1/Lam, lam2/Lam
    sig  = +1.0 if sector == 'even' else -1.0

    a  = sps.diags(np.sqrt(np.arange(1, N, dtype=float)), 1, format='csr')
    I  = sps.identity(N, format='csr')
    a1 = sps.kron(a, I, format='csr')
    a2 = sps.kron(I, a, format='csr')
    b  = c*a1 + sig*s*a2
    bd = b.T.tocsr()
    G1 = (bd - b).tocsr()
    G2 = (0.5*(b @ b - bd @ bd)).tocsr()
    G3 = (bd @ bd @ bd - b @ b @ b).tocsr()

    vac = np.zeros(N*N); vac[0] = 1.0

    def lobe(bb, tt):
        v = vac
        if tt != 0.0:
            v = expm_multiply(tt*G3, v)
        if r != 0.0:
            v = expm_multiply(r*G2, v)
        return expm_multiply(bb*G1, v)

    psi = lobe(+beta, -tau) + sign*lobe(-beta, +tau)
    psi = psi/np.linalg.norm(psi)
    return psi, (eta, r, tau, c*beta, sig*s*beta)


def wigner_joint_cut_exact(psi_ph, nph, uvec, vvec, plane='real'):
    """W(α1, α2) en un corte 2D (Re-Re o Im-Im), vector de dos modos idx = m1*nph + m2."""
    psi_mat = psi_ph.reshape(nph, nph)
    par = (-1.0) ** np.arange(nph)
    Par = np.outer(par, par)

    def _Ddag(val):
        a = (val + 0j) if plane == 'real' else (1j*val)
        return disp_matrix_complex(nph-1, -a)

    Ddag_u  = [_Ddag(u) for u in uvec]
    Ddag_vT = [_Ddag(v).T for v in vvec]
    pref = (2.0/np.pi)**2
    W = np.zeros((len(vvec), len(uvec)))
    for iu in range(len(uvec)):
        left = Ddag_u[iu] @ psi_mat
        for iv in range(len(vvec)):
            tpsi = left @ Ddag_vT[iv]
            W[iv, iu] = pref * np.sum(Par * (np.abs(tpsi)**2))
    return W


# ══════════════════════════════════════════════════════════════════
#  UTILIDADES DE GRÁFICAS (originales)
# ══════════════════════════════════════════════════════════════════
def _nice_tick(m):
    for step in [0.5, 0.25, 0.2, 0.1, 0.05]:
        if step <= m:
            return step
    return np.round(m, 2)


def _fmt_ticks(vals):
    return ['0' if abs(v) < 1e-9 else f'{v:.2f}' for v in vals]


def _lo_hi(cut, m, tol):
    hp = np.max(cut) >  tol
    hn = np.min(cut) < -tol
    if hp and hn: return (-m, m)
    if hp:        return (0.0, m)
    if hn:        return (-m, 0.0)
    return (-m, m)


def _smooth(y, window=7, poly=3):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < window:
        return y
    ys = y.copy()
    if (~m).any():
        idx = np.arange(len(y))
        ys[~m] = np.interp(idx[~m], idx[m], y[m])
    win = min(window, len(ys) - (1 - len(ys) % 2))
    if win < poly + 2:
        return y
    if win % 2 == 0:
        win -= 1
    out = savgol_filter(ys, win, poly)
    out[~m] = np.nan
    return out


def _spline_smooth(x, y, s_factor=1e-4):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return y
    xv, yv = x[m], y[m]
    s = s_factor * len(yv) * np.nanvar(yv)
    spl = UnivariateSpline(xv, yv, k=2, s=s)
    out = np.full_like(y, np.nan)
    out[m] = spl(xv)
    return out


class SciFormatter(ticker.ScalarFormatter):
    """Notación científica con multiplicador común y cifras fijas en los ticks."""
    def __init__(self, fformat='%1.1f'):
        self.fformat = fformat
        super().__init__(useOffset=False, useMathText=True)

    def _set_format(self):
        self.format = r'$%s$' % self.fformat


def plot_wigner_stack_num(Ws, cuts_p0, cuts_x0, xvec, pvec, cmap='RdBu',
                          axis_lim=(-4.2, 4.2), ylim=(-2.1, 2.1),
                          data_aspect=0.5, cut_margin=1.15, panel_frac=0.30,
                          tol=1e-3, labeleven=True, figsize=(6.0, 5.5), suptitle=None,
                          savepath=None, fig=None, label_offset=0):
    re = xvec; im = pvec; n = len(Ws)
    X = axis_lim[1]-axis_lim[0]; Y = ylim[1]-ylim[0]; Fw, Fh = figsize
    vmax = max(np.max(np.abs(W)) for W in Ws)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    wm = 0.62; hm = data_aspect*wm*(Fw/Fh)*(Y/X)
    wr = panel_frac*wm - 0.01; hb = panel_frac*hm + 0.01
    padr, padb = 0.01, 0.016
    Lm, Tm     = 0.10, 0.12
    gap_v      = 0.06
    gap_cb, hcb = 0.1, 0.035
    Hg = hm + padb + hb
    axes = []
    for i in range(n):
        y_main = (1.0 - Tm - hm) - i*(Hg + gap_v)
        ax_main  = fig.add_axes([Lm,         y_main,         wm, hm])
        ax_below = fig.add_axes([Lm,         y_main-padb-hb, wm, hb])
        ax_right = fig.add_axes([Lm+wm+padr, y_main,         wr, hm])
        axes.append((ax_main, ax_below, ax_right))
        ax_main.pcolormesh(re, im, Ws[i], cmap=cmap, norm=norm, shading='auto', zorder=0)
        ax_main.axhline(0, color='k', ls='--', lw=1.3, zorder=1)
        ax_main.axvline(0, color='k', ls='--', lw=1.3, zorder=1)
        ax_main.set_xlim(axis_lim[0], axis_lim[1])
        ax_main.set_ylim(*ylim)
        ax_main.yaxis.set_major_locator(ticker.MultipleLocator(2 if labeleven else 1))
        ax_main.tick_params(axis='both', direction='in', length=3.5, top=True, bottom=True, left=True, right=True)
        ax_main.tick_params(axis='x', labelbottom=False, labeltop=(i == 0))
        for spine in ax_main.spines.values():
            spine.set_zorder(5)
        ax_main.xaxis.set_label_position('top')
        if i == 0:
            ax_main.set_xlabel(r'$\mathrm{Re}(\alpha)$', labelpad=7.5)
        ax_main.set_ylabel(r'$\mathrm{Im}(\alpha)$', labelpad=5)
        ax_main.text(0.015, 0.94, f'({chr(97+label_offset+i)})', transform=ax_main.transAxes,
                     ha='left', va='top', bbox=dict(boxstyle='round,pad=0.12', fc='none', ec='none'))
        m = cut_margin * max(np.max(np.abs(cuts_p0[i])), np.max(np.abs(cuts_x0[i])))
        ax_below.plot(re, cuts_p0[i], 'k', lw=1.2)
        ax_below.set_xlim(axis_lim[1], axis_lim[0])
        lo, hi = _lo_hi(cuts_p0[i], m, tol); ax_below.set_ylim(lo, hi)
        ax_below.set_ylabel(r'$W$', rotation=0, labelpad=14, va='center')
        ax_below.tick_params(axis='x', labelbottom=False, direction='in', top=True, bottom=True, size=3.5)
        ax_below.tick_params(axis='y', direction='in', left=True, right=True, size=3.5)
        ax_right.plot(cuts_x0[i], im, 'k', lw=1.2)
        ax_right.set_ylim(*ylim)
        lo, hi = _lo_hi(cuts_x0[i], m, tol); ax_right.set_xlim(hi, lo)
        ax_right.set_xlabel(r'$W$')
        ax_right.tick_params(axis='y', labelleft=False, direction='in', left=True, right=True, size=3.5, rotation=45)
        ax_right.tick_params(axis='x', direction='in', top=True, bottom=True, size=3.5)
    y_last_below = (1.0 - Tm - hm) - (n-1)*(Hg + gap_v) - padb - hb
    y_cb = y_last_below - gap_cb - hcb
    cax = fig.add_axes([Lm, y_cb, wm, hcb])
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('bottom'); cax.xaxis.set_label_position('bottom')
    cax.text(1.035, 0.5, r'$W$', transform=cax.transAxes, ha='left', va='center')
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', transparent=False, dpi=300)
    return fig, axes


def plot_joint_wigner_2x2(W_re, W_im, uvec, vvec, theta, axis_lim=(-3.5, 3.5), cmap='RdBu',
                          figsize=(9.0, 5.2), n_pts=401, share_scale=True, cut_margin=1.15,
                          show_cut_line=True, cut_size="45%"):
    u_cut = np.linspace(axis_lim[0], axis_lim[1], n_pts)
    v_cut = u_cut*np.tan(theta)

    def _cut(W):
        itp = RegularGridInterpolator((vvec, uvec), W, bounds_error=False, fill_value=np.nan)
        return itp(np.column_stack([v_cut, u_cut]))

    cut_re = _cut(W_re)
    cut_im = _cut(W_im)
    if share_scale:
        vmax = max(np.nanmax(np.abs(W_re)), np.nanmax(np.abs(W_im)))
        vmax_re = vmax_im = vmax
    else:
        vmax_re = np.nanmax(np.abs(W_re)); vmax_im = np.nanmax(np.abs(W_im))

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.5)

    def _block(cell, W, cut, vmax, xlabel, ylabel, show_ylabel_W):
        ax = fig.add_subplot(cell)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        h = ax.pcolormesh(uvec, vvec, W, cmap=cmap, norm=norm, shading='auto')
        if show_cut_line:
            ax.plot(u_cut, v_cut, color='k', ls=':', lw=1.4)
        ax.set_xlim(*axis_lim); ax.set_ylim(*axis_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axisbelow(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelbottom=False, direction='in', bottom=True, top=True, size=5, width=1.5)
        ax.tick_params(axis='y', direction='in', left=True, right=True, size=5, width=1.5)
        div    = make_axes_locatable(ax)
        ax_cut = div.append_axes("bottom", size=cut_size, pad=0.06, sharex=ax)
        ax_cut.plot(u_cut, cut, color='k', lw=1.4)
        m = cut_margin*np.nanmax(np.abs(cut))
        ax_cut.set_ylim(-m, m)
        tv = _nice_tick(m)
        ax_cut.set_yticks([-tv, 0.0, tv])
        ax_cut.set_yticklabels(_fmt_ticks([-tv, 0.0, tv]))
        ax_cut.set_xlabel(xlabel, labelpad=5)
        if show_ylabel_W:
            ax_cut.set_ylabel(r'$W$', rotation=0, labelpad=10, va='center')
        else:
            ax_cut.tick_params(axis='y', labelleft=False)
        ax_cut.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True, size=5, width=1.5)
        return ax, ax_cut, h

    ax_a, ax_c, h_re = _block(gs[0, 0], W_re, cut_re, vmax_re, r'$\mathrm{Re}(\alpha_1)$', r'$\mathrm{Re}(\alpha_2)$', True)
    ax_b, ax_d, h_im = _block(gs[0, 1], W_im, cut_im, vmax_im, r'$\mathrm{Im}(\alpha_1)$', r'$\mathrm{Im}(\alpha_2)$', False)
    if share_scale:
        cax = ax_b.inset_axes([1.10, 0.0, 0.075, 0.8])
        fig.colorbar(h_im, cax=cax)
        cax.set_title(r'$W$', pad=10)
        cax.tick_params(direction='in')
    else:
        for axm, h in [(ax_a, h_re), (ax_b, h_im)]:
            cax = axm.inset_axes([1.10, 0.0, 0.075, 0.8])
            fig.colorbar(h, cax=cax)
            cax.set_title(r'$W$', pad=10)
            cax.tick_params(direction='in')
    return fig, (ax_a, ax_b, ax_c, ax_d)


def _save(fig, name, svg=False):
    fig.savefig(os.path.join(OUT_DIR, name + '.png'), bbox_inches='tight', transparent=False, dpi=300)
    if svg:
        fig.savefig(os.path.join(OUT_DIR, name + '.svg'), bbox_inches='tight', transparent=False)
    plt.close(fig)


def _plot_signed(ax, x, y, sym, ls, color, lw=1.0):
    yy = y[np.isfinite(y)]
    s  = -1.0 if (yy.size and np.mean(yy) < 0) else 1.0
    ax.plot(x, s*y, ls, color=color, lw=lw, label=(rf'$-{sym}$' if s < 0 else rf'${sym}$'))


# ══════════════════════════════════════════════════════════════════
#  FIGURAS
# ══════════════════════════════════════════════════════════════════
C_A = plt.cm.seismic(0.12)    # azul
C_B = plt.cm.seismic(0.88)    # rojo
LS  = {0.5: '-', 1.0: '--', 1.5: ':'}


def fig_params(S, suffix, fix_tau0):
    """Parámetros del gato vs Λ (corte θ = π/4)."""
    xlim_by_cv = {0.5: 4.0, 1.0: 4.5, 1.5: 4.5}
    ylim_by_cv = {0.5: 0.3, 1.0: 0.55, 1.5: 0.7}
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        for cv in cv_list:
            d = S[('params', cv)]
            Lc = d['Lam']
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc,  d['eta']/2, '-',  color=C_B, lw=1.0, label=r'$\eta/2$')
            ax.plot(Lc, -d['r'],     '--', color=C_A, lw=1.0, label=r'$-r$')
            if not fix_tau0:
                _plot_signed(ax, Lc, d['tau'], r'\tau', ':', 'green')
            ax.set_xlim(Lc[0], xlim_by_cv[cv])
            ax.set_ylim(0.0, ylim_by_cv[cv])
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'Variational parameters')
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'cut_cat_params_rho{RHO}_cv{cv}{suffix}')


def fig_infid(S, suffix):
    C_PLUS, C_MINUS = 'black', plt.cm.seismic(0.95)
    Lam_max = 5.0
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        # ── versión con eje doble ──
        fig, ax_o1 = plt.subplots(figsize=(6.5, 5.5))
        ax_e1 = ax_o1.twinx()
        for cv in cv_list:
            d = S[('twin', cv)]
            ax_o1.plot(d['Lam'], d['infid_p'], LS[cv], color=C_PLUS,  lw=1.4)
            ax_e1.plot(d['Lam'], d['infid_m'], LS[cv], color=C_MINUS, lw=1.4)
        Lc = d['Lam']
        ax_o1.set_xlabel(r'$\Lambda$')
        ax_o1.set_ylabel(r'$1-|\langle\mathrm{Cat}_+^{\mathrm{exact}}|\mathrm{Cat}_+^{\mathrm{var}}\rangle|^2$', color=C_PLUS)
        ax_o1.tick_params(axis='y', colors=C_PLUS, direction='in')
        ax_o1.tick_params(axis='x', direction='in')
        ax_o1.set_xlim(Lc[0], Lam_max)
        ax_e1.set_ylabel(r'$1-|\langle\mathrm{Cat}_-^{\mathrm{exact}}|\mathrm{Cat}_-^{\mathrm{var}}\rangle|^2$', color=C_MINUS)
        ax_e1.tick_params(axis='y', colors=C_MINUS, direction='in')
        ax_e1.spines['right'].set_color(C_MINUS)
        ax_e1.spines['left'].set_color(C_PLUS)
        ax_o1.set_xticks(np.arange(0, np.floor(Lc[-1]) + 1, 1))
        handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv], lw=1.4,
                                  label=rf'$J_\alpha/\omega_0={cv}$') for cv in cv_list]
        handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
                    mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
        ax_o1.legend(handles=handles, loc='upper left', ncol=1, frameon=True,
                     edgecolor='black', framealpha=0.0, fancybox=False,
                     borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'cat_infid_1D_twin_rho{RHO}{suffix}')

        # ── versión con un solo eje ──
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        for cv in cv_list:
            d = S[('single', cv)]
            ax.plot(d['Lam'], d['infid_p'], LS[cv], color=C_PLUS,  lw=1.4)
            ax.plot(d['Lam'], d['infid_m'], LS[cv], color=C_MINUS, lw=1.4)
        Lc = d['Lam']
        ax.set_xlabel(r'$\Lambda$')
        ax.set_ylabel(r'$1-\mathcal{F}$')
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
        ax.set_xlim(Lc[0], Lam_max)
        ax.set_xticks(np.arange(0, np.floor(Lc[-1]) + 1, 1))
        ax.legend(handles=handles, loc='upper left', ncol=1, frameon=True,
                  edgecolor='black', framealpha=0.0, fancybox=False,
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'cat_infid_1D_rho{RHO}{suffix}')


def _cut_quantities(d):
    """p±, β̃ (exacto y variacional) a partir de los solapamientos."""
    o_ex, o_var = np.abs(d['o_ex']), np.abs(d['o_var'])
    q = dict(Lc=d['Lam'],
             pp_e=0.5*(1.0 + o_ex), pm_e=0.5*(1.0 - o_ex),
             pp_v=0.5*(1.0 + o_var), pm_v=0.5*(1.0 - o_var),
             th_e=np.sqrt(np.clip(-0.5*np.log(np.clip(o_ex,  1e-300, None)), 0.0, None)),
             th_v=np.sqrt(np.clip(-0.5*np.log(np.clip(o_var, 1e-300, None)), 0.0, None)),
             ov_e=o_ex, ov_v=o_var)
    return q


def fig_prob_theta_overlap(S, suffix):
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        for cv in cv_list:
            q  = _cut_quantities(S[('cut', cv)])
            Lc = q['Lc']

            # (a) probabilidades
            C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
            pp_e = _spline_smooth(Lc, _smooth(q['pp_e'], window=15, poly=3), s_factor=1e-4)
            pm_e = _spline_smooth(Lc, _smooth(q['pm_e'], window=15, poly=3), s_factor=1e-4)
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            ax.plot(Lc, q['pp_v'], '-',  color=C_PLUS,  lw=1.5)
            ax.plot(Lc, pp_e,      '--', color=C_PLUS,  lw=1.5)
            ax.plot(Lc, q['pm_v'], '-',  color=C_MINUS, lw=1.5)
            ax.plot(Lc, pm_e,      '--', color=C_MINUS, lw=1.5)
            ax.set_xlabel(r'$\Lambda$')
            ax.set_ylabel(r'Measurement probability', labelpad=5)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
            ax.set_xlim(Lc[0], 5)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
            ax.text(0.975, 0.85, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax.transAxes, color=C_PLUS,  ha='right', va='center', fontsize=20)
            ax.text(0.975, 0.15, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax.transAxes, color=C_MINUS, ha='right', va='center', fontsize=20)
            leg = [mlines.Line2D([], [], color='k', ls='-',  lw=1.5, label='variational'),
                   mlines.Line2D([], [], color='k', ls='--', lw=1.5, label='exact')]
            ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='center left', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'prob_cat_rho{RHO}_cv{cv}{suffix}')

            # (b) β̃
            C_VAR, C_EX = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
            th_e = _spline_smooth(Lc, _smooth(q['th_e'], window=13, poly=3), s_factor=1e-4)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc, q['th_v'], '-',  color=C_VAR, lw=1.5, label='variational')
            ax.plot(Lc, th_e,      '--', color=C_EX,  lw=1.5, label='exact')
            ax.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\tilde{\beta}$')
            ax.set_xlim(0.0, Lc[-1])
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'theta_tilde_rho{RHO}_cv{cv}{suffix}')

            # (c) solapamientos
            ov_e = _smooth(q['ov_e'], window=39, poly=3)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc, q['ov_v'], '-',  color=C_VAR, lw=1.5, label='variational')
            ax.plot(Lc, ov_e,      '--', color=C_EX,  lw=1.5, label='exact')
            ax.plot(Lc, np.exp(-Lc**2/2), 'k:', lw=1.5, label=r'$e^{-\Lambda^2/2}$')
            ax.set_xlabel(r'$\Lambda$')
            ax.set_ylabel(r'$|\langle\varphi_{b,1}|\hat D(\Lambda)|\varphi_{b,2}\rangle|$')
            ax.set_xlim(Lc[0], Lc[-1]); ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4)
            plt.tight_layout()
            _save(fig, f'overlap_cat_rho{RHO}_cv{cv}{suffix}')


def fig_negativity(NE, NV, suffix):
    C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        # exacto
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NE['Lam'], NE['dp'], '--', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NE['Lam'], NE['dm'], '--', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(NE['Lam'][0], NE['Lam'][-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.set_xticks(np.arange(0, 7, 1))
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'negativity_vs_Lambda{suffix}')

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NE['Lam'], NE['np_'], '--', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NE['Lam'], NE['nm'],  '--', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.axvline(2.45, color='k', ls=':', lw=1.5)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\int W\,\mathrm{d}^2\alpha$')
        ax.set_xlim(NE['Lam'][0], NE['Lam'][-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='best',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'diag_norm_check{suffix}')

        # variacional
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NV['Lam'], NV['dp'], '-', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NV['Lam'], NV['dm'], '-', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(NV['Lam'][0], 7.0)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.set_xticks(np.arange(0, 7, 1))
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'negativity_variational_vs_Lambda{suffix}')

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(NV['Lam'], NV['np_'], '-', color=C_PLUS,  lw=1.5, label=r'$(+)$')
        ax.plot(NV['Lam'], NV['nm'],  '-', color=C_MINUS, lw=1.5, label=r'$(-)$')
        ax.axhline(1.0, color='k', ls='--', lw=1.0)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\int W\,\mathrm{d}^2\alpha$')
        ax.set_xlim(NV['Lam'][0], 7.0)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='best',
                  borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'diag_norm_check_variational{suffix}')

        # comparación
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(NE['Lam'], NE['dp'], '--', color=C_PLUS,  lw=1.25)
        ax.plot(NE['Lam'], NE['dm'], '--', color=C_MINUS, lw=1.25)
        ax.plot(NV['Lam'], NV['dp'], '-',  color=C_PLUS,  lw=1.5)
        ax.plot(NV['Lam'], NV['dm'], '-',  color=C_MINUS, lw=1.5)
        ax.axvline(2*np.sqrt(2), color='k', ls=':', lw=1.5)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(0.01, 7.0)
        ax.set_xticks(np.arange(0, 7, 1))
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        leg = [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$(+)$'),
               mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$(-)$')]
        ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                  loc='lower right', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        _save(fig, f'negativity_exact_vs_variational{suffix}')


def fig_compare(sweeps, negs):
    """Exacto vs D S|0> (τ = 0) vs D S T|0> en la misma figura:
    negatividad δ_W, probabilidades p± y β̃."""
    C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)
    LS_EX, LS_DS, LS_DST = '--', '-', ':'
    style_handles = [mlines.Line2D([], [], color='k', ls=LS_EX,  lw=1.5, label='exact'),
                     mlines.Line2D([], [], color='k', ls=LS_DS,  lw=1.5, label=r'$\hat D\hat S|0\rangle$'),
                     mlines.Line2D([], [], color='k', ls=LS_DST, lw=1.8, label=r'$\hat D\hat S\hat T|0\rangle$')]
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        # ── negatividad ──
        NE, NV0 = negs['tau0']
        _,  NV1 = negs['trisq']
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(NE['Lam'],  NE['dp'],  LS_EX,  color=C_PLUS,  lw=1.25)
        ax.plot(NE['Lam'],  NE['dm'],  LS_EX,  color=C_MINUS, lw=1.25)
        ax.plot(NV0['Lam'], NV0['dp'], LS_DS,  color=C_PLUS,  lw=1.5)
        ax.plot(NV0['Lam'], NV0['dm'], LS_DS,  color=C_MINUS, lw=1.5)
        ax.plot(NV1['Lam'], NV1['dp'], LS_DST, color=C_PLUS,  lw=1.8)
        ax.plot(NV1['Lam'], NV1['dm'], LS_DST, color=C_MINUS, lw=1.8)
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\delta$')
        ax.set_xlim(0.01, 7.0); ax.set_xticks(np.arange(0, 7, 1))
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        leg = style_handles + [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$(+)$'),
                               mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$(-)$')]
        ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                  loc='lower right', borderpad=0.6, handlelength=2.2, labelspacing=0.4, fontsize=16)
        plt.tight_layout()
        _save(fig, 'negativity_exact_DS_DST')

        for cv in cv_list:
            q0 = _cut_quantities(sweeps['tau0'][('cut', cv)])
            q1 = _cut_quantities(sweeps['trisq'][('cut', cv)])
            Lc = q0['Lc']

            # ── probabilidades ──
            pp_e = _spline_smooth(Lc, _smooth(q0['pp_e'], window=15, poly=3), s_factor=1e-4)
            pm_e = _spline_smooth(Lc, _smooth(q0['pm_e'], window=15, poly=3), s_factor=1e-4)
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            ax.plot(Lc, pp_e,       LS_EX,  color=C_PLUS,  lw=1.5)
            ax.plot(Lc, pm_e,       LS_EX,  color=C_MINUS, lw=1.5)
            ax.plot(Lc, q0['pp_v'], LS_DS,  color=C_PLUS,  lw=1.5)
            ax.plot(Lc, q0['pm_v'], LS_DS,  color=C_MINUS, lw=1.5)
            ax.plot(Lc, q1['pp_v'], LS_DST, color=C_PLUS,  lw=1.8)
            ax.plot(Lc, q1['pm_v'], LS_DST, color=C_MINUS, lw=1.8)
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'Measurement probability', labelpad=5)
            ax.set_xlim(Lc[0], 5); ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
            ax.text(0.975, 0.85, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax.transAxes, color=C_PLUS,  ha='right', va='center', fontsize=20)
            ax.text(0.975, 0.15, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax.transAxes, color=C_MINUS, ha='right', va='center', fontsize=20)
            ax.legend(handles=style_handles, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='center left', borderpad=0.6, handlelength=2.2, labelspacing=0.4, fontsize=16)
            plt.tight_layout()
            _save(fig, f'prob_cat_exact_DS_DST_rho{RHO}_cv{cv}')

            # ── β̃ ──
            th_e = _spline_smooth(Lc, _smooth(q0['th_e'], window=13, poly=3), s_factor=1e-4)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot(Lc, th_e,       LS_EX,  color='k',   lw=1.5)
            ax.plot(Lc, q0['th_v'], LS_DS,  color=C_PLUS,  lw=1.5)
            ax.plot(Lc, q1['th_v'], LS_DST, color=C_MINUS, lw=1.8)
            ax.plot(Lc, Lc/2.0, color='gray', ls='-.', lw=1.2)
            ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\tilde{\beta}$')
            ax.set_xlim(0.0, Lc[-1])
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
            leg = [mlines.Line2D([], [], color='k',     ls=LS_EX,  lw=1.5, label='exact'),
                   mlines.Line2D([], [], color=C_PLUS,  ls=LS_DS,  lw=1.5, label=r'$\hat D\hat S|0\rangle$'),
                   mlines.Line2D([], [], color=C_MINUS, ls=LS_DST, lw=1.8, label=r'$\hat D\hat S\hat T|0\rangle$'),
                   mlines.Line2D([], [], color='gray',  ls='-.',   lw=1.2, label=r'$\Lambda/2$')]
            ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      borderpad=0.6, handlelength=2.2, labelspacing=0.4, fontsize=16)
            plt.tight_layout()
            _save(fig, f'theta_tilde_exact_DS_DST_rho{RHO}_cv{cv}')


def fig_2x2(S, NE, NV, suffix):
    cv = CV_2X2
    LAM_XMAX = 5.0
    XTICKS   = np.arange(0, np.floor(LAM_XMAX) + 1, 1)
    C_PLUS, C_MINUS = plt.cm.seismic(0.88), plt.cm.seismic(0.12)

    q  = _cut_quantities(S[('cut', cv)])
    Lc = q['Lc']
    th_e = _spline_smooth(Lc, _smooth(q['th_e'], window=13, poly=3), s_factor=1e-4)
    pp_e = _spline_smooth(Lc, _smooth(q['pp_e'], window=15, poly=3), s_factor=1e-4)
    pm_e = _spline_smooth(Lc, _smooth(q['pm_e'], window=15, poly=3), s_factor=1e-4)

    fig = plt.figure(figsize=(10.0, 8.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_a)

    for cv_i in cv_list:
        ray = S[('twin', cv_i)]
        ax_a.plot(ray['Lam'], ray['infid_p'], LS[cv_i], color=C_PLUS,  lw=1.4)
        ax_a.plot(ray['Lam'], ray['infid_m'], LS[cv_i], color=C_MINUS, lw=1.4)
    ax_a.set_ylabel(r'$1-F$')
    fmt_a = SciFormatter('%1.1f')
    fmt_a.set_powerlimits((0, 0))
    ax_a.yaxis.set_major_formatter(fmt_a)
    ax_a.yaxis.get_offset_text().set_fontsize(22)
    handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv_i], lw=1.4,
                              label=rf'$J_\alpha/\omega_0={cv_i}$') for cv_i in cv_list]
    handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
                mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
    ax_a.legend(handles=handles, loc='upper left', ncol=1, frameon=True, edgecolor='black',
                framealpha=0.0, fancybox=False, borderpad=0.25, handlelength=1.2, labelspacing=0.4, fontsize=16)
    ax_a.text(0.85, 0.95, r'(a)', transform=ax_a.transAxes, ha='left', va='top', fontsize=20)

    ax_b.plot(Lc, q['th_v'], '-',  color='k', lw=1.5)
    ax_b.plot(Lc, th_e,      '--', color='k', lw=1.5)
    ax_b.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
    ax_b.set_ylabel(r'$\tilde{\beta}$')
    ax_b.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                borderpad=0.25, handlelength=1.2, labelspacing=0.4, loc='lower right')
    ax_b.text(0.05, 0.95, r'(b)', transform=ax_b.transAxes, ha='left', va='top', fontsize=22)

    ax_c.plot(Lc, q['pp_v'], '-',  color=C_PLUS,  lw=1.5)
    ax_c.plot(Lc, pp_e,      '--', color=C_PLUS,  lw=1.5)
    ax_c.plot(Lc, q['pm_v'], '-',  color=C_MINUS, lw=1.5)
    ax_c.plot(Lc, pm_e,      '--', color=C_MINUS, lw=1.5)
    ax_c.set_xlabel(r'$\Lambda$')
    ax_c.set_ylabel(r'$p_{\pm}$', labelpad=5)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
    ax_c.text(0.975, 0.8, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax_c.transAxes,
              color=C_PLUS,  ha='right', va='center', fontsize=22)
    ax_c.text(0.975, 0.2, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax_c.transAxes,
              color=C_MINUS, ha='right', va='center', fontsize=22)
    ax_c.text(0.05, 0.925, r'(c)', transform=ax_c.transAxes, ha='left', va='top', fontsize=22)

    ax_d.plot(NE['Lam'], _smooth(NE['dp'], window=13, poly=3), '--', color=C_PLUS,  lw=1.25)
    ax_d.plot(NE['Lam'], _smooth(NE['dm'], window=13, poly=3), '--', color=C_MINUS, lw=1.25)
    ax_d.plot(NV['Lam'], NV['dp'], '-', color=C_PLUS,  lw=1.5)
    ax_d.plot(NV['Lam'], NV['dm'], '-', color=C_MINUS, lw=1.5)
    ax_d.set_xlabel(r'$\Lambda$')
    ax_d.set_ylabel(r'$\delta_W$')
    leg = [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$|\mathrm{Cat}_{+}\rangle$'),
           mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$|\mathrm{Cat}_{-}\rangle$')]
    ax_d.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                loc='lower right', borderpad=0.25, handlelength=1.2, labelspacing=0.4, fontsize=16)
    ax_d.text(0.05, 0.95, r'(d)', transform=ax_d.transAxes, ha='left', va='top', fontsize=22)

    ax_a.set_xlim(0.0, LAM_XMAX)
    ax_a.set_xticks(XTICKS)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                       left=True, right=True, size=5, width=1.5)
    for ax in (ax_a, ax_b):
        ax.tick_params(axis='x', labelbottom=False)
    _save(fig, f'infid_beta_prob_neg_2x2_rho{RHO}_cv{cv}{suffix}', svg=True)


def fig_wigner_stacks(suffix, fix_tau0):
    xvec = pvec = WIG_GRID
    for lam_val, figsize in WIG_POINTS:
        Lam = np.sqrt(2.0)*lam_val
        M   = _Mof_Lam(Lam)
        _, vec, _ = E_cat_exact(lam_val, lam_val, CV_WIG, M=M)
        psi_ex = exact_cat_state(lam_val, lam_val, vec, M, sign=+1)
        W_ex, cp_ex, cx_ex = wigner_and_cuts_num(psi_ex, xvec, pvec, verbose=False)
        _, x = minimize_cat_num(Lam, CV_WIG, M, fix_tau0)
        psi_v = variational_cat_vector(Lam, x, +1, M)
        W_v, cp_v, cx_v = wigner_and_cuts_padded(psi_v, xvec, pvec)
        fig, _ = plot_wigner_stack_num([W_ex, W_v], [cp_ex, cp_v], [cx_ex, cx_v],
                                       xvec=xvec, pvec=pvec, axis_lim=(-4.2, 4.2),
                                       labeleven=True, figsize=figsize)
        _save(fig, f'wigner_cat_exact_vs_var_lam{lam_val}{suffix}')
        print(f'Wigner exacto vs variacional, λ={lam_val}: (η, r, τ) = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})')


def fig_joint_wigner(suffix, fix_tau0):
    lam1 = LAM_JOINT*np.cos(THETA_JOINT)
    lam2 = LAM_JOINT*np.sin(THETA_JOINT)
    psi, pars = cat_bimodal(lam1, lam2, CV_JOINT, N_FOCK_JOINT, SECTOR_JOINT, SIGN_JOINT, fix_tau0)
    print(f'{SECTOR_JOINT} sign={SIGN_JOINT}: η={pars[0]:.4f} r={pars[1]:.4f} τ={pars[2]:.4f} '
          f'γ1={pars[3]:.3f} γ2={pars[4]:.3f}', flush=True)
    W_re = wigner_joint_cut_exact(psi, N_FOCK_JOINT, UV_JOINT, UV_JOINT, plane='real')
    W_im = wigner_joint_cut_exact(psi, N_FOCK_JOINT, UV_JOINT, UV_JOINT, plane='imag')
    fig, _ = plot_joint_wigner_2x2(W_re, W_im, UV_JOINT, UV_JOINT, THETA_JOINT,
                                   axis_lim=(-2.5, 2.5), figsize=(7.0, 7.0), share_scale=True)
    _save(fig, f'wigner_joint_{SECTOR_JOINT}_{"minus" if SIGN_JOINT < 0 else "plus"}_Lam{LAM_JOINT:.3f}{suffix}', svg=True)
    return W_re, W_im


# ══════════════════════════════════════════════════════════════════
#  TESTS IMPRESOS
# ══════════════════════════════════════════════════════════════════
def _hr(title):
    print('\n' + '='*72 + f'\n {title}\n' + '='*72, flush=True)


def tests_pre():
    _hr('TESTS PREVIOS')
    rng = np.random.default_rng(7)
    # C1: funcional numérico (τ=0) vs funcional analítico original
    for Lam in (1.0, 2.5, 4.0, 5.5):
        ops = get_ops(_Mof_Lam(Lam))
        d = []
        for _ in range(100):
            x = np.array([rng.uniform(0.0, Lam), rng.uniform(-0.6, 0.0)])   # rango físico de los óptimos
            d.append(abs(energy_cat_num(x, Lam, 1.0, ops, True) - energy_grad_cat(x, Lam, 1.0)[0]))
        print(f'[C1] Λ={Lam:3.1f}  máx|E_num(τ=0) - E_analítica| = {max(d):.2e}')
    # C2: óptimo numérico (τ=0) vs _minimize_cat original
    for cv in cv_list:
        for Lam in (0.5, 1.5, 2.5, 3.5, 4.5):
            l1, l2 = _lam12_from_Lam_rho(Lam, RHO)
            Ea, xa = _minimize_cat(l1, l2, cv)
            En, xn = minimize_cat_num(Lam, cv, _Mof_Lam(Lam), True)
            print(f'[C2] cv={cv}  Λ={Lam:3.1f}  E_num={En:+.10f}  E_an={Ea:+.10f}  dif={En-Ea:+.2e}'
                  f'  |Δη|={abs(xn[0]-xa[0]):.1e}  |Δr|={abs(xn[1]-xa[1]):.1e}')
    # C3: gato bimodal: centros de los lóbulos y norma (τ=0 y τ≠0)
    lam1 = LAM_JOINT*np.cos(THETA_JOINT); lam2 = LAM_JOINT*np.sin(THETA_JOINT)
    for fix in (True, False):
        psi, pars = cat_bimodal(lam1, lam2, CV_JOINT, 30, 'even', +1, fix)
        print(f'[C3] bimodal (τ{"=0" if fix else " libre"}): norma={np.linalg.norm(psi):.12f}  '
              f'(η,r,τ)=({pars[0]:.4f},{pars[1]:.4f},{pars[2]:.4f})  γ1={pars[3]:.4f}  γ2={pars[4]:.4f}')
    # C4: paridad del gato variacional: <Π> = sign
    for Lam in (1.0, 3.0):
        M = _Mof_Lam(Lam)
        for fix in (True, False):
            _, x = minimize_cat_num(Lam, 1.0, M, fix)
            for sgn in (+1, -1):
                psi = variational_cat_vector(Lam, x, sgn, M)
                par = np.sum(((-1.0)**np.arange(len(psi)))*psi**2)
                print(f'[C4] Λ={Lam:3.1f} τ{"=0" if fix else " libre"} rama {sgn:+d}: <Π> = {par:+.12f}')


def tests_post(sweeps, negs):
    _hr('TESTS POSTERIORES')
    for tag, S in sweeps.items():
        for (g, cv), d in S.items():
            gap = d['Evar'] - d['Eex']
            line = (f'[C5] {tag:5s} malla={g:6s} cv={cv}:  min(E_var - E_ex) = {np.nanmin(gap):+.2e}'
                    f'  violaciones cota = {int(np.sum(gap < -1e-9))}'
                    f'  máx. peso últimos 5 Fock = {np.nanmax(d["leak"]):.1e}')
            if tag == 'trisq':
                line += (f'  máx|τ| = {np.nanmax(np.abs(d["tau"])):.4f}'
                         f'  τ en la cota = {int(np.sum(np.abs(d["tau"]) > TAU_MAX - 1e-6))}')
            print(line)
    if 'tau0' in sweeps and 'trisq' in sweeps:
        for key in sweeps['trisq']:
            diff = sweeps['trisq'][key]['Evar'] - sweeps['tau0'][key]['Evar']
            print(f'[C6] malla={key[0]:6s} cv={key[1]}:  máx(E_DST - E_DS) = {np.nanmax(diff):+.2e}'
                  f'  violaciones = {int(np.sum(diff > 1e-9))}  mejora máx. = {-np.nanmin(diff):.2e}')
    for tag, (NE, NV) in negs.items():
        print(f'[C7] {tag:5s}  máx|∫W - 1|: exacto (+)={np.nanmax(np.abs(NE["np_"]-1)):.2e} '
              f'(−)={np.nanmax(np.abs(NE["nm"]-1)):.2e} | variacional (+)={np.nanmax(np.abs(NV["np_"]-1)):.2e} '
              f'(−)={np.nanmax(np.abs(NV["nm"]-1)):.2e}')


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
_NEG_EXACT = None


def run(fix_tau0):
    global _NEG_EXACT
    tag    = 'tau0' if fix_tau0 else 'trisq'
    suffix = '_tau0' if fix_tau0 else ''
    _hr(f'CORRIDA: {"τ = 0 (prueba)" if fix_tau0 else "trisqueezing"}')

    print('Barridos en Λ ...', flush=True)
    S = compute_sweeps(fix_tau0)
    fig_params(S, suffix, fix_tau0)
    fig_infid(S, suffix)
    fig_prob_theta_overlap(S, suffix)

    print('Negatividad δ_W ...', flush=True)
    if _NEG_EXACT is None:                     # el exacto no depende de τ
        _NEG_EXACT = compute_negativity_exact()
    NV = compute_negativity_var(fix_tau0)
    fig_negativity(_NEG_EXACT, NV, suffix)
    fig_2x2(S, _NEG_EXACT, NV, suffix)

    print('Wigner exacto vs variacional ...', flush=True)
    fig_wigner_stacks(suffix, fix_tau0)

    print('Wigner conjunta de dos modos ...', flush=True)
    fig_joint_wigner(suffix, fix_tau0)

    print(f'Corrida {tag} lista.', flush=True)
    return S, (_NEG_EXACT, NV)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    tests_pre()

    sweeps, negs = {}, {}
    if MODE in ('tau0', 'all'):
        sweeps['tau0'], negs['tau0'] = run(True)
    if MODE in ('trisq', 'all'):
        sweeps['trisq'], negs['trisq'] = run(False)

    if 'tau0' in sweeps and 'trisq' in sweeps:
        _hr('FIGURAS DE COMPARACIÓN: exacto vs D S vs D S T')
        fig_compare(sweeps, negs)

    tests_post(sweeps, negs)
    print(f'\nFiguras guardadas en: {OUT_DIR}')
