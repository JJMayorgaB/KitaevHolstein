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


plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
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
})

# ── Utilidades ────────────────────────────────────────────────────────────────

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


def phonon_weight_CM(vec, m_r, m_c, nph, n_ph):
    """Peso fonónico |<m_r,m_c|psi>|^2 sumado sobre los dos bloques electrónicos.
    Índice en el bloque fonónico: idx = m_r * nph + m_c"""
    idx = m_r * nph + m_c
    return abs(vec[idx])**2 + abs(vec[n_ph + idx])**2


def build_CM_hamiltonians(M, eps1, eps2, t, Delta, omega0, g1, g2):
    """
    Hamiltoniano ANISOTRÓPICO (λ₁≠λ₂) en base rel/CoM |elec_sector, m_r, m_c>.

    OJO: cada sector define su PROPIA rotación rel/CoM (â_o,Â_o para impar;
    â_e,Â_e para par), pero ambas comparten el mismo desplazamiento efectivo
    Λ = √(λ₁²+λ₂²). En su base propia, cada sector tiene la estructura de un
    modo ACOPLADO (relativo para impar, CoM para par) y un modo espectador
    libre, idéntica al caso isotrópico salvo √2·λ → Λ.

    Sector impar (o): |1,0> y |0,1>,  modo relativo m_r acoplado
      H_o = [[ ε̃₁ + Ω,     t·D(Λ)      ],
             [ t·D(Λ)ᵀ,     ε̃₂ + Ω      ]]

    Sector par (e):   |0,0> y |1,1>,  modo CoM m_c acoplado
      H_e = [[ Ω,               Δ·D(Λ)†     ],
             [ Δ·D(Λ),          ε̃₁+ε̃₂ + Ω  ]]

    ε̃ᵢ = εᵢ - gᵢ²/ω₀ ,   Λ = √(λ₁²+λ₂²) ,   λᵢ = gᵢ/ω₀
    D(Λ)  actúa en m_r (impar);  D(Λ)† actúa en m_c (par)
    Ω = (m_r + m_c)·ω₀  (diagonal, incluye AMBOS modos)
    """
    nph  = M + 1
    n_ph = nph * nph

    lam1 = g1 / omega0
    lam2 = g2 / omega0
    Lam  = np.sqrt(lam1**2 + lam2**2)          # desplazamiento efectivo

    eps1_t = eps1 - g1**2 / omega0             # ε̃₁
    eps2_t = eps2 - g2**2 / omega0             # ε̃₂

    D_L = disp_matrix(M, Lam)                  # D(Λ)   (real)
    I   = np.eye(nph)

    D_t_full = np.kron(D_L,   I)               # D(Λ) ⊗ I   → acopla m_r (impar)
    D_D_full = np.kron(I, D_L.T)               # I ⊗ D(Λ)†  → acopla m_c (par)

    Omega = np.array([(mr + mc) * omega0
                      for mr in range(nph) for mc in range(nph)])

    # ── Sector impar (modo relativo acoplado) ──
    H_o = np.zeros((2*n_ph, 2*n_ph))
    H_o[:n_ph, :n_ph] = np.diag(eps1_t + Omega)
    H_o[n_ph:, n_ph:] = np.diag(eps2_t + Omega)
    H_o[:n_ph, n_ph:] = t * D_t_full
    H_o[n_ph:, :n_ph] = t * D_t_full.T

    # ── Sector par (modo CoM acoplado) ──
    H_e = np.zeros((2*n_ph, 2*n_ph))
    H_e[:n_ph, :n_ph] = np.diag(Omega)
    H_e[n_ph:, n_ph:] = np.diag(eps1_t + eps2_t + Omega)
    H_e[:n_ph, n_ph:] = Delta * D_D_full
    H_e[n_ph:, :n_ph] = Delta * D_D_full.T

    return H_e, H_o, nph, n_ph


def diagonalize_CM(M, eps1, eps2, t, Delta, omega0, g1, g2):
    H_e, H_o, nph, n_ph = build_CM_hamiltonians(
        M, eps1, eps2, t, Delta, omega0, g1, g2)
    ev_e, evec_e = eigh(H_e)
    ev_o, evec_o = eigh(H_o)
    return ev_e, evec_e, ev_o, evec_o, nph, n_ph


def compute_point_CM(lam1, lam2, Delta):
    """Un punto del barrido. Usa globales eps_val, t, omega0, M_max,
    phonon_states_keys (como en tu versión isotrópica)."""
    g1 = lam1 * omega0
    g2 = lam2 * omega0
    M_loc = get_M_CM(lam1, lam2, M_max)
    ev_e, evec_e, ev_o, evec_o, nph, n_ph = diagonalize_CM(
        M=M_loc, eps1=eps_val, eps2=eps_val, t=t, Delta=Delta,
        omega0=omega0, g1=g1, g2=g2)
    gs_e = evec_e[:, 0]
    gs_o = evec_o[:, 0]
    p    = ev_e[0] - ev_o[0]   # splitting con signo: E_e - E_o
    w_e  = {(mr, mc): phonon_weight_CM(gs_e, mr, mc, nph, n_ph)
            for (mr, mc) in phonon_states_keys}
    w_o  = {(mr, mc): phonon_weight_CM(gs_o, mr, mc, nph, n_ph)
            for (mr, mc) in phonon_states_keys}
    return p, w_e, w_o

# ══════════════════════════════════════════════════════════════════
#  FUNCIONES VARIACIONALES Y EXACTAS — CASO ANISOTRÓPICO (λ₁≠λ₂)
#
#  Ansatz de 4 parámetros por sector: (α₁,α₂,r₁,r₂) para el impar,
#  (β₁,β₂,ξ₁,ξ₂) para el par. 
#  Desplazamiento efectivo Λ=√(λ₁²+λ₂²); σ=+1 impar, σ=−1 par.
#
#  Globales requeridas: omega0, disp_matrix, squeezed_vacuum, eigh, minimize.
# ══════════════════════════════════════════════════════════════════

def _Lambda(lam1, lam2):
    return np.sqrt(lam1**2 + lam2**2)

def _M_aniso(lam1, lam2):
    """Truncación de Fock adaptada a Λ (el desplazamiento efectivo)."""
    Lam = _Lambda(lam1, lam2)
    return int(min(170, max(40, np.ceil(4.0*Lam*Lam) + 25)))

# ─────────────────────────────────────────────────────────────────
#  FUNCIONAL DE ENERGÍA GENERAL (4 parámetros)
#  E_s con eps=ε, epsp=ε', s=acoplamiento (t o Δ), sigma=±1
# ─────────────────────────────────────────────────────────────────
def energy_grad_general(x, eps, epsp, s, Lam, sigma):
    """E_s(a1,a2,r1,r2) y gradiente analítico. σ=+1 impar, σ=−1 par."""
    a1, a2, r1, r2 = x
    s1, s2 = np.sinh(r1)**2, np.sinh(r2)**2
    P   = 0.5*omega0*(a1*a1 + a2*a2 + s1 + s2)
    dlt = 0.5*(eps - epsp) + 0.5*omega0*(a1*a1 - a2*a2 + s1 - s2)
    rho = r2 - r1
    arg = sigma*Lam + a2 - a1
    Dd  = 1.0 + np.exp(2*rho)
    g   = arg*arg*np.exp(2*r2)/Dd
    F   = np.exp(-g)/np.sqrt(np.cosh(rho))
    R   = np.sqrt(dlt*dlt + s*s*F*F)
    E   = 0.5*(eps + epsp) + P - R

    # gradientes
    dP  = np.array([omega0*a1, omega0*a2,
                    0.5*omega0*np.sinh(2*r1), 0.5*omega0*np.sinh(2*r2)])
    ddl = np.array([omega0*a1, -omega0*a2,
                    0.5*omega0*np.sinh(2*r1), -0.5*omega0*np.sinh(2*r2)])
    # ∂g/∂(a1,a2,r1,r2)
    dg  = np.array([-2*arg*np.exp(2*r2)/Dd,          # ∂/∂a1 (arg tiene −a1)
                     2*arg*np.exp(2*r2)/Dd,          # ∂/∂a2 (arg tiene +a2)
                     2*g*np.exp(2*rho)/Dd,           # ∂/∂r1 (rho=r2−r1 ⇒ ∂Dd/∂r1)
                     2*g - 2*g*np.exp(2*rho)/Dd])    # ∂/∂r2 (e^{2r2} y Dd)
    th  = np.tanh(rho)
    # ∂F/∂x = F·(−∂g/∂x) + F·(½ th)·∂rho/∂x ; rho=r2−r1 ⇒ ∂rho/∂r1=−1,∂rho/∂r2=+1
    dF  = np.array([F*(-dg[0]),
                    F*(-dg[1]),
                    F*(-dg[2] + 0.5*th),
                    F*(-dg[3] - 0.5*th)])
    gr  = dP if R < 1e-300 else dP - (dlt*ddl + s*s*F*dF)/R
    return E, gr

# ─────────────────────────────────────────────────────────────────
#  MINIMIZACIÓN por sector
# ─────────────────────────────────────────────────────────────────
def _minimize_sector(eps, epsp, s, Lam, sigma, x_warm=None):
    h = 0.5*Lam
    seeds = [(0,0,0,0), (-h,h,-0.1,-0.1), (h,-h,-0.1,-0.1), (-h,h,0,0),
             (0,0,-0.1,-0.1), (-Lam,0,-0.1,-0.1), (0,Lam,-0.1,-0.1),
             (h,-h,0,0), (0.5*Lam,-0.5*Lam,-0.2,-0.2)]
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = [(-Lam-1, Lam+1)]*2 + [(-1.5, 1.5)]*2
    lo = [b[0] for b in bnd]; hi = [b[1] for b in bnd]
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0 = np.clip(np.array(sd, float), lo, hi)
        res = minimize(energy_grad_general, x0, args=(eps, epsp, s, Lam, sigma),
                       jac=True, method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 800})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx

def minimize_odd_aniso(lam1, lam2, eps1, eps2, x_warm=None):
    """Sector impar: ε=ε̃₁, ε'=ε̃₂, s=t, σ=+1."""
    Lam  = _Lambda(lam1, lam2)
    e1t  = eps1 - lam1**2 * omega0
    e2t  = eps2 - lam2**2 * omega0
    return _minimize_sector(e1t, e2t, t, Lam, +1.0, x_warm)

def minimize_even_aniso(lam1, lam2, eps1, eps2, Delta, x_warm=None):
    """Sector par: ε=0, ε'=ε̃₁+ε̃₂, s=Δ, σ=−1."""
    Lam  = _Lambda(lam1, lam2)
    e1t  = eps1 - lam1**2 * omega0
    e2t  = eps2 - lam2**2 * omega0
    return _minimize_sector(0.0, e1t + e2t, Delta, Lam, -1.0, x_warm)

# ─────────────────────────────────────────────────────────────────
#  ENERGÍAS EXACTAS por sector (diagonalización rel/CoM, 1 modo acoplado)
# ─────────────────────────────────────────────────────────────────
def E_odd_exact_aniso(lam1, lam2, eps1, eps2, M=None):
    if M is None: M = _M_aniso(lam1, lam2)
    Lam = _Lambda(lam1, lam2)
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    n1  = M + 1
    DL  = disp_matrix(M, Lam)
    n   = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = e1t*np.eye(n1) + n
    H[n1:, n1:] = e2t*np.eye(n1) + n
    H[:n1, n1:] = t*DL;  H[n1:, :n1] = t*DL.T
    return np.linalg.eigvalsh(H)[0]

def E_even_exact_aniso(lam1, lam2, eps1, eps2, Delta, M=None):
    if M is None: M = _M_aniso(lam1, lam2)
    Lam = _Lambda(lam1, lam2)
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    n1  = M + 1
    DL  = disp_matrix(M, Lam)
    n   = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = n
    H[n1:, n1:] = (e1t + e2t)*np.eye(n1) + n
    H[:n1, n1:] = Delta*DL.T;  H[n1:, :n1] = Delta*DL
    return np.linalg.eigvalsh(H)[0]

# ─────────────────────────────────────────────────────────────────
#  FIDELIDADES por sector
# ─────────────────────────────────────────────────────────────────
def _weights_2x2(A, B, C):
    """Pesos (c1,c2) normalizados del menor autovalor de [[A,C],[C,B]]."""
    Emin = 0.5*(A+B) - np.sqrt(0.25*(A-B)**2 + C*C)
    c1, c2 = C, (Emin - A)
    nn = np.hypot(c1, c2)
    return c1/nn, c2/nn

def fidelity_odd_aniso(lam1, lam2, eps1, eps2, x, M=None):
    """|<ψ_exact|ψ_var>|² sector impar. x=(α₁,α₂,r₁,r₂)."""
    if M is None: M = _M_aniso(lam1, lam2)
    Lam = _Lambda(lam1, lam2)
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    n1  = M + 1
    DL  = disp_matrix(M, Lam)
    n   = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = e1t*np.eye(n1) + n
    H[n1:, n1:] = e2t*np.eye(n1) + n
    H[:n1, n1:] = t*DL;  H[n1:, :n1] = t*DL.T
    _, V = eigh(H); psi_ex = V[:, 0]

    a1, a2, r1, r2 = x
    phi1 = disp_matrix(M, a1) @ squeezed_vacuum(M, r1)
    phi2 = disp_matrix(M, a2) @ squeezed_vacuum(M, r2)
    A = e1t + omega0*(a1*a1 + np.sinh(r1)**2)
    B = e2t + omega0*(a2*a2 + np.sinh(r2)**2)
    rho = r2 - r1; arg = Lam + a2 - a1
    F = np.exp(-arg*arg*np.exp(2*r2)/(1+np.exp(2*rho)))/np.sqrt(np.cosh(rho))
    c1, c2 = _weights_2x2(A, B, t*F)
    psi_var = np.concatenate([c1*phi1, c2*phi2]); psi_var /= np.linalg.norm(psi_var)
    return abs(np.vdot(psi_ex, psi_var))**2

def fidelity_even_aniso(lam1, lam2, eps1, eps2, Delta, x, M=None):
    """|<ψ_exact|ψ_var>|² sector par. x=(β₁,β₂,ξ₁,ξ₂)."""
    if M is None: M = _M_aniso(lam1, lam2)
    Lam = _Lambda(lam1, lam2)
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    n1  = M + 1
    DL  = disp_matrix(M, Lam)
    n   = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = n
    H[n1:, n1:] = (e1t + e2t)*np.eye(n1) + n
    H[:n1, n1:] = Delta*DL.T;  H[n1:, :n1] = Delta*DL
    _, V = eigh(H); psi_ex = V[:, 0]

    b1, b2, x1, x2 = x
    phi1 = disp_matrix(M, b1) @ squeezed_vacuum(M, x1)
    phi2 = disp_matrix(M, b2) @ squeezed_vacuum(M, x2)
    A = 0.0        + omega0*(b1*b1 + np.sinh(x1)**2)
    B = (e1t+e2t)  + omega0*(b2*b2 + np.sinh(x2)**2)
    rho = x2 - x1
    arg = -Lam + b2 - b1          # ← corregido (antes: b1 - b2)
    F = np.exp(-arg*arg*np.exp(2*x2)/(1+np.exp(2*rho)))/np.sqrt(np.cosh(rho))
    c1, c2 = _weights_2x2(A, B, Delta*F)
    psi_var = np.concatenate([c1*phi1, c2*phi2]); psi_var /= np.linalg.norm(psi_var)
    return abs(np.vdot(psi_ex, psi_var))**2

    
def squeezed_vacuum(M, r):
    """S(r)|0> en base de Fock, convención S(r)=exp[r/2(â²−â†²)]. Estable vía C(2k,k)."""
    vec  = np.zeros(M + 1)
    pref = 1.0 / np.sqrt(np.cosh(r))
    th   = np.tanh(r)
    for k in range(M // 2 + 1):
        if 2*k <= M:
            vec[2*k] = pref * ((-th)**k) * np.sqrt(float(_C(2*k, k))) / (2.0**k)
    return vec


def _Mof_Lam(Lam):
    return int(min(170, max(50, np.ceil(4.0*Lam*Lam) + 25)))


# ══════════════════════════════════════════════════════════════════
#  FUNCIONAL Y MINIMIZACIÓN DEL GATO (2 params), detuning cero
# ══════════════════════════════════════════════════════════════════
def energy_grad_cat(x, Lam, s):
    """E(a,r) y gradiente. s = t (impar) o Δ (par). Detuning cero."""
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
    """Minimiza E(a,r). Devuelve (E, (a,r))."""
    Lam = np.sqrt(lam1**2 + lam2**2)          # ← CON raíz
    seeds = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = [(-Lam-1, Lam+1), (-1.0, 0.0)]
    lo = [b[0] for b in bnd]; hi = [b[1] for b in bnd]
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0 = np.clip(np.array(sd, float), lo, hi)
        res = minimize(energy_grad_cat, x0, args=(Lam, s), jac=True, method='L-BFGS-B', bounds=bnd, options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


# ══════════════════════════════════════════════════════════════════
#  EXACTO Y FIDELIDAD DEL GATO — ambas ramas ±
# ══════════════════════════════════════════════════════════════════
def E_cat_exact(lam1, lam2, s, M=None):
    """GS exacto en detuning cero. Devuelve (E, evec, M). s = t o Δ."""
    Lam = _Lambda(lam1, lam2)
    if M is None: M = _Mof_Lam(Lam)
    n1 = M + 1
    DL = disp_matrix(M, Lam)
    n  = np.diag(np.arange(n1)*omega0)
    H = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = n
    H[n1:, n1:] = n
    H[:n1, n1:] = s*DL;  H[n1:, :n1] = s*DL.T
    ev, V = eigh(H)
    return ev[0], V[:, 0], M

def _cat_overlap(lam1, lam2, a, r, evec_ex, M, sign=+1):
    """
    (|fid|, o_ex, o_var) para la rama sign del gato.
    Esquema de cutoff escalonado (10 tramos) para o_ex:
      Λ < 1.5            → sin cutoff (estado completo; colas físicas)
      1.5 <= Λ < ... 4.0 → cutoff decreciente de 0.999 a 0.995 en 10 escalones
      Λ >= 4.0           → cutoff 0.995 (ultra-fuerte, corrige saturación)
    Los cortes suavizan la transición del estado completo al recorte agresivo,
    respetando las colas físicas en el intermedio y corrigiendo la saturación
    asintótica del overlap exacto (D(Λ) truncada no es unitaria sobre Fock altos).
    La fidelidad NO se recorta (usa desplazamiento Λ/2, insensible al problema).
    """
    # cortes y cutoffs fijos (10 escalones entre Λ=1.5 y Λ=4.0)
    LAM_CUTS = np.linspace(1.5, 4.0, 20)
    CUTOFFS  = np.linspace(0.999, 0.995, 20)

    Lam = _Lambda(lam1, lam2)
    n1 = M + 1

    # bloques crudos, tal cual salen del vector (peso+signo dentro)
    phi1 = evec_ex[:n1]
    phi2 = evec_ex[n1:]

    sq   = squeezed_vacuum(M, r)
    g_pa = disp_matrix(M, +0.5*a) @ sq     # |r, α/2>
    g_ma = disp_matrix(M, -0.5*a) @ sq     # |r,−α/2>

    DpL2 = disp_matrix(M, +0.5*Lam)
    DmL2 = disp_matrix(M, -0.5*Lam)

    # gato variacional (rama sign)
    cat_var = DpL2 @ g_ma + sign * (DmL2 @ g_pa)
    cat_var /= np.linalg.norm(cat_var)

    # gato exacto (misma rama física): −sign por el signo intrínseco del vector
    cat_ex = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    cat_ex /= np.linalg.norm(cat_ex)

    fid_overlap = np.vdot(cat_var, cat_ex)

    # overlap variacional: desplazamiento absorbido en el bra
    g_ma_L = disp_matrix(M, -0.5*a + Lam) @ sq      # |r,−α/2+Λ>
    o_var  = np.vdot(g_pa, g_ma_L).real             # <r,α/2|D(Λ)|r,−α/2>

    # ── overlap exacto: elegir cutoff según el tramo de Λ ──
    n_passed = int(np.searchsorted(LAM_CUTS, Lam, side='right'))

    if n_passed == 0:
        # Λ por debajo del primer corte: estado completo, sin cutoff
        o_ex = 2.0 * np.vdot(phi1, disp_matrix(M, +Lam) @ phi2).real
    else:
        c = CUTOFFS[min(n_passed - 1, len(CUTOFFS) - 1)]

        def _Mc(v):
            cs = np.cumsum(np.abs(v)**2) / max(np.vdot(v, v).real, 1e-30)
            return int(np.searchsorted(cs, c)) + 1

        Mc = max(_Mc(phi1), _Mc(phi2))
        Mc = min(Mc, n1)

        b1 = phi1[:Mc]
        b2 = phi2[:Mc]
        o_ex = 2.0 * np.vdot(b1, disp_matrix(Mc - 1, +Lam) @ b2).real

    return abs(fid_overlap), o_ex, o_var

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


def exact_cat_state(lam1, lam2, evec_ex, M, sign=+1):
    """Estado gato EXACTO (vector Fock, dim M+1), misma construcción
    que dentro de _cat_overlap."""
    Lam = _Lambda(lam1, lam2)
    n1  = M + 1

    phi1 = evec_ex[:n1]
    phi2 = evec_ex[n1:]

    DpL2 = disp_matrix(M, +0.5*Lam)
    DmL2 = disp_matrix(M, -0.5*Lam)

    cat_ex = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    cat_ex /= np.linalg.norm(cat_ex)
    return cat_ex


def wigner_cat(psi, xvec, pvec, verbose=True, n_jobs=-1):
    """W(alpha) = (2/pi) * sum_n (-1)^n |<n| D(-alpha) |psi>|^2
       Vectorizado + paralelo. Usa simetría vertical W(x,-p)=W(x,p)
       (válida para gatos con lóbulos en el eje real): calcula p>=0 y refleja."""
    N = len(psi)

    # ── precomputar lo independiente de alpha ──
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
    pow_alpha = None  # placeholder

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

    # ── índices de p con p>=0 (mitad a calcular) ──
    pvec = np.asarray(pvec)
    idx_pos = np.where(pvec >= 0.0)[0]      # filas a calcular explícitamente
    p_pos   = pvec[idx_pos]

    # calcular solo la mitad p>=0
    if n_jobs == 1:
        rows_pos = []
        for k, p in enumerate(p_pos):
            rows_pos.append(_row(p))
            if verbose and (k+1) % 10 == 0:
                print(f"   fila {k+1:3d}/{len(p_pos)} (p>=0)  "
                      f"({100.0*(k+1)/len(p_pos):5.1f}%)", flush=True)
        rows_pos = np.array(rows_pos)
    else:
        import joblib
        if verbose:
            print(f"   calculando {len(p_pos)} filas (p>=0) en paralelo ...", flush=True)
        rows_pos = np.array(joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_row)(p) for p in p_pos))
        if verbose:
            print(f"   listo ({len(p_pos)} filas); reflejando a p<0 ...", flush=True)

    # ── ensamblar W completo usando W(x,-p) = W(x,p) ──
    # mapa: para cada p en pvec, buscar su |p| en p_pos
    W = np.zeros((len(pvec), len(xvec)))
    # diccionario valor_p>=0 -> fila calculada
    pos_lookup = {round(pp, 12): rows_pos[k] for k, pp in enumerate(p_pos)}
    for ip, p in enumerate(pvec):
        key = round(abs(p), 12)
        if key in pos_lookup:
            W[ip, :] = pos_lookup[key]
        else:
            # p negativo cuyo |p| no está exacto en la malla: calcular (fallback)
            W[ip, :] = _row(p)
    return W

def wigner_and_cuts_num(psi, xvec, pvec, verbose=True):
    W = wigner_cat(psi, xvec, pvec, verbose=verbose)
    ip0 = np.argmin(np.abs(pvec))
    ix0 = np.argmin(np.abs(xvec))
    cut_p0 = W[ip0, :]
    cut_x0 = W[:, ix0]
    return W, cut_p0, cut_x0


def get_wigner_data_num(states, xvec, pvec):
    """states: lista de vectores de Fock (no Qobj de QuTiP)."""
    Ws, cuts_p0, cuts_x0 = [], [], []
    for psi in states:
        W, cut_p, cut_x = wigner_and_cuts_num(psi, xvec, pvec)
        Ws.append(W)
        cuts_p0.append(cut_p)
        cuts_x0.append(cut_x)
    return Ws, cuts_p0, cuts_x0

def wigner_negativity(W, xvec, pvec):
    """Volumen de negatividad de Kenfack-Zyczkowski:
    delta = 1/2 * (integral |W| d^2alpha - 1) = integral |W_-| d^2alpha
    Requiere que W esté en la convención 2/pi (normalizada a integral=1
    sobre d(Re alpha) d(Im alpha), consistente con wigner_cat)."""
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]

    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)

    return delta, norm_check

def plot_wigner_single(W, cut_reim0, cut_re0im, title,
                       xvec, pvec, cmap='RdBu',
                       axis_lim=(-1.5*np.sqrt(2), 1.5*np.sqrt(2)),
                       cut_margin=1.15, figsize=(8.5, 8.5),
                       cbar_shrink=0.4, cbar_x=-0.12, cbar_w=0.04, labeleven=False):
        
    re = xvec        # ya en convención Re(alpha) directa (sin dividir por sqrt(2))
    im = pvec        # ya en convención Im(alpha) directa

    vmax = np.max(np.abs(W))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax_main = plt.figure(figsize=figsize), None
    ax_main = fig.add_subplot(1, 1, 1)

    # ── mapa central ──────────────────────────────────────────────
    im_h = ax_main.pcolormesh(re, im, W, cmap=cmap, norm=norm, shading='auto')
    ax_main.axhline(0.0, color='k', ls='--', lw=1.5)
    ax_main.axvline(0.0, color='k', ls='--', lw=1.5)
    ax_main.set_xlim(*axis_lim); ax_main.set_ylim(*axis_lim)

    # ── Forzar xticks cada 1 ──────────────────────────────
    ax_main.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if labeleven == True:
        ax_main.yaxis.set_major_locator(ticker.MultipleLocator(2))

    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_axisbelow(False)

    # ── paneles laterales y colorbar anclados al mapa ─────────────
    divider  = make_axes_locatable(ax_main)
    ax_below = divider.append_axes("bottom", size="30%", pad=0.05, sharex=ax_main)
    ax_right = divider.append_axes("right",  size="30%", pad=0.05, sharey=ax_main)

    ax_main.tick_params(axis='x', labelbottom=False, labeltop=True, direction='in', top=True, bottom=True)
    ax_main.tick_params(axis='y', direction='in', left=True, right=True)
    ax_main.text(0.0, 1.05, r'$\mathrm{Re}(\alpha)$', transform=ax_main.transAxes, ha='left', va='bottom')
    ax_main.text(-0.05, 1.0, r'$\mathrm{Im}(\alpha)$', transform=ax_main.transAxes, ha='right', va='top', rotation=90)

    # ── límite simétrico común de los cortes ──────────────────────
    m = cut_margin * max(np.max(np.abs(cut_reim0)), np.max(np.abs(cut_re0im)))

    # ── corte inferior: W(Re α, Im α = 0)  [eje x invertido] ──────
    ax_below.plot(re, cut_reim0, color='k', lw=1.3)
    ax_below.set_xlim(axis_lim[0], axis_lim[1])
    ax_below.invert_xaxis()
    if np.all(cut_reim0 > 0):       # estrictamente positivo
        ax_below.set_ylim(0.0, m)
    else:                           # hay W < 0  o  W = 0
        ax_below.set_ylim(-m, m)
    ax_below.set_ylabel(r'$W$', rotation=0, labelpad=15, va='center')
    ax_below.tick_params(axis='x', labelbottom=False, direction='in', top=True, right=True)
    ax_below.tick_params(axis='y', direction='in', top=True, right=True)

    # ── corte derecho: W(Re α = 0, Im α) ──────────────────────────
    ax_right.plot(cut_re0im, im, color='k', lw=1.3)
    tol = 1e-3   # umbral para ignorar positividad/negatividad espuria
    has_pos = np.max(cut_re0im) >  tol
    has_neg = np.min(cut_re0im) < -tol

    if has_pos and has_neg:          # cambia de signo
        ax_right.set_xlim(m, -m)
    elif has_pos:                    # esencialmente positivo
        ax_right.set_xlim(m, 0.0)    # invertido: +m (izq) … 0 (der)
    elif has_neg:                    # esencialmente negativo
        ax_right.set_xlim(0.0, -m)   # invertido: 0 (izq) … −m (der)
    else:                            # ~0 en todo el corte
        ax_right.set_xlim(m, -m)
    ax_right.set_xlabel(r'$W$')
    ax_right.tick_params(axis='y', labelleft=False, direction='in', top=True, right=True)
    ax_right.tick_params(axis='x', direction='in', top=True, right=True)

    # ── colorbar como inset del mapa: centrada y con "shrink" ─────
    cax = ax_main.inset_axes([cbar_x, (1 - cbar_shrink)/2, cbar_w, cbar_shrink])
    cbar = fig.colorbar(im_h, cax=cax)
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    cax.set_title(r'$W$')
    return fig

    fig, axes = plot_wigner_stack_num(
    [W, W_minus],
    [cut_p0, cut_p0_minus],
    [cut_x0, cut_x0_minus],
    xvec=xvec, pvec=pvec,
    axis_lim=(-4.2, 4.2), labeleven=True,
    suptitle=fr'Gato exacto, $\lambda_1={lam1_val}$, $\lambda_2={lam2_val}$',
    savepath='Resultados/Wigner_cat_pm.png'   # o None si no quieres guardar
)
plt.show()

def plot_wigner_stack_num(Ws, cuts_p0, cuts_x0,
                          xvec, pvec, cmap='RdBu',
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
        ax_main.tick_params(axis='x', labelbottom=False, labeltop=(i==0))
        for spine in ax_main.spines.values():
            spine.set_zorder(5)
        ax_main.xaxis.set_tick_params(zorder=5)
        ax_main.yaxis.set_tick_params(zorder=5)
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
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('bottom'); cax.xaxis.set_label_position('bottom')
    cax.text(1.035, 0.5, r'$W$', transform=cax.transAxes, ha='left', va='center')

    if suptitle:
        fig.suptitle(suptitle, y=1.0)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', transparent=False, dpi=300)

    return fig, axes

def _lo_hi(cut, m, tol):
    hp = np.max(cut) >  tol
    hn = np.min(cut) < -tol
    if hp and hn: return (-m, m)
    if hp:        return (0.0, m)
    if hn:        return (-m, 0.0)
    return (-m, m)

# ══════════════════════════════════════════════════════════════════
#  GATO VARIACIONAL (QuTiP) + negatividad — para comparar con el exacto
# ══════════════════════════════════════════════════════════════════

def coh_squeezed(beta, r, N):
    """|r,β⟩ = D(β) S(r)|0⟩  (qutip ket)."""
    return qt.displace(N, beta) * qt.squeeze(N, r) * qt.basis(N, 0)

def cat_norm(beta, r, sign):
    """N_± = [2(1 ± e^{-2β²e^{2r}})]^{-1/2}."""
    overlap = np.exp(-2.0 * beta**2 * np.exp(2.0*r))
    return 1.0 / np.sqrt(2.0 * (1.0 + sign*overlap))

def cat_state_variational(beta, r, sign, N):
    """|Cat_±⟩ = N_± ( |r,β⟩ ± |r,-β⟩ )  (normalización analítica)."""
    psi = coh_squeezed(beta, r, N) + sign * coh_squeezed(-beta, r, N)
    return cat_norm(beta, r, sign) * psi

def _neg_point(Lam, sign, xvec, pvec, cv_val=1.0):
    """δ y norm_check (∫W) de la Wigner para un Λ y una rama."""
    lam1 = lam2 = Lam / np.sqrt(2.0)
    M = _Mof_Lam(Lam)
    Ee, vec, _ = E_cat_exact(lam1, lam2, cv_val, M=M)

    n1 = M + 1
    phi1 = vec[:n1]; phi2 = vec[n1:]
    DpL2 = disp_matrix(M, +0.5*Lam); DmL2 = disp_matrix(M, -0.5*Lam)
    cat = DpL2 @ phi2 + (-sign) * (DmL2 @ phi1)
    psi = cat / np.linalg.norm(cat)

    W = wigner_cat(psi, xvec, pvec, verbose=False)
    delta, norm_check = wigner_negativity(W, xvec, pvec)
    return delta, norm_check

def _neg_point_variational(Lam, sign, xvec, pvec, N=None, cv_val=1.0):
    """δ y ∫W del gato VARIACIONAL a un Λ (rama sign). Recalcula (α,r) en ρ=1."""
    lam1 = lam2 = Lam / np.sqrt(2.0)
    E, x = _minimize_cat(lam1, lam2, cv_val)      # x = (alpha, r)
    a, r = x[0], x[1]
    theta = 0.5 * (Lam - a)                        # desplazamiento del lóbulo (β)
    if N is None:
        N = _Mof_Lam(Lam)
    psi = cat_state_variational(theta, r, sign, N)
    W = qt.wigner(psi, xvec, pvec)
    return wigner_negativity(W, xvec, pvec)

def cat_bimodal(lam1, lam2, cv, N, sector='even', sign=+1, r_thresh=0.05):
    """
    Estado gato bimodal entrelazado (base de sitio), construido con QuTiP.
      sector='even': lóbulos DIAGONAL   (γ1,γ2) ± (-γ1,-γ2), squeeze two-mode signo −
      sector='odd' : lóbulos ANTIDIAG.  (γ1,-γ2) ± (-γ1,γ2), squeeze two-mode signo +
      sign = ±1: rama del gato.
      r_thresh: si |r| < r_thresh no se aplica S(r) (en strong coupling r≈0).
    Devuelve (psi_qobj, (alpha, r, gam1, gam2)).
    """
    Lam = np.sqrt(lam1**2 + lam2**2)
    E, x = _minimize_cat(lam1, lam2, cv)
    alpha, r = x[0], x[1]
    gam1 = lam1 * (0.5 - alpha/(2.0*Lam))
    gam2 = lam2 * (0.5 - alpha/(2.0*Lam))

    a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
    a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
    vac = qt.tensor(qt.basis(N, 0), qt.basis(N, 0))

    # squeezing generalizado de dos modos (solo si |r| supera el umbral)
    if abs(r) >= r_thresh:
        c1  = (lam1/Lam)**2                 # cos²θ
        c2  = (lam2/Lam)**2                 # sin²θ
        c12 = 2.0*lam1*lam2/Lam**2          # sin2θ
        sgn_tm = +1.0 if sector == 'odd' else -1.0
        G = c1*(a1**2 - a1.dag()**2) + c2*(a2**2 - a2.dag()**2) \
            + sgn_tm * c12 * (a1*a2 - a1.dag()*a2.dag())
        S = ((r/2.0) * G).expm()
        base = S * vac
    else:
        base = vac

    D1 = lambda g: qt.tensor(qt.displace(N, g), qt.qeye(N))
    D2 = lambda g: qt.tensor(qt.qeye(N), qt.displace(N, g))

    if sector == 'even':       # diagonal
        lobeA = D1(+gam1) * D2(+gam2) * base
        lobeB = D1(-gam1) * D2(-gam2) * base
    else:                      # antidiagonal
        lobeA = D1(+gam1) * D2(-gam2) * base
        lobeB = D1(-gam1) * D2(+gam2) * base

    psi = (lobeA + sign * lobeB).unit()
    return psi, (alpha, r, gam1, gam2)

def wigner_joint_cut(psi, N, uvec, vvec, plane='real', verbose=True, tag=''):
    """
    Wigner conjunta de dos modos W(α1,α2) en un corte 2D, optimizada.
      plane='real': Re(α1)=u vs Re(α2)=v   (Im=0)
      plane='imag': Im(α1)=u vs Im(α2)=v   (Re=0)

    Usa W = (2/π)² Σ_{n1,n2} (-1)^{n1+n2} |<n1 n2| D1†(α1)⊗D2†(α2) |ψ>|²
    con ψ como matriz N×N: el desplazamiento factoriza por modo
    (D1† por la izquierda, D2†^T por la derecha), sin trazas N²×N².
    Reporta avance cada ~10% de las columnas si verbose.
    """
    # ψ como matriz [n1, n2]
    psi_mat = psi.full().reshape(N, N)

    # signos de paridad por modo
    par = (-1.0) ** np.arange(N)
    Par = np.outer(par, par)            # (-1)^{n1+n2}

    # precomputar las matrices de desplazamiento D†(α) para cada valor del eje.
    # D†(α) = D(-α). Reusamos: mismos valores en u y v (misma malla en ambos ejes).
    def _Ddag(val):
        a = (val + 0j) if plane == 'real' else (1j*val)
        return qt.displace(N, -a).full()        # D†(α) = D(-α), matriz N×N

    # como uvec y vvec pueden diferir, precomputamos ambos
    Ddag_u = [_Ddag(u) for u in uvec]           # para el modo 1
    Ddag_v = [_Ddag(v) for v in vvec]           # para el modo 2
    # transpuestas de las del modo 2 (para aplicar por la derecha)
    Ddag_vT = [Dv.T for Dv in Ddag_v]

    pref = (2.0/np.pi)**2
    nu = len(uvec)
    step10 = max(1, nu // 10)

    W = np.zeros((len(vvec), len(uvec)))
    for iu in range(nu):
        # aplicar D1†(α1) por la izquierda: (N×N)@(N×N) = N×N
        left = Ddag_u[iu] @ psi_mat             # [n1', n2]
        for iv in range(len(vvec)):
            # aplicar D2†(α2) por la derecha (vía transpuesta): left @ D2†^T
            tpsi = left @ Ddag_vT[iv]           # [n1', n2'] = <n1' n2'|D1†⊗D2†|ψ>
            W[iv, iu] = pref * np.sum(Par * (np.abs(tpsi)**2))
        if verbose and ((iu+1) % step10 == 0 or (iu+1) == nu):
            print(f"      [{tag}] {100.0*(iu+1)/nu:5.1f}%", flush=True)
    return W


def wigner_joint_and_cuts(psi, N, uvec, vvec, plane='real', verbose=True, tag=''):
    """W conjunta + cortes centrales (u=0 y v=0)."""
    W = wigner_joint_cut(psi, N, uvec, vvec, plane=plane, verbose=verbose, tag=tag)
    iv0 = np.argmin(np.abs(vvec))
    iu0 = np.argmin(np.abs(uvec))
    return W, W[iv0, :], W[:, iu0]

def _nice_tick(m):
    """Un valor de tick redondo (0.1, 0.2, 0.25, 0.5...) que quepa en [0, m]."""
    for step in [0.5, 0.25, 0.2, 0.1, 0.05]:
        if step <= m:
            return step
    return np.round(m, 2)

def _fmt_ticks(vals):
    """Etiquetas: '0' para el cero, 2 decimales para el resto."""
    return ['0' if abs(v) < 1e-9 else f'{v:.2f}' for v in vals]

def plot_wigner_single_joint(W, cut_reim0, cut_re0im, xlabel, ylabel,
                             xvec, pvec, cmap='RdBu',
                             axis_lim=(-4.2, 4.2),
                             cut_margin=1.15, figsize=(8.5, 8.5),
                             cbar_shrink=0.35, cbar_x=-0.12, cbar_w=0.04, labeleven=False):
    re = xvec
    im = pvec
    vmax = np.max(np.abs(W))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=figsize)
    ax_main = fig.add_subplot(1, 1, 1)

    im_h = ax_main.pcolormesh(re, im, W, cmap=cmap, norm=norm, shading='auto')
    ax_main.axhline(0.0, color='k', ls='--', lw=1.5)
    ax_main.axvline(0.0, color='k', ls='--', lw=1.5)
    ax_main.set_xlim(*axis_lim); ax_main.set_ylim(*axis_lim)
    ax_main.yaxis.set_major_locator(ticker.MultipleLocator(1))
    if labeleven:
        ax_main.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_main.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_axisbelow(False)

    divider  = make_axes_locatable(ax_main)
    ax_below = divider.append_axes("bottom", size="30%", pad=0.05, sharex=ax_main)
    ax_right = divider.append_axes("right",  size="30%", pad=0.05, sharey=ax_main)

    ax_main.tick_params(axis='x', labelbottom=False, labeltop=True, direction='in', top=True, bottom=True)
    ax_main.tick_params(axis='y', direction='in', left=True, right=True)
    # posiciones ORIGINALES (las que ya te funcionaban), solo parametrizadas
    ax_main.text(-0.025, 1.025, xlabel, transform=ax_main.transAxes, ha='left', va='bottom',fontsize=20)
    ax_main.text(-0.025, 1.025, ylabel, transform=ax_main.transAxes, ha='right', va='top', rotation=90, fontsize=20)

    m = cut_margin * max(np.max(np.abs(cut_reim0)), np.max(np.abs(cut_re0im)))

    tol = 1e-2
    # m independiente por panel (usa el rango real de cada corte)
    m_below = cut_margin * np.max(np.abs(cut_reim0))
    m_right = cut_margin * np.max(np.abs(cut_re0im))

    has_pos1 = np.max(cut_re0im) >  tol
    has_neg1 = np.min(cut_re0im) < -tol
    has_pos2 = np.max(cut_reim0) >  tol
    has_neg2 = np.min(cut_reim0) < -tol

    # panel inferior
    ax_below.plot(re, cut_reim0, color='k', lw=1.3)
    ax_below.set_xlim(axis_lim[0], axis_lim[1])
    #ax_below.invert_xaxis()

    tick_val_b = np.round(0.75 * m_below / cut_margin, 2)

    if has_pos2 and has_neg2:
        ax_below.set_ylim(-m_below, m_below)
        tv = _nice_tick(m_below)
        ax_below.set_yticks([-tv, 0.0, tv])
        ax_below.set_yticklabels(_fmt_ticks([-tv, 0.0, tv]))
    elif has_pos2:
        ax_below.set_ylim(0.0, m_below)
        tv = _nice_tick(m_below)
        ax_below.set_yticks([0.0, tv])
        ax_below.set_yticklabels(_fmt_ticks([0.0, tv]))
    elif has_neg2:
        ax_below.set_ylim(-m_below, 0.0)
        tv = _nice_tick(m_below)
        ax_below.set_yticks([-tv, 0.0])
        ax_below.set_yticklabels(_fmt_ticks([-tv, 0.0]))
    else:
        ax_below.set_ylim(-m_below, m_below)
        tv = _nice_tick(m_below)
        ax_below.set_yticks([-tv, 0.0, tv])
        ax_below.set_yticklabels(_fmt_ticks([-tv, 0.0, tv]))
    ax_below.set_ylabel(r'$W$', rotation=0, labelpad=15, va='center')
    ax_below.tick_params(axis='x', labelbottom=False, direction='in', top=True, right=True)
    ax_below.tick_params(axis='y', direction='in', top=True, right=True)

    # panel derecho
    # panel derecho
    ax_right.plot(cut_re0im, im, color='k', lw=1.3)

    # valor de tick "redondo" dentro del rango (p.ej. 3/4 de m, redondeado)
    tick_val = np.round(0.75 * m_right / cut_margin, 2)   # un tick simétrico legible

    if has_pos1 and has_neg1:
        ax_right.set_xlim(m_right, -m_right)
        tv = _nice_tick(m_right)
        ax_right.set_xticks([tv, 0.0, -tv])
        ax_right.set_xticklabels(_fmt_ticks([tv, 0.0, -tv]))
        ax_right.tick_params(axis='x', labelrotation=45)
    elif has_pos1:
        ax_right.set_xlim(m_right, 0.0)
        tv = _nice_tick(m_right)
        ax_right.set_xticks([tv, 0.0])
        ax_right.set_xticklabels(_fmt_ticks([tv, 0.0]))
    elif has_neg1:
        ax_right.set_xlim(0.0, -m_right)
        tv = _nice_tick(m_right)
        ax_right.set_xticks([0.0, -tv])
        ax_right.set_xticklabels(_fmt_ticks([0.0, -tv]))
    else:
        ax_right.set_xlim(m_right, -m_right)
        tv = _nice_tick(m_right)
        ax_right.set_xticks([tv, 0.0, -tv])
        ax_right.set_xticklabels(_fmt_ticks([tv, 0.0, -tv]))
        ax_right.tick_params(axis='x', labelrotation=45)
    ax_right.set_xlabel(r'$W$')
    ax_right.tick_params(axis='y', labelleft=False, direction='in', top=True, right=True)
    ax_right.tick_params(axis='x', direction='in', top=True, right=True)

    cax = ax_main.inset_axes([cbar_x, (1 - cbar_shrink)/2, cbar_w, cbar_shrink])
    cbar = fig.colorbar(im_h, cax=cax)
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    cax.set_title(r'$W$')
    return fig

def reduced_dm_mode(psi, mode, N):
    """Matriz densidad reducida de un modo (traza el otro).
       mode=1 → ρ1 = Tr_2|ψ><ψ|;  mode=2 → ρ2 = Tr_1|ψ><ψ|.
       psi: Qobj de dos modos (QuTiP)."""
    rho = psi * psi.dag()
    keep = 0 if mode == 1 else 1
    return rho.ptrace(keep)      # QuTiP traza el otro modo


def wigner_negativity_dm(rho1, xvec, pvec):
    """δ y ∫W de la Wigner de un modo de una matriz densidad ρ (QuTiP).
       Usa qt.wigner (acepta matriz densidad directamente)."""
    W = qt.wigner(rho1, xvec, pvec)
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    norm_check = np.trapz(np.trapz(W, xvec, axis=1), pvec)
    delta = 0.5 * (np.trapz(np.trapz(np.abs(W), xvec, axis=1), pvec) - 1.0)
    return delta, norm_check


def marginal_negativities(psi, xvec, pvec):
    """δ1, δ2 (negatividades marginales) del estado bimodal psi (Qobj).
       Reduce cada modo, calcula la Wigner de un modo del reducido, integra."""
    rho1 = reduced_dm_mode(psi, 1, None)
    rho2 = reduced_dm_mode(psi, 2, None)
    d1, n1 = wigner_negativity_dm(rho1, xvec, pvec)
    d2, n2 = wigner_negativity_dm(rho2, xvec, pvec)
    return d1, d2, n1, n2

def entanglement_measures_var(psi, N, tol=1e-12):
    """E_N y S del gato bimodal variacional (Qobj de dos modos, estado PURO).
       Los valores singulares de ψ como matriz N×N son los sqrt(λ_i) de Schmidt:
         E_N = 2 log2( Σ_i s_i ),   S = -Σ_i λ_i log2 λ_i,  λ_i = s_i²."""
    pm = psi.full().reshape(N, N)
    s  = np.linalg.svd(pm, compute_uv=False)
    s  = s[s > np.sqrt(tol)]
    E_N = 2.0*np.log2(np.sum(s))
    lam = s**2
    S   = -np.sum(lam*np.log2(lam))
    return E_N, S


def _cut_along_theta(W, uvec, vvec, theta, s_vec):
    """Perfil de W a lo largo de la recta (u,v) = s·(cosθ, sinθ).
       W está indexada W[iv, iu] → la malla del interpolador es (vvec, uvec)."""
    itp = RegularGridInterpolator((vvec, uvec), W, bounds_error=False, fill_value=np.nan)
    u = s_vec*np.cos(theta)
    v = s_vec*np.sin(theta)
    return itp(np.column_stack([v, u]))


def plot_joint_wigner_2x2(W_re, W_im, uvec, vvec, theta,
                          axis_lim=(-3.5, 3.5), cmap='RdBu',
                          figsize=(9.0, 5.2), n_pts=401,
                          share_scale=True, cut_margin=1.15,
                          show_cut_line=True, cut_size="45%"):
    """
    Dos bloques (Re-Re | Im-Im). Cada bloque: mapa conjunto + corte a lo largo
    de la dirección θ, anclado al mapa con make_axes_locatable (misma anchura,
    mismo eje x). El corte se parametriza por la coordenada horizontal del mapa.
    """

    # corte parametrizado por el eje horizontal u: (u, v) = (u, u·tanθ)
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

        # panel de corte anclado al mapa: misma anchura, mismo eje x
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
    ax_b, ax_d, h_im = _block(gs[0, 1], W_im, cut_im, vmax_im,  r'$\mathrm{Im}(\alpha_1)$', r'$\mathrm{Im}(\alpha_2)$', False)

    if share_scale:
        cax  = ax_b.inset_axes([1.10, 0.0, 0.075, 0.8])
        cbar = fig.colorbar(h_im, cax=cax)
        cax.set_title(r'$W$', pad=10)
        cax.tick_params(direction='in')
    else:
        for axm, h in [(ax_a, h_re), (ax_b, h_im)]:
            cax = axm.inset_axes([1.10, 0.0, 0.075, 0.8])
            cb  = fig.colorbar(h, cax=cax)
            cax.set_title(r'$W$', pad=10)
            cax.tick_params(direction='in')

    return fig, (ax_a, ax_b, ax_c, ax_d)

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


def marginal_negativities_exact(psi_ph, nph, xvec, pvec, n_jobs=-1):
    """δ1, δ2 marginales del estado bimodal exacto (vector de Fock)."""
    rho1 = reduced_dm_mode_exact(psi_ph, nph, 1)
    rho2 = reduced_dm_mode_exact(psi_ph, nph, 2)
    d1, n1 = wigner_dm_negativity(rho1, xvec, pvec, n_jobs=n_jobs)
    d2, n2 = wigner_dm_negativity(rho2, xvec, pvec, n_jobs=n_jobs)
    return d1, d2, n1, n2

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



##########################
# FIGURAS QUE DESEO SACAR
##########################

# ══════════════════════════════════════════════════════════════════
# EJECUCIÓN — GATO (2 params), detuning cero. Dos ramas de fidelidad.
# ══════════════════════════════════════════════════════════════════
omega0   = 1.0
cv_list  = [0.5, 1.0, 1.5]
n_lam    = 300
lam1_arr = np.linspace(0.01, 4.0, n_lam)
lam2_arr = np.linspace(0.01, 4.0, n_lam)

def _cart_point(i2, j1, cv):
    lam1 = lam1_arr[j1]; lam2 = lam2_arr[i2]
    Lam  = np.sqrt(lam1**2 + lam2**2)
    M = _Mof_Lam(Lam)

    E, x = _minimize_cat(lam1, lam2, cv)
    Ee, vec, _ = E_cat_exact(lam1, lam2, cv, M=M)   

    ov_p, o_ex, o_var = _cat_overlap(lam1, lam2, x[0], x[1], vec, M, sign=+1)
    ov_m, _,    _     = _cat_overlap(lam1, lam2, x[0], x[1], vec, M, sign=-1)

    Fp = 1.0 - ov_p**2
    Fm = 1.0 - ov_m**2
    Re = abs(E - Ee)
    return (i2, j1, x[0], x[1], E, Ee, Fp, Fm, Re, o_ex, o_var)

def _blank(): return {cv: np.full((n_lam, n_lam), np.nan) for cv in cv_list}
alpha_m=_blank(); r_m=_blank(); Evar=_blank(); Eex=_blank()
fid_p=_blank(); fid_m=_blank(); rel=_blank()
oex=_blank(); ovar=_blank()

for cv in cv_list:
    print(f"cv={cv}: gato {n_lam}×{n_lam} = {n_lam*n_lam} nodos ...", flush=True)
    args = [(i2, j1, cv) for i2 in range(n_lam) for j1 in range(n_lam)]
    chunks = np.array_split(np.arange(len(args)), 20); done = 0
    with joblib.parallel_backend('loky', n_jobs=-1):
        for chunk_idx in chunks:
            batch = [args[a] for a in chunk_idx]
            res = joblib.Parallel()(joblib.delayed(_cart_point)(*a) for a in batch)
            for (i2,j1,al,rr,Eo,Eeo,Fp,Fm,Re,o_ex,o_var) in res:
                alpha_m[cv][i2,j1]=al; r_m[cv][i2,j1]=rr
                Evar[cv][i2,j1]=Eo; Eex[cv][i2,j1]=Eeo
                fid_p[cv][i2,j1]=Fp; fid_m[cv][i2,j1]=Fm; rel[cv][i2,j1]=Re
                oex[cv][i2,j1]=o_ex; ovar[cv][i2,j1]=o_var
            done += len(chunk_idx)
            print(f"   {100.0*done/len(args):5.1f}%", flush=True)

print("Ejecución gato lista.")

# ══════════════════════════════════════════════════════════════════
#  GUARDAR
# ══════════════════════════════════════════════════════════════════
_SAVE_MAP = [
    (alpha_m, 'alpha'),        (r_m,   'r'),
    (Evar,    'E_var'),        (Eex,   'E_exact'),
    (fid_p,   'fidelity_plus'),(fid_m, 'fidelity_minus'),
    (rel,     'abs_error'),
    (oex,     'overlap_exact'),(ovar,  'overlap_var'),
]

root_label = f"omega0{omega0:.1f}_nl1_{n_lam}_nl2_{n_lam}"
root_dir   = f"cat_state_variational_params_{root_label}"
os.makedirs(root_dir, exist_ok=True)

for cv in cv_list:
    param_label = f"{cv:.1f}_omega0{omega0:.1f}_nl1_{n_lam}_nl2_{n_lam}"
    cv_dir = os.path.join(root_dir, f"cv_{cv}")
    os.makedirs(cv_dir, exist_ok=True)

    hdr_common = (f"cv={cv}, t=Delta={cv}, omega0={omega0}, detuning cero (eps_tilde=0)\n"
                  f"malla 2D [filas=lam2, columnas=lam1], "
                  f"lam1: {n_lam} pts en [{lam1_arr[0]:.3f},{lam1_arr[-1]:.3f}], "
                  f"lam2: {n_lam} pts en [{lam2_arr[0]:.3f},{lam2_arr[-1]:.3f}]")

    np.savetxt(os.path.join(cv_dir, f"coord_lambda1_{param_label}.txt"),
               lam1_arr, fmt='%.8e', header=f"Array lambda1 (columnas): {n_lam} puntos")
    np.savetxt(os.path.join(cv_dir, f"coord_lambda2_{param_label}.txt"),
               lam2_arr, fmt='%.8e', header=f"Array lambda2 (filas): {n_lam} puntos")

    for dct, name in _SAVE_MAP:
        np.savetxt(os.path.join(cv_dir, f"{name}_{param_label}.txt"),
                   dct[cv], fmt='%.8e', header=f"{name}\n{hdr_common}")

    print(f"cv={cv}: guardado en {os.path.abspath(cv_dir)}/", flush=True)

print(f"\nTodos los .txt guardados en {os.path.abspath(root_dir)}/")

extent = [lam1_arr[0], lam1_arr[-1], lam2_arr[0], lam2_arr[-1]]

# entradas: (dict, signo, label, ftag, zoom)   zoom=True → genera zoom extra
param_maps = [(alpha_m, 1, r'$\eta$', 'alpha', False), (r_m, -1, r'$-r$', 'mr', False)]

ZOOM_LIM = (0.0, 1.5)   # límites del zoom en ambos ejes

def _param_heatmaps(param_maps):
    for dct, sgn, lbl, ftag, zoom in param_maps:
        for cv in cv_list:
            data = sgn*dct[cv]; vmax = np.nanmax(data)

            # ── heatmap principal ──
            fig, ax = plt.subplots(figsize=(6, 5.5))
            im = ax.imshow(data, origin='lower', aspect='equal', extent=extent, cmap='inferno', vmin=0.0, vmax=vmax)
            cbar =fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_title(lbl,  pad=10)
            ax.set_xlabel(r'$\lambda_1$'); ax.set_ylabel(r'$\lambda_2$')
            ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True, color='white', size=7.5, width=2.0)
            ax.set_xticks([ 1.0,  2.0,  3.0,  4.0])
            ax.set_yticks([0, 1.0, 2.0,  3.0, 4.0])
            ax.set_title(rf'{lbl},  $J_\alpha /\omega_0={cv}$', pad=15)
            plt.tight_layout()
            plt.savefig(f'Resultados_aniso/heatmap_cat_{ftag}_cv{cv}.png', bbox_inches='tight', transparent=False, dpi=300)
            plt.show()

            # ── heatmap de zoom (opcional), colorbar reescalada a la máscara ──
            if zoom:
                lo, hi = ZOOM_LIM
                mask1 = (lam1_arr >= lo) & (lam1_arr <= hi)
                mask2 = (lam2_arr >= lo) & (lam2_arr <= hi)
                sub = data[np.ix_(mask2, mask1)]           # [i2, j1]
                vmin_z, vmax_z = np.nanmin(sub), np.nanmax(sub)
                extent_z = [lam1_arr[mask1][0], lam1_arr[mask1][-1], lam2_arr[mask2][0], lam2_arr[mask2][-1]]

                fig, ax = plt.subplots(figsize=(6, 5.5))
                im = ax.imshow(sub, origin='lower', aspect='equal', extent=extent_z, cmap='inferno', vmin=vmin_z, vmax=vmax_z)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lbl)
                ax.set_xlabel(r'$\lambda_1$'); ax.set_ylabel(r'$\lambda_2$')
                ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True, color='white')
                ax.set_xticks([0.0, 0.5, 1.0, 1.5]); ax.set_yticks([0.0, 0.5, 1.0, 1.5])
                ax.set_title(rf'{lbl} (zoom),  $J_\alpha /\omega_0={cv}$', pad=15)
                plt.tight_layout()
                plt.savefig(f'Resultados_aniso/heatmap_cat_{ftag}_cv{cv}_zoom.png', bbox_inches='tight', transparent=False, dpi=300)
                plt.show()

_param_heatmaps(param_maps)

C_A = plt.cm.seismic(0.12)   # azul  (squeezing)
C_B = plt.cm.seismic(0.88)   # rojo  (desplazamiento)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

xlim_by_cv = {0.5: 4.0, 1.0: 4.5, 1.5: 4.5}
ylim_by_cv = {0.5: 0.3, 1.0: 0.55, 1.5: 0.7}
rho_list   = [1.0]

n_Lam   = 200
Lam_max = 4.0*np.sqrt(2)

def _lam12_from_Lam_rho(Lam, rho):
    lam2 = Lam / np.sqrt(1.0 + rho**2)
    lam1 = rho * lam2
    return lam1, lam2

def _cat_param_ray(rho, cv):
    """Barrido 1D radial: parámetros de gato (α, r) vs Λ a ρ fijo."""
    Lam_arr = np.linspace(0.01, Lam_max, n_Lam)
    out = {k: np.full(n_Lam, np.nan) for k in ('Lam', 'alpha', 'r')}
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = _lam12_from_Lam_rho(Lam, rho)
        out['Lam'][k] = Lam
        E, x = _minimize_cat(lam1, lam2, s=cv, x_warm=warm); warm = x
        out['alpha'][k], out['r'][k] = x
    return out

# ── Cortes 1D — parámetros de gato (α, r) ──────────────
for rho in rho_list:
    for cv in cv_list:
        print(f'Cut cat params, rho={rho}, cv={cv}', flush=True)
        ray = _cat_param_ray(rho, cv)
        Lc = ray['Lam']
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(Lc,  ray['alpha']/2, '-',  color=C_B, lw=1.0, label=r'$\eta/2$')
        ax.plot(Lc, -ray['r'],       '--', color=C_A, lw=1.0, label=r'$-r$')
        ax.set_xlim(Lc[0], xlim_by_cv[cv])
        ax.set_ylim(0.0, ylim_by_cv[cv])
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'Variational parameters')
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        plt.savefig(f'Resultados_aniso/cut_cat_params_rho{rho}_cv{cv}.png', bbox_inches='tight', transparent=False, dpi=300)
        plt.show()

# ══════════════════════════════════════════════════════════════════
# HEATMAPS de infidelidad (AMBAS RAMAS) y error absoluto del GATO
# ══════════════════════════════════════════════════════════════════
def _single_heatmap(data, label, ftag):
    vmax = np.nanmax(data)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(data, origin='lower', aspect='equal', extent=extent, cmap='inferno', vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_xlabel(r'$\lambda_1$'); ax.set_ylabel(r'$\lambda_2$')
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True, color='white', size=7.5, width=2.0)
    ax.set_xticks([1.0,  2.0,  3.0, 4.0])
    ax.set_yticks([0, 1.0, 2.0,  3.0, 4.0])
    plt.tight_layout()
    plt.savefig(f'Resultados_aniso/heatmap_cat_{ftag}.png', bbox_inches='tight', transparent=False, dpi=300)
    plt.show()

for cv in cv_list:
    print(rf'Cat infidelity (+), $J_\alpha/\omega_0={cv}$')
    _single_heatmap(fid_p[cv], r'$1-|\langle\mathrm{Cat}_+^{\mathrm{exact}}|\mathrm{Cat}_+^{\mathrm{var}}\rangle|^2$', f'infid_plus_cv{cv}')

    print(rf'Cat infidelity (-), $J_\alpha/\omega_0={cv}$')
    _single_heatmap(fid_m[cv], r'$1-|\langle\mathrm{Cat}_-^{\mathrm{exact}}|\mathrm{Cat}_-^{\mathrm{var}}\rangle|^2$', f'infid_minus_cv{cv}')

    print(rf'Absolute error, $J_\alpha/\omega_0={cv}$')
    _single_heatmap(np.abs(Evar[cv] - Eex[cv]), r'$|E_{\mathrm{var}}-E_{\mathrm{exact}}|/\omega_0$', f'abserr_cv{cv}')


def _lam12_from_Lam_rho(Lam, rho):
    lam2 = Lam / np.sqrt(1.0 + rho**2)
    lam1 = rho * lam2
    return lam1, lam2

def _cat_ray(rho, cv):
    """Barrido 1D radial del gato a ρ fijo. Infidelidad de ambas ramas vs Λ."""
    Lam_arr = np.linspace(0.01, Lam_max, n_Lam)
    out = {k: np.full(n_Lam, np.nan) for k in ('Lam', 'infid_p', 'infid_m')}
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = _lam12_from_Lam_rho(Lam, rho)
        M = _Mof_Lam(Lam)
        out['Lam'][k] = Lam
        E, x = _minimize_cat(lam1, lam2, cv, x_warm=warm); warm = x
        Ee, vec, _ = E_cat_exact(lam1, lam2, cv, M=M)
        ov_p, _, _ = _cat_overlap(lam1, lam2, x[0], x[1], vec, M, sign=+1)
        ov_m, _, _ = _cat_overlap(lam1, lam2, x[0], x[1], vec, M, sign=-1)
        out['infid_p'][k] = 1.0 - ov_p**2
        out['infid_m'][k] = 1.0 - ov_m**2
    return out


# ══════════════════════════════════════════════════════════════════
# CORTE λ₂=ρλ₁ del GATO — barrido 1D radial (sin interpolar mapa 2D)
#   Infidelidad de las dos ramas: (+) negro eje izq / (−) rojo eje der
# ══════════════════════════════════════════════════════════════════
LS      = {0.5: '-', 1.0: '--', 1.5: ':'}
C_PLUS  = 'black'               # rama (+)
C_MINUS = plt.cm.seismic(0.95)  # rama (−)

n_Lam   = 200
Lam_max = 5.0


for rho in rho_list:
    fig, ax_o1 = plt.subplots(figsize=(6.5, 5.5))
    ax_e1 = ax_o1.twinx()      # eje derecho: rama (−)

    Lc = None
    for cv in cv_list:
        print(f"  rho={rho}, cv={cv}: calculando rayo ...", flush=True)
        ray = _cat_ray(rho, cv)
        Lc = ray['Lam']
        ax_o1.plot(Lc, ray['infid_p'], LS[cv], color=C_PLUS,  lw=1.4)
        ax_e1.plot(Lc, ray['infid_m'], LS[cv], color=C_MINUS, lw=1.4)

    # ── formato ──
    ax_o1.set_xlabel(r'$\Lambda$')
    ax_o1.set_ylabel(r'$1-|\langle\mathrm{Cat}_+^{\mathrm{exact}}|\mathrm{Cat}_+^{\mathrm{var}}\rangle|^2$',
                     color=C_PLUS)
    ax_o1.tick_params(axis='y', colors=C_PLUS, direction='in')
    ax_o1.tick_params(axis='x', direction='in')
    ax_o1.set_xlim(Lc[0], Lam_max)
    ax_e1.set_ylabel(r'$1-|\langle\mathrm{Cat}_-^{\mathrm{exact}}|\mathrm{Cat}_-^{\mathrm{var}}\rangle|^2$',
                     color=C_MINUS)
    ax_e1.tick_params(axis='y', colors=C_MINUS, direction='in')
    ax_e1.spines['right'].set_color(C_MINUS)
    ax_e1.spines['left'].set_color(C_PLUS)
    ax_o1.set_xticks(np.arange(0, np.floor(Lc[-1]) + 1, 1))

    # ── leyenda dentro del axes, arriba-izquierda, una columna ──
    handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv], lw=1.4,
                              label=rf'$J_\alpha/\omega_0={cv}$') for cv in cv_list]
    handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
                mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
    ax_o1.legend(handles=handles, loc='upper left', ncol=1, frameon=True,
                 edgecolor='black', framealpha=0.0, fancybox=False,
                 borderpad=0.6, handlelength=2.2, labelspacing=0.4)

    plt.tight_layout()
    plt.savefig(f'Resultados_aniso/cat_infid_1D_rho{rho}.png',
                bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    print(f'Cat infidelity (both branches, 1D), rho={rho}')


def _lam12_from_Lam_rho(Lam, rho):
    lam2 = Lam / np.sqrt(1.0 + rho**2)
    lam1 = rho * lam2
    return lam1, lam2

def _cat_ray(rho, cv):
    """Barrido 1D radial del gato a ρ fijo. Infidelidad de ambas ramas vs Λ."""
    Lam_arr = np.linspace(0.01, Lam_max, n_Lam)
    out = {k: np.full(n_Lam, np.nan) for k in ('Lam', 'infid_p', 'infid_m')}
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = _lam12_from_Lam_rho(Lam, rho)
        M = _Mof_Lam(Lam)
        out['Lam'][k] = Lam
        E, x = _minimize_cat(lam1, lam2, cv, x_warm=warm); warm = x
        Ee, vec, _ = E_cat_exact(lam1, lam2, cv, M=M)
        ov_p, _, _ = _cat_overlap(lam1, lam2, x[0], x[1], vec, M, sign=+1)
        ov_m, _, _ = _cat_overlap(lam1, lam2, x[0], x[1], vec, M, sign=-1)
        out['infid_p'][k] = 1.0 - ov_p**2
        out['infid_m'][k] = 1.0 - ov_m**2
    return out

# ══════════════════════════════════════════════════════════════════
# CORTE λ₂=ρλ₁ del GATO — barrido 1D radial (sin interpolar mapa 2D)
#   Infidelidad de las dos ramas, mismo eje/escala. (+) negro, (−) rojo.
# ══════════════════════════════════════════════════════════════════
LS      = {0.5: '-', 1.0: '--', 1.5: ':'}
C_PLUS  = 'black'               # rama (+)
C_MINUS = plt.cm.seismic(0.95)  # rama (−)

rho_list = [1.0]
cv_list = [0.5, 1.0, 1.5]

n_Lam   = 300
Lam_max = 5.0


for rho in rho_list:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    Lc = None
    for cv in cv_list:
        print(f"  rho={rho}, cv={cv}: calculando rayo ...", flush=True)
        ray = _cat_ray(rho, cv)
        Lc = ray['Lam']
        ax.plot(Lc, ray['infid_p'], LS[cv], color=C_PLUS,  lw=1.4)
        ax.plot(Lc, ray['infid_m'], LS[cv], color=C_MINUS, lw=1.4)

    # ── formato ──
    ax.set_xlabel(r'$\Lambda$')
    ax.set_ylabel(r'$1-\mathcal{F}$')
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   right=True, left=True)
    ax.set_xlim(Lc[0], Lam_max)
    ax.set_xticks(np.arange(0, np.floor(Lc[-1]) + 1, 1))

    # ── leyenda dentro del axes, arriba-izquierda, una columna ──
    handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv], lw=1.4,
                              label=rf'$J_\alpha/\omega_0={cv}$') for cv in cv_list]
    handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
                mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
    ax.legend(handles=handles, loc='upper left', ncol=1, frameon=True,
              edgecolor='black', framealpha=0.0, fancybox=False,
              borderpad=0.6, handlelength=2.2, labelspacing=0.4)

    plt.tight_layout()
    plt.savefig(f'Resultados_aniso/cat_infid_1D_rho{rho}.png',
                bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    print(f'Cat infidelity (both branches, 1D), rho={rho}')

def _smooth(y, window=7, poly=3):
    """Savitzky-Golay ligero: quita los saltos del cutoff sin distorsionar la forma.
       Ignora NaN (los rellena por interpolación temporal antes de filtrar)."""
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < window:
        return y
    ys = y.copy()
    # rellenar NaN internos para que savgol no propague
    if (~m).any():
        idx = np.arange(len(y))
        ys[~m] = np.interp(idx[~m], idx[m], y[m])
    win = min(window, len(ys) - (1 - len(ys) % 2))  # impar y ≤ len
    if win < poly + 2:
        return y
    if win % 2 == 0:
        win -= 1
    out = savgol_filter(ys, win, poly)
    out[~m] = np.nan  # restaurar NaN donde no había dato
    return out

def _spline_smooth(x, y, s_factor=1e-4):
    """Spline suave final: pule la curva ya pasada por SG.
       s pequeño → sigue de cerca los datos, solo quita rugosidad residual.
       Ignora NaN."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return y
    xv, yv = x[m], y[m]
    # s escala con el nº de puntos y la varianza → s_factor relativo
    s = s_factor * len(yv) * np.nanvar(yv)
    spl = UnivariateSpline(xv, yv, k=2, s=s)
    out = np.full_like(y, np.nan)
    out[m] = spl(xv)
    return out

# ══════════════════════════════════════════════════════════════════
#  PUNTO 3 — p_±, θ̃ desde overlaps guardados. Exacto vs variacional.
# ══════════════════════════════════════════════════════════════════
omega0   = 1.0
n_lam    = 300
cv_list  = [0.5, 1.0, 1.5]
rho_list = [1.0]

root_dir = f"cat_state_variational_params_omega0{omega0:.1f}_nl1_{n_lam}_nl2_{n_lam}"

def _load(cv, name):
    pl = f"{cv:.1f}_omega0{omega0:.1f}_nl1_{n_lam}_nl2_{n_lam}"
    return np.loadtxt(os.path.join(root_dir, f"cv_{cv}", f"{name}_{pl}.txt"))

lam1_arr = _load(cv_list[0], "coord_lambda1")
lam2_arr = _load(cv_list[0], "coord_lambda2")
extent   = [lam1_arr[0], lam1_arr[-1], lam2_arr[0], lam2_arr[-1]]

# ── derivar p_± y θ̃ (exacto Y variacional) ──────────────────────
p_plus_ex = {}; p_minus_ex = {}
p_plus_var = {}; p_minus_var = {}
theta_ex = {}; theta_var = {}
for cv in cv_list:
    o_ex  = np.abs(_load(cv, "overlap_exact"))   # exacto sale negativo → |·|
    o_var = np.abs(_load(cv, "overlap_var"))
    p_plus_ex[cv]   = 0.5*(1.0 + o_ex)
    p_minus_ex[cv]  = 0.5*(1.0 - o_ex)
    p_plus_var[cv]  = 0.5*(1.0 + o_var)
    p_minus_var[cv] = 0.5*(1.0 - o_var)
    theta_ex[cv]  = np.sqrt(np.clip(-0.5*np.log(np.clip(o_ex,  1e-300, None)), 0.0, None))
    theta_var[cv] = np.sqrt(np.clip(-0.5*np.log(np.clip(o_var, 1e-300, None)), 0.0, None))

# ══════════════════════════════════════════════════════════════════
#  HEATMAPS (inferno) — paneles 1×3 por cv
# ══════════════════════════════════════════════════════════════════
def _row_heatmap(data_by_cv, label, ftag, limits=False, col_ticks='white', sigma=0.0):
    vmax = max(np.nanmax(data_by_cv[cv]) for cv in cv_list)
    if limits:
        vmin = 0.5
    else: vmin = 0.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for idx, cv in enumerate(cv_list):
        ax = axes[idx]
        data = data_by_cv[cv]
        if sigma and sigma > 0:                       
            data = gaussian_filter(data, sigma=sigma, mode='nearest')
        im = ax.imshow(data, origin='lower', aspect='equal', extent=extent, cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'$\lambda_1$')
        ax.tick_params(axis='x', direction='in', bottom=True, top=True, right=True, left=True, color=col_ticks, size=7.5, width=2.0)
        ax.tick_params(axis='y', direction='in', right=True, left=True, color=col_ticks, size=7.5, width=2.0, labelleft=(idx == 0))
        if idx == 0:
            ax.set_ylabel(r'$\lambda_2$')
        ax.set_title(rf'$J_\alpha /\omega_0={cv}$', pad=12)
        ax.set_xticks([0.0, 1.0,  2.0, 3.0,  4.0])
    cbar =fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.025, shrink=0.6)
    cbar.ax.set_title(label, pad=10)
    plt.savefig(f'Resultados_aniso/heatmap_cat_{ftag}.png', bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    print(f'Heatmap panel: {ftag}')

_row_heatmap(p_plus_ex,  r'$p_+$',            'pplus_exact',  True,  sigma=4.0)
_row_heatmap(p_minus_ex, r'$p_-$',            'pminus_exact', False, col_ticks='black', sigma=4.0)
_row_heatmap(theta_ex,   r'$\tilde{\beta}$', 'theta_exact',  False, sigma=4.0)
_row_heatmap(p_plus_var,  r'$p_+$',            'pplus_var',  True)
_row_heatmap(p_minus_var, r'$p_-$',            'pminus_var', False, col_ticks='black')
_row_heatmap(theta_var,   r'$\tilde{\beta}$', 'theta_var',  False)

# ══════════════════════════════════════════════════════════════════
#  CORTES λ₂=ρλ₁
# ══════════════════════════════════════════════════════════════════
def _cut(data2d, rho, n_cut=300):
    itp = RegularGridInterpolator((lam2_arr, lam1_arr), data2d, bounds_error=False, fill_value=np.nan)
    l1max = min(lam1_arr[-1], lam2_arr[-1]/rho)
    l1 = np.linspace(max(lam1_arr[0], 0.01), l1max, n_cut)
    l2 = rho*l1
    Lam = np.sqrt(l1**2 + l2**2)
    return Lam, itp(np.column_stack([l2, l1]))

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ── (a) PROBABILIDADES ──────────────────────────────────────────
C_PLUS  = plt.cm.seismic(0.88)
C_MINUS = plt.cm.seismic(0.12)

for rho in rho_list:
    for cv in cv_list:
        Lc, pp_e = _cut(p_plus_ex[cv],   rho)
        _,  pm_e = _cut(p_minus_ex[cv],  rho)
        _,  pp_v = _cut(p_plus_var[cv],  rho)
        _,  pm_v = _cut(p_minus_var[cv], rho)

        pp_e = _smooth(pp_e, window=15, poly=3)           
        pp_e = _spline_smooth(Lc, pp_e, s_factor=1e-4)

        pm_e = _smooth(pm_e, window=15, poly=3)           
        pm_e = _spline_smooth(Lc, pm_e, s_factor=1e-4)

        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(Lc, pp_v, '-',  color=C_PLUS,  lw=1.5)
        ax.plot(Lc, pp_e, '--', color=C_PLUS,  lw=1.5)
        ax.plot(Lc, pm_v, '-',  color=C_MINUS, lw=1.5)
        ax.plot(Lc, pm_e, '--', color=C_MINUS, lw=1.5)

        ax.set_xlabel(r'$\Lambda$')
        ax.set_ylabel(r'Measurement probability', labelpad=5)
        ax.set_xlim(Lc[0], Lc[-1]); ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.set_xlim(Lc[0], 5)
        ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
        ax.text(0.975, 0.85, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax.transAxes, color=C_PLUS,  ha='right', va='center', fontsize=20)
        ax.text(0.975, 0.15, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax.transAxes, color=C_MINUS, ha='right', va='center', fontsize=20)
        leg = [mlines.Line2D([], [], color='k', ls='-',  lw=1.5, label='variational'), mlines.Line2D([], [], color='k', ls='--', lw=1.5, label='exact')]
        ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='center left', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        plt.savefig(f'Resultados_aniso/prob_cat_rho{rho}_cv{cv}.png', bbox_inches='tight', transparent=False, dpi=300)
        plt.show()
        print(f'Probabilities (var vs exact), rho={rho}, cv={cv}')

# ── (b) θ̃ ──────────────────────────────────────────────────────
C_VAR = plt.cm.seismic(0.88)
C_EX  = plt.cm.seismic(0.12)

for rho in rho_list:
    for cv in cv_list:
        Lc, th_v = _cut(theta_var[cv], rho)
        _,  th_e = _cut(theta_ex[cv],  rho)

        th_e = _smooth(th_e, window=13, poly=3)           
        th_e = _spline_smooth(Lc, th_e, s_factor=1e-4)

        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(Lc, th_v, '-',  color=C_VAR, lw=1.5, label='variational')
        ax.plot(Lc, th_e, '--', color=C_EX,  lw=1.5, label='exact')
        ax.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
        ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'$\tilde{\beta}$')
        ax.set_xlim(0.0, Lc[-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        plt.savefig(f'Resultados_aniso/theta_tilde_rho{rho}_cv{cv}.png', bbox_inches='tight', transparent=False, dpi=300)
        plt.show()
        print(f'theta_tilde (var vs exact), rho={rho}, cv={cv}')

C_VAR = plt.cm.seismic(0.88)
C_EX  = plt.cm.seismic(0.12)

for rho in rho_list:
    for cv in cv_list:
        Lc, ov_v = _cut(np.abs(_load(cv, "overlap_var")),  rho)
        _,  ov_e = _cut(np.abs(_load(cv, "overlap_exact")), rho)

        ov_e = _smooth(ov_e, window=39, poly=3)           
        #ov_e = _spline_smooth(Lc, ov_e, s_factor=1e-4)

        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(Lc, ov_v, '-',  color=C_VAR, lw=1.5, label='variational')
        ax.plot(Lc, ov_e, '--', color=C_EX,  lw=1.5, label='exact')
        ax.plot(Lc, np.exp(-Lc**2/2), 'k:', lw=1.5, label=r'$e^{-\Lambda^2/2}$')
        ax.set_xlabel(r'$\Lambda$')
        ax.set_ylabel(r'$|\langle\varphi_{b,1}|\hat D(\Lambda)|\varphi_{b,2}\rangle|$')
        ax.set_xlim(Lc[0], Lc[-1]); ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
        ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, borderpad=0.6, handlelength=2.2, labelspacing=0.4)
        plt.tight_layout()
        plt.savefig(f'Resultados_aniso/overlap_cat_rho{rho}_cv{cv}.png', bbox_inches='tight', transparent=False, dpi=300)
        plt.show()
        print(f'Overlaps (var vs exact), rho={rho}, cv={cv}')

# ══════════════════════════════════════════════════════════════════
#  FIGURA 2x2 — (a) 1-F | (b) β̃ | (c) probabilidades | (d) δ(Λ)
#  eje x (Λ) compartido; etiquetas solo en la fila inferior
# ══════════════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec

cv  = 1.0
rho = 1.0

LS       = {0.5: '-', 1.0: '--', 1.5: ':'}
n_Lam    = 200
Lam_max  = 5.0
LAM_XMAX = 5.0                              # ← rango común del eje Λ
XTICKS   = np.arange(0, np.floor(LAM_XMAX) + 1, 1)

C_PLUS  = plt.cm.seismic(0.88)
C_MINUS = plt.cm.seismic(0.12)

class SciFormatter(ticker.ScalarFormatter):
    """Notación científica con multiplicador común y cifras fijas en los ticks."""
    def __init__(self, fformat='%1.1f'):
        self.fformat = fformat
        super().__init__(useOffset=False, useMathText=True)
    def _set_format(self):
        self.format = r'$%s$' % self.fformat

# ── datos panel (a): infidelidad (todas las cv) ──
rays = {}
for cv_i in cv_list:
    print(f"  rho={rho}, cv={cv_i}: calculando rayo ...", flush=True)
    rays[cv_i] = _cat_ray(rho, cv_i)

# ── datos panel (b): β̃ ──
Lc, th_v = _cut(theta_var[cv], rho)
_,  th_e = _cut(theta_ex[cv],  rho)
th_e = _smooth(th_e, window=13, poly=3)
th_e = _spline_smooth(Lc, th_e, s_factor=1e-4)

# ── datos panel (c): probabilidades ──
_, pp_e = _cut(p_plus_ex[cv],   rho)
_, pm_e = _cut(p_minus_ex[cv],  rho)
_, pp_v = _cut(p_plus_var[cv],  rho)
_, pm_v = _cut(p_minus_var[cv], rho)
pp_e = _smooth(pp_e, window=15, poly=3)
pp_e = _spline_smooth(Lc, pp_e, s_factor=1e-4)
pm_e = _smooth(pm_e, window=15, poly=3)
pm_e = _spline_smooth(Lc, pm_e, s_factor=1e-4)

# ── datos panel (d): negatividad δ(Λ) ──
_ex = np.loadtxt('Resultados_aniso/negativity_vs_Lambda.txt')
_va = np.loadtxt('Resultados_aniso/negativity_variational_vs_Lambda.txt')
Lam_ex, dp_ex, dm_ex = _ex[:, 0], _ex[:, 1], _ex[:, 2]
Lam_va, dp_va, dm_va = _va[:, 0], _va[:, 1], _va[:, 2]

# ── figura ──
fig = plt.figure(figsize=(10.0, 8.0))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.28)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a)
ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_d = fig.add_subplot(gs[1, 1], sharex=ax_a)

# ---------- panel (a): infidelidad ----------
for cv_i in cv_list:
    ray = rays[cv_i]
    ax_a.plot(ray['Lam'], ray['infid_p'], LS[cv_i], color=C_PLUS,  lw=1.4)
    ax_a.plot(ray['Lam'], ray['infid_m'], LS[cv_i], color=C_MINUS, lw=1.4)
ax_a.set_ylabel(r'$1-F$')
fmt_a = SciFormatter('%1.1f')
fmt_a.set_powerlimits((0, 0))            # fuerza siempre el multiplicador 10^n
ax_a.yaxis.set_major_formatter(fmt_a)
ax_a.yaxis.get_offset_text().set_fontsize(22)
handles  = [mlines.Line2D([], [], color='gray', ls=LS[cv_i], lw=1.4,
                          label=rf'$J_\alpha/\omega_0={cv_i}$') for cv_i in cv_list]
handles += [mlines.Line2D([], [], color=C_PLUS,  lw=1.4, label=r'$|\mathrm{Cat}_+\rangle$'),
            mlines.Line2D([], [], color=C_MINUS, lw=1.4, label=r'$|\mathrm{Cat}_-\rangle$')]
ax_a.legend(handles=handles, loc='upper left', ncol=1, frameon=True, edgecolor='black',
            framealpha=0.0, fancybox=False, borderpad=0.25, handlelength=1.2, labelspacing=0.4, fontsize=16)
ax_a.text(0.85, 0.95, r'(a)', transform=ax_a.transAxes, ha='left', va='top', fontsize=20)

# ---------- panel (b): β̃ ----------
ax_b.plot(Lc, th_v, '-',  color='k', lw=1.5)
ax_b.plot(Lc, th_e, '--', color='k', lw=1.5)
ax_b.plot(Lc, Lc/2.0, 'k:', lw=1.5, label=r'$\Lambda/2$')
ax_b.set_ylabel(r'$\tilde{\beta}$')
ax_b.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
            borderpad=0.25, handlelength=1.2, labelspacing=0.4, loc='lower right')
ax_b.text(0.05, 0.95, r'(b)', transform=ax_b.transAxes, ha='left', va='top', fontsize=22)

# ---------- panel (c): probabilidades ----------
ax_c.plot(Lc, pp_v, '-',  color=C_PLUS,  lw=1.5)
ax_c.plot(Lc, pp_e, '--', color=C_PLUS,  lw=1.5)
ax_c.plot(Lc, pm_v, '-',  color=C_MINUS, lw=1.5)
ax_c.plot(Lc, pm_e, '--', color=C_MINUS, lw=1.5)
ax_c.set_xlabel(r'$\Lambda$')
ax_c.set_ylabel(r'$p_{\pm}$', labelpad=5)
ax_c.set_ylim(-0.02, 1.02)
ax_c.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))
ax_c.text(0.975, 0.8, r'$\dfrac{1}{4\,|\mathcal{N}_+|^{2}}$', transform=ax_c.transAxes,
          color=C_PLUS,  ha='right', va='center', fontsize=22)
ax_c.text(0.975, 0.2, r'$\dfrac{1}{4\,|\mathcal{N}_-|^{2}}$', transform=ax_c.transAxes,
          color=C_MINUS, ha='right', va='center', fontsize=22)
ax_c.text(0.05, 0.925, r'(c)', transform=ax_c.transAxes, ha='left', va='top', fontsize=22)

# ---------- panel (d): negatividad δ(Λ) ----------
ax_d.plot(Lam_ex, _smooth(dp_ex, window=13, poly=3), '--', color=C_PLUS,  lw=1.25)
ax_d.plot(Lam_ex, _smooth(dm_ex, window=13, poly=3), '--', color=C_MINUS, lw=1.25)
ax_d.plot(Lam_va, dp_va, '-', color=C_PLUS,  lw=1.5)
ax_d.plot(Lam_va, dm_va, '-', color=C_MINUS, lw=1.5)
ax_d.set_xlabel(r'$\Lambda$')
ax_d.set_ylabel(r'$\delta_W$')
leg = [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$|\mathrm{Cat}_{+}\rangle$'),
       mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$|\mathrm{Cat}_{-}\rangle$')]
ax_d.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
            loc='lower right', borderpad=0.25, handlelength=1.2, labelspacing=0.4, fontsize=16)
ax_d.text(0.05, 0.95, r'(d)', transform=ax_d.transAxes, ha='left', va='top', fontsize=22)

# ── eje x común: rango, ticks y etiquetas solo abajo ──
ax_a.set_xlim(0.0, LAM_XMAX)
ax_a.set_xticks(XTICKS)
for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=5, width=1.5)
for ax in (ax_a, ax_b):
    ax.tick_params(axis='x', labelbottom=False)

fig.savefig(f'paper_figures/infid_beta_prob_neg_2x2_rho{rho}_cv{cv}.png', bbox_inches='tight', transparent=False, dpi=300)
fig.savefig(f'paper_figures/infid_beta_prob_neg_2x2_rho{rho}_cv{cv}.svg', bbox_inches='tight', transparent=False)
plt.show()
print(f'Figura 2x2 (a) 1-F + (b) β̃ + (c) probabilidades + (d) δ, rho={rho}')

# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN — curva de negatividad δ(Λ) para ambas ramas (ρ=1)
# ══════════════════════════════════════════════════════════════════
omega0   = 1.0
cv_val   = 1.0
nLam     = 200
Lam_arr  = np.linspace(0.01, 7.0, nLam)

# malla de Wigner 
xvec = np.linspace(-5.0, 5.0, 301)
pvec = np.linspace(-2.2, 2.2, 301)

delta_p = np.full(nLam, np.nan); norm_p = np.full(nLam, np.nan)
delta_m = np.full(nLam, np.nan); norm_m = np.full(nLam, np.nan)

_step10 = max(1, nLam // 10)
for k, Lam in enumerate(Lam_arr):
    delta_p[k], norm_p[k] = _neg_point(Lam, +1, xvec, pvec, cv_val=cv_val)
    delta_m[k], norm_m[k] = _neg_point(Lam, -1, xvec, pvec, cv_val=cv_val)
    if (k+1) % _step10 == 0 or (k+1) == nLam:
        print(f"  {100.0*(k+1)/nLam:5.1f}%  Λ={Lam:5.3f}  "
              f"δ(+)={delta_p[k]:.4f} (∫W={norm_p[k]:.3f}) | "
              f"δ(-)={delta_m[k]:.4f} (∫W={norm_m[k]:.3f})", flush=True)

print("Barrido de negatividad listo.")

os.makedirs('Resultados_aniso', exist_ok=True)
_out = np.column_stack([Lam_arr, delta_p, delta_m, norm_p, norm_m])
np.savetxt('Resultados_aniso/negativity_vs_Lambda.txt', _out, fmt='%.8e',  header='Lambda  delta_plus  delta_minus  norm_plus  norm_minus')
print('Guardado: Resultados_aniso/negativity_vs_Lambda.txt')


# ══════════════════════════════════════════════════════════════════
#  NEGATIVIDAD δ(Λ) — ambas ramas
# ══════════════════════════════════════════════════════════════════
C_PLUS  = plt.cm.seismic(0.88)
C_MINUS = plt.cm.seismic(0.12)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(Lam_arr, delta_p, '--', color=C_PLUS,  lw=1.5, ms=3, label=r'$(+)$')
ax.plot(Lam_arr, delta_m, '--', color=C_MINUS, lw=1.5, ms=3, label=r'$(-)$')
ax.set_xlabel(r'$\Lambda$')
ax.set_ylabel(r'$\delta$')
ax.set_xlim(Lam_arr[0], Lam_arr[-1])
ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax.set_xticks(np.arange(0, 7, 1))
ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
plt.tight_layout()
plt.savefig('Resultados_aniso/negativity_vs_Lambda.png', bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print('Curva de negatividad guardada.')

# ══════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO — normalización ∫W dα (debe ser 1 en todo Λ)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(Lam_arr, norm_p, '--', color=C_PLUS,  lw=1.5, ms=3, label=r'$(+)$')
ax.plot(Lam_arr, norm_m, '--', color=C_MINUS, lw=1.5, ms=3, label=r'$(-)$')
ax.axvline(2.45, color='k', ls=':', lw=1.5)
ax.set_xlabel(r'$\Lambda$')
ax.set_ylabel(r'$\int W\,\mathrm{d}^2\alpha$')
ax.set_xlim(Lam_arr[0], Lam_arr[-1])
ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='best', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
plt.tight_layout()
plt.savefig('Resultados_aniso/diag_norm_check.png', bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print('Diagnóstico de normalización guardado.')

# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN — negatividad VARIACIONAL δ(Λ), ambas ramas (ρ=1)
#    Recalcula α, r con _minimize_cat en cada Λ ∈ [0.01, 8.0]
# ══════════════════════════════════════════════════════════════════
omega0  = 1.0
cv_val  = 1.0
nLam     = 300
Lam_arr  = np.linspace(0.01, 7.0, nLam)

# malla de Wigner 
xvec = np.linspace(-9.0, 9.0, 601)
pvec = np.linspace(-3.0, 3.0, 301)

delta_p_v = np.full(nLam, np.nan); norm_p_v = np.full(nLam, np.nan)
delta_m_v = np.full(nLam, np.nan); norm_m_v = np.full(nLam, np.nan)


_step10 = max(1, nLam // 10)
for k, Lam in enumerate(Lam_arr):
    delta_p_v[k], norm_p_v[k] = _neg_point_variational(Lam, +1, xvec, pvec, cv_val=cv_val)
    delta_m_v[k], norm_m_v[k] = _neg_point_variational(Lam, -1, xvec, pvec)
    if (k+1) % _step10 == 0 or (k+1) == nLam:
        print(f"  {100.0*(k+1)/nLam:5.1f}%  Λ={Lam:5.3f}  "
              f"δ_v(+)={delta_p_v[k]:.4f} (∫W={norm_p_v[k]:.3f}) | "
              f"δ_v(-)={delta_m_v[k]:.4f} (∫W={norm_m_v[k]:.3f})", flush=True)

print("Barrido variacional listo.")

os.makedirs('Resultados_aniso', exist_ok=True)
_out = np.column_stack([Lam_arr, delta_p_v, delta_m_v, norm_p_v, norm_m_v])
np.savetxt('Resultados_aniso/negativity_variational_vs_Lambda.txt', _out, fmt='%.8e', header='Lambda  delta_plus_var  delta_minus_var  norm_plus_var  norm_minus_var')
print('Guardado: negativity_variational_vs_Lambda.txt')

# ══════════════════════════════════════════════════════════════════
#  NEGATIVIDAD VARIACIONAL δ(Λ) — ambas ramas
# ══════════════════════════════════════════════════════════════════
C_PLUS  = plt.cm.seismic(0.88)
C_MINUS = plt.cm.seismic(0.12)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(Lam_arr, delta_p_v, '-', color=C_PLUS,  lw=1.5, ms=3, label=r'$(+)$')
ax.plot(Lam_arr, delta_m_v, '-', color=C_MINUS, lw=1.5, ms=3, label=r'$(-)$')
ax.set_xlabel(r'$\Lambda$')
ax.set_ylabel(r'$\delta$')
ax.set_xlim(Lam_arr[0], 7.0)
ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax.set_xticks(np.arange(0, 7, 1))
ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
plt.tight_layout()
plt.savefig('Resultados_aniso/negativity_variational_vs_Lambda.png', bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print('Curva de negatividad variacional guardada.')

# ══════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO VARIACIONAL — normalización ∫W dα (debe ser 1)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(Lam_arr, norm_p_v, '-', color=C_PLUS,  lw=1.5, ms=3, label=r'$(+)$')
ax.plot(Lam_arr, norm_m_v, '-', color=C_MINUS, lw=1.5, ms=3, label=r'$(-)$')
ax.axhline(1.0, color='k', ls='--', lw=1.0)
ax.set_xlabel(r'$\Lambda$')
ax.set_ylabel(r'$\int W\,\mathrm{d}^2\alpha$')
ax.set_xlim(Lam_arr[0], 7.0)
ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='best', borderpad=0.6, handlelength=2.2, labelspacing=0.4)
plt.tight_layout()
plt.savefig('Resultados_aniso/diag_norm_check_variational.png', bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print('Diagnóstico de normalización variacional guardado.')

# ══════════════════════════════════════════════════════════════════
#  NEGATIVIDAD δ(Λ) — exacto (dashed) vs variacional (solid), ambas ramas
# ══════════════════════════════════════════════════════════════════
# cargar los dos archivos
_ex = np.loadtxt('Resultados_aniso/negativity_vs_Lambda.txt')
_va = np.loadtxt('Resultados_aniso/negativity_variational_vs_Lambda.txt')

Lam_ex, dp_ex, dm_ex = _ex[:, 0], _ex[:, 1], _ex[:, 2]
Lam_va, dp_va, dm_va = _va[:, 0], _va[:, 1], _va[:, 2]

C_PLUS  = plt.cm.seismic(0.88)   # rojo  → rama +
C_MINUS = plt.cm.seismic(0.12)   # azul  → rama −
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# variacional: línea sólida
# exacto: línea dashed
ax.plot(Lam_ex, dp_ex, '--', color=C_PLUS,  lw=1.25)
ax.plot(Lam_ex, dm_ex, '--', color=C_MINUS, lw=1.25)
ax.plot(Lam_va, dp_va, '-', color=C_PLUS,  lw=1.5)
ax.plot(Lam_va, dm_va, '-', color=C_MINUS, lw=1.5)
ax.axvline(2*np.sqrt(2), color='k', ls=':', lw=1.5)

ax.set_xlabel(r'$\Lambda$')
ax.set_ylabel(r'$\delta$')
ax.set_xlim(0.01, 7.0)
ax.set_xticks(np.arange(0, 7, 1))
ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)

leg = [mlines.Line2D([], [], color=C_PLUS,  lw=1.5, label=r'$(+)$'),
       mlines.Line2D([], [], color=C_MINUS, lw=1.5, label=r'$(-)$')]
ax.legend(handles=leg, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False, loc='lower right', borderpad=0.6, handlelength=2.2, labelspacing=0.4)

plt.tight_layout()
plt.savefig('Resultados_aniso/negativity_exact_vs_variational.png', bbox_inches='tight', transparent=False, dpi=300)
plt.show()
print('Comparación negatividad exacto vs variacional guardada.')

# ── parámetros del punto a evaluar ──
lam1_val = 3.0
lam2_val = 3.0
omega0 = 1.0
cv_val   = 1.0          # s = t (impar) o Δ (par), según qué sector uses
sign_val = +1           # rama del gato: +1 o -1

# ── diagonalización exacta y estado gato ──
Lam = np.sqrt(lam1_val**2 + lam2_val**2)
M   = _Mof_Lam(Lam)

Ee, vec, _ = E_cat_exact(lam1_val, lam2_val, cv_val, M=M)
cat_state  = exact_cat_state(lam1_val, lam2_val, vec, M, sign=sign_val)

# ── grilla en el plano Re(alpha) vs Im(alpha) ──
xvec = np.linspace(-4.2, 4.2, 301)
pvec = np.linspace(-4.2, 4.2, 301)

# ── Wigner numérica + cortes ──
W_exact1, cut_p0_exact1, cut_x0_exact1 = wigner_and_cuts_num(cat_state, xvec=xvec, pvec=pvec, verbose=True)


# ── minimización variacional: (alpha, r) que minimizan E(a,r) ──
lam1_val = 3.0
lam2_val = 3.0

Ev, xv = _minimize_cat(lam1_val, lam2_val, cv_val)
a_val, r_val = xv
theta_val = 0.5 * (Lam - a_val)     # desplazamiento del lóbulo, β

# mismo cutoff de Fock que el caso exacto, para comparar en igualdad de condiciones
N_var = M

cat_state_var = cat_state_variational(theta_val, r_val, sign_val, N_var)

# ── Wigner (QuTiP) + cortes, mismo formato que el bloque exacto ──
W_var = qt.wigner(cat_state_var, xvec, pvec, g=2)

ip0 = np.argmin(np.abs(pvec))
ix0 = np.argmin(np.abs(xvec))
cut_p0_var = W_var[ip0, :]
cut_x0_var = W_var[:, ix0]


fig, axes = plot_wigner_stack_num(
    [W_exact1, W_var],
    [cut_p0_exact1, cut_p0_var],
    [cut_x0_exact1, cut_x0_var],
    xvec=xvec, pvec=pvec,
    axis_lim=(-4.2, 4.2), labeleven=True, figsize=(7.0, 7.0)
)
plt.show()

# ── parámetros del punto a evaluar ──
lam1_val = 2.0
lam2_val = 2.0
omega0 = 1.0
cv_val   = 1.0          # s = t (impar) o Δ (par), según qué sector uses
sign_val = +1           # rama del gato: +1 o -1

# ── diagonalización exacta y estado gato ──
Lam = np.sqrt(lam1_val**2 + lam2_val**2)
M   = _Mof_Lam(Lam)

Ee, vec, _ = E_cat_exact(lam1_val, lam2_val, cv_val, M=M)
cat_state  = exact_cat_state(lam1_val, lam2_val, vec, M, sign=sign_val)

# ── grilla en el plano Re(alpha) vs Im(alpha) ──
xvec = np.linspace(-4.2, 4.2, 301)
pvec = np.linspace(-4.2, 4.2, 301)

# ── Wigner numérica + cortes ──
W_exact2, cut_p0_exact2, cut_x0_exact2 = wigner_and_cuts_num(cat_state, xvec=xvec, pvec=pvec, verbose=True)

# ── minimización variacional: (alpha, r) que minimizan E(a,r) ──
lam1_val = 2.0
lam2_val = 2.0
Ev, xv = _minimize_cat(lam1_val, lam2_val, cv_val)
a_val, r_val = xv
theta_val = 0.5 * (Lam - a_val)     # desplazamiento del lóbulo, β

# mismo cutoff de Fock que el caso exacto, para comparar en igualdad de condiciones
N_var = M

cat_state_var = cat_state_variational(theta_val, r_val, sign_val, N_var)

# ── Wigner (QuTiP) + cortes, mismo formato que el bloque exacto ──
W_var2 = qt.wigner(cat_state_var, xvec, pvec, g=2)

ip0 = np.argmin(np.abs(pvec))
ix0 = np.argmin(np.abs(xvec))
cut_p0_var2 = W_var2[ip0, :]
cut_x0_var2 = W_var2[:, ix0]

fig, axes = plot_wigner_stack_num(
    [W_exact2, W_var2],
    [cut_p0_exact2, cut_p0_var2],
    [cut_x0_exact2, cut_x0_var2],
    xvec=xvec, pvec=pvec,
    axis_lim=(-4.2, 4.2), labeleven=True, figsize=(9.0, 9.0)
)
plt.show()

fig, axes = plot_wigner_stack_num(
    [W_exact2, W_var2],
    [cut_p0_exact2, cut_p0_var2],
    [cut_x0_exact2, cut_x0_var2],
    xvec=xvec, pvec=pvec,
    axis_lim=(-4.2, 4.2), labeleven=True, figsize=(9.0, 9.0)
)
plt.show()

# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN + FIGURA — gato bimodal par (−), Λ=3√2, cortes Re-Re e Im-Im
# ══════════════════════════════════════════════════════════════════
omega0   = 1.0
cv_val   = 1.0
Lam      = 3*np.sqrt(2)
theta    = np.pi/4
lam1_v   = Lam*np.cos(theta)
lam2_v   = Lam*np.sin(theta)
N_FOCK   = 50
r_thresh = 0.05

uvec = np.linspace(-3.5, 3.5, 201)
vvec = np.linspace(-3.5, 3.5, 201)

psi, pars = cat_bimodal(lam1_v, lam2_v, cv_val, N_FOCK,
                        sector='even', sign=-1, r_thresh=r_thresh)
print(f"even sign=-1: norma={psi.norm():.6f}, "
      f"α={pars[0]:.4f} r={pars[1]:.4f} γ1={pars[2]:.3f} γ2={pars[3]:.3f}", flush=True)

W_re, _, _ = wigner_joint_and_cuts(psi, N_FOCK, uvec, vvec, plane='real', tag='even− Re')
W_im, _, _ = wigner_joint_and_cuts(psi, N_FOCK, uvec, vvec, plane='imag', tag='even− Im')



fig, axes = plot_joint_wigner_2x2(W_re, W_im, uvec, vvec, theta, axis_lim=(-2.5, 2.5), figsize=(7.0, 7.0), share_scale=True)
fig.savefig(f'paper_figures/wigner_joint_even_minus_Lam{Lam:.3f}.png',  bbox_inches='tight', transparent=False, dpi=300)
fig.savefig(f'paper_figures/wigner_joint_even_minus_Lam{Lam:.3f}.svg',  bbox_inches='tight')
plt.show()
print('Figura Wigner conjunta (par −): mapas Re-Re / Im-Im + cortes en θ.')