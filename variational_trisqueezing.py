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


#Grafica no.1 paridad fermionica en el plano Delta vs lambda
# ─── parámetros del sistema ──────────────────────────────────────
t        = 1.0
omega0   = 1.0
eps1_val = 0.25
eps2_val = 0.25
_S2      = np.sqrt(2.0)

# ─── malla (λ/t, Δ/t) ────────────────────────────────────────────
n_lam, n_Delta = 200, 200
lam_arr   = np.linspace(0.0, 1.5, n_lam)     # λ/t  (λ₁=λ₂=λ)
Delta_arr = np.linspace(0.0, 1.5, n_Delta)   # Δ/t

# ─── E_o(λ): sector impar NO depende de Δ (1D, warm start) ───────
E_odd_1d = np.empty(n_lam)
warm = None
for k, lam in enumerate(lam_arr):
    E, x = minimize_odd_aniso(lam, lam, eps1_val, eps2_val, x_warm=warm)
    warm = x
    E_odd_1d[k] = E
print("E_o (1D) listo.")

# ─── E_e(λ,Δ): barrido 2D, una fila por Δ con continuación en λ ──
def _even_row(Delta):
    row = np.empty(n_lam); w = None
    for k, lam in enumerate(lam_arr):
        E, x = minimize_even_aniso(lam, lam, eps1_val, eps2_val, Delta, x_warm=w)
        w = x
        row[k] = E
    return row

print(f"Grid {n_lam}×{n_Delta}: calculando E_e ...")
E_even_2d = np.empty((n_Delta, n_lam))
chunks = np.array_split(np.arange(n_Delta), 20)
done = 0
with joblib.parallel_backend('loky', n_jobs=-1):
    for idx in chunks:
        res = joblib.Parallel()(joblib.delayed(_even_row)(Delta_arr[i]) for i in idx)
        for i, row in zip(idx, res):
            E_even_2d[i] = row
        done += len(idx)
        print(f"  E_e: {100.0*done/n_Delta:5.1f}% completado")

# ─── splitting δE = E_e − E_o ────────────────────────────────────
dE = E_even_2d - E_odd_1d[None, :]
print("δE: min=%.4f  max=%.4f" % (dE.min(), dE.max()))

# ══════════════════════════════════════════════════════════════════
#  GRAFICACIÓN
# ══════════════════════════════════════════════════════════════════
extent    = [lam_arr[0], lam_arr[-1], Delta_arr[0], Delta_arr[-1]]
xlabel    = r'$g/t$'
ylabel    = r'$\Delta/t$'
param_str = rf"$t={t:.1f},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.4f},\ \omega_0={omega0}$"
norm_p    = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(np.tanh(dE), origin='lower', aspect='auto',
               extent=extent, cmap='seismic', norm=norm_p)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\tanh(\delta E)$')

ax.contour(lam_arr, Delta_arr, dE, levels=[0.0], colors='k',
           linewidths=2.0, alpha=1.0, linestyles='--')
ax.set(xlabel=xlabel, ylabel=ylabel)

import matplotlib.patheffects as pe

ax.text(1.1, 1.1, r'\begin{center} Even GS \\ $|\psi_e\rangle$ \end{center}',
        fontsize=20, color='0.15', ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',
                  pad=5, boxstyle='round,pad=0.25'))

ax.text(0.4, 0.4, r'\begin{center} Odd GS \\ $|\psi_o\rangle$ \end{center}',
        fontsize=20, color='black', ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.25, edgecolor='none',
                  pad=5, boxstyle='round,pad=0.25'))

ax.text(0.15, 0.95, '(a)', transform=ax.transAxes, ha='right', va='top',
        fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('Resultados_aniso/Parity_Splitting_aniso.png',
            bbox_inches='tight', transparent=False, dpi=300)
plt.show()

print(r'Parity Splitting $\delta E = E^e_0 - E^o_0$  (anisotrópico)')
print(param_str)

#Grafica 2: Parametros variacionales

# ══════════════════════════════════════════════════════════════════
# CELDA DE EJECUCIÓN — caso ANISOTRÓPICO, eje = Λ, anisotropía δλ fija
# ══════════════════════════════════════════════════════════════════
omega0   = 1.0
eps1_val = 0.25
eps2_val = 0.25
_S2      = np.sqrt(2.0)

cv_list = [0.5, 1.0, 1.5]       

# eje de control: Λ = √(λ₁²+λ₂²)
n_Lam   = 200
Lam_arr = np.linspace(0.01, 4.0*np.sqrt(2), n_Lam)


# diccionarios indexados por δλ ------------------------------------
a1_o = {}; a2_o = {}; r1_o = {}; r2_o = {}; Evar_o = {}; Eex_o = {}; fid_o = {}; rel_o = {}
a1_e = {}; a2_e = {}; r1_e = {}; r2_e = {}; Evar_e = {}; Eex_e = {}; fid_e = {}; rel_e = {}
lam1_rec = {}; lam2_rec = {}      # λ₁,λ₂ reconstruidos por si los quieres

for cv in cv_list:
    t = cv
    A1o=np.full(n_Lam,np.nan); A2o=np.full(n_Lam,np.nan)
    R1o=np.full(n_Lam,np.nan); R2o=np.full(n_Lam,np.nan)
    Evo=np.full(n_Lam,np.nan); Eeo=np.full(n_Lam,np.nan)
    Fo =np.full(n_Lam,np.nan); Reo=np.full(n_Lam,np.nan)
    A1e=np.full(n_Lam,np.nan); A2e=np.full(n_Lam,np.nan)
    R1e=np.full(n_Lam,np.nan); R2e=np.full(n_Lam,np.nan)
    Eve=np.full(n_Lam,np.nan); Eee=np.full(n_Lam,np.nan)
    Fe =np.full(n_Lam,np.nan); Ree=np.full(n_Lam,np.nan)
    L1 =np.full(n_Lam,np.nan); L2 =np.full(n_Lam,np.nan)

    warm_o = warm_e = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = Lam_to_lam12(Lam, 0)
        if np.isnan(lam1):
            continue
        L1[k], L2[k] = lam1, lam2
        
        M = _Mof_Lam(Lam)

        # ----- SECTOR IMPAR -----
        E, x = minimize_odd_aniso(lam1, lam2, eps1_val, eps2_val, x_warm=warm_o)
        warm_o = x
        A1o[k], A2o[k], R1o[k], R2o[k] = x
        Evo[k] = E
        Eeo[k] = E_odd_exact_aniso(lam1, lam2, eps1_val, eps2_val, M=M)
        Fo[k]  = fidelity_odd_aniso(lam1, lam2, eps1_val, eps2_val, x, M=M)
        Reo[k] = abs(E - Eeo[k]) / abs(Eeo[k]) if abs(Eeo[k])>1e-12 else np.nan

        # ----- SECTOR PAR -----
        Delta_val = cv
        E, x = minimize_even_aniso(lam1, lam2, eps1_val, eps2_val, Delta_val, x_warm=warm_e)
        warm_e = x
        A1e[k], A2e[k], R1e[k], R2e[k] = x
        Eve[k] = E
        Eee[k] = E_even_exact_aniso(lam1, lam2, eps1_val, eps2_val, Delta_val, M=M)
        Fe[k]  = fidelity_even_aniso(lam1, lam2, eps1_val, eps2_val, Delta_val, x, M=M)
        Ree[k] = abs(E - Eee[k]) / abs(Eee[k]) if abs(Eee[k])>1e-12 else np.nan

    a1_o[cv]=A1o; a2_o[cv]=A2o; r1_o[cv]=R1o; r2_o[cv]=R2o
    Evar_o[cv]=Evo; Eex_o[cv]=Eeo; fid_o[cv]=Fo; rel_o[cv]=Reo
    a1_e[cv]=A1e; a2_e[cv]=A2e; r1_e[cv]=R1e; r2_e[cv]=R2e
    Evar_e[cv]=Eve; Eex_e[cv]=Eee; fid_e[cv]=Fe; rel_e[cv]=Ree
    lam1_rec[cv]=L1; lam2_rec[cv]=L2

print("Ejecución  lista.")

# ══════════════════════════════════════════════════════════════════
# GRAFICACIÓN — parámetros, fidelidad, error (eje = Λ)
# ══════════════════════════════════════════════════════════════════
C_A = plt.cm.seismic(0.12)   # azul  (bloque 2)
C_B = plt.cm.seismic(0.88)   # rojo  (bloque 1)
COLS = [plt.cm.seismic(0.12), plt.cm.seismic(0.6), plt.cm.seismic(0.88)]
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

ylim_by_cv = {0.5: 0.35, 1.0: 0.6, 1.5: 0.8}
xlim_by_cv = {0.5: 4.0, 1.0: 4.5, 1.5: 5.0}

# ── (1) Parámetros variacionales ──────
for cv in cv_list:
    pstr = rf"$\delta\lambda={0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f},\ \Delta/\omega_0={cv}$"
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(Lam_arr, -a1_e[cv], '-',  color=C_B, lw=1.0, label=r'$-\beta_1$')
    ax.plot(Lam_arr, -r1_e[cv], '--', color=C_B, lw=1.0, label=r'$-\xi_1$')
    ax.plot(Lam_arr,  a2_e[cv], '-',  color=C_A, lw=1.0, label=r'$\beta_2$')
    ax.plot(Lam_arr, -r2_e[cv], '--', color=C_A, lw=1.0, label=r'$-\xi_2$')
    ax.plot(Lam_arr, a1_o[cv], '-',  color='green', lw=1.0, label=r'$\alpha$')
    ax.plot(Lam_arr, -r1_o[cv], '--', color='green', lw=1.0, label=r'$-r$')
    ax.axvline(0.5*np.sqrt(2), color='k', lw=1.0, ls=':', alpha=0.5)
    ax.set_xlim(Lam_arr[0], xlim_by_cv[cv])
    ax.set_ylim(0.0, ylim_by_cv[cv])
    ax.set_xlabel(r'$\Lambda$'); ax.set_ylabel(r'Variational parameters')
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True)
    ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
              borderpad=0.6, handlelength=2.2, labelspacing=0.4)
    plt.tight_layout()
    plt.savefig(f'Resultados_aniso/Optimal_parameters_iso_s{cv}_even.png',
                bbox_inches='tight', transparent=False, dpi=300)
    plt.show()
    print('Optimal parameters (even), ' + pstr)


# Grafica 3: Fidelidad y error relativo respecto al estado exacto:LS = {0.5: '-', 1.0: '--', 1.5: ':'}
C_ODD = 'black'
C_EVEN = plt.cm.seismic(0.95)   # rojo intenso

fid_o_err = {cv: 1.0 - fid_o[cv] for cv in cv_list}
fid_e_err = {cv: 1.0 - fid_e[cv] for cv in cv_list}


fig, (ax_o1, ax_o2) = plt.subplots(1, 2, figsize=(12, 6))

# ═══════════════════════════════════════════════════════════════
# Panel (a): Fidelidad
# ═══════════════════════════════════════════════════════════════
ax_e1 = ax_o1.twinx()

for cv in cv_list:
    ax_o1.plot(Lam_arr, fid_o_err[cv], LS[cv], color=C_ODD, lw=1.4)
    ax_e1.plot(Lam_arr, fid_e_err[cv], LS[cv], color=C_EVEN, lw=1.4)

ax_o1.set_xlabel(r'$\Lambda$')
ax_o1.set_ylabel(r'$1-|\langle\psi_{\mathrm{exact}}|\psi_{\mathrm{var}}\rangle|^2$',
                 color=C_ODD)
ax_o1.tick_params(axis='y', colors=C_ODD,  direction='in')
ax_o1.tick_params(axis='x', direction='in')
ax_o1.set_xlim(Lam_arr[0], Lam_arr[-1])

ax_e1.tick_params(axis='y', colors=C_EVEN, direction='in')
ax_e1.spines['right'].set_color(C_EVEN)
ax_e1.spines['left'].set_color(C_ODD)

ax_o1.text(0.95, 0.95, '(a)', transform=ax_o1.transAxes,
           ha='right', va='top', fontweight='bold')

ax_o1.set_xticks([0, 1, 2, 3, 4, 5])

# ═══════════════════════════════════════════════════════════════
# Panel (b): Error relativo
# ═══════════════════════════════════════════════════════════════
ax_e2 = ax_o2.twinx()

for cv in cv_list:
    ax_o2.plot(Lam_arr, 100*rel_o[cv], LS[cv], color=C_ODD,  lw=1.4)
    ax_e2.plot(Lam_arr, 100*rel_e[cv], LS[cv], color=C_EVEN, lw=1.4)

ax_o2.set_xlabel(r'$\Lambda$')
ax_o2.set_ylabel(r'$|E_{\mathrm{var}}-E_{\mathrm{exact}}|/|E_{\mathrm{exact}}|$',
                 color=C_ODD)
ax_o2.tick_params(axis='y', colors=C_ODD,  direction='in')
ax_o2.tick_params(axis='x', direction='in')
ax_o2.set_xlim(Lam_arr[0], Lam_arr[-1])
ax_o2.text(0.0, 1.02, r'$\%$', transform=ax_o2.transAxes,
           color=C_ODD, ha='center', va='bottom')

ax_e2.tick_params(axis='y', colors=C_EVEN, direction='in')
ax_e2.spines['right'].set_color(C_EVEN)
ax_e2.spines['left'].set_color(C_ODD)
ax_e2.text(1.0, 1.01, r'$\%$', transform=ax_e2.transAxes,
           color=C_EVEN, ha='center', va='bottom')

ax_o2.text(0.95, 0.95, '(b)', transform=ax_o2.transAxes,
           ha='right', va='top', fontweight='bold')

ax_o2.set_xticks([0, 1, 2, 3, 4, 5])

# ═══════════════════════════════════════════════════════════════
# Leyenda compartida — a la derecha de la figura
# ═══════════════════════════════════════════════════════════════
handles = [mlines.Line2D([], [], color='gray', ls='-', lw=1.4,
                         label=rf'$s/\omega_0={cv/omega0}$') for cv in cv_list]
handles += [mlines.Line2D([], [], color=C_ODD,  lw=1.4, label='odd'),
            mlines.Line2D([], [], color=C_EVEN, lw=1.4, label='even')]

fig.legend(handles=handles,
           loc='upper center',
           bbox_to_anchor=(0.5,1.01),     # centrada sobre la figura
           ncol=len(handles),               # una sola fila
           frameon=False,
           borderpad=0.6, handlelength=2.2, labelspacing=0.4, columnspacing=1.5)

plt.tight_layout()
plt.subplots_adjust(top=0.85)              # espacio para la leyenda arriba
plt.savefig(f'Resultados_aniso/Fidelidad_y_error_relativo_por_sector.png',
            bbox_inches='tight', transparent=False, dpi=300)
plt.show()


print(r'Fidelity by sector, ' + rf'omega_0={omega0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f}')
print(r'Relative error by sector, ' + rf'omega_0={omega0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f}')