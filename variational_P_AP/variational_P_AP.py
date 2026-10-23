# -*- coding: utf-8 -*-
"""
Ansatz variacional polarón + antipolarón (P+AP)S para la cadena de Kitaev de dos sitios
con cavidades locales (base de Lang-Firsov, modos colectivos).

Convención de pseudoespín (notas):
    sector impar:  |↑_o> = |10>,  |↓_o> = |01>
    sector par:    |↑_e> = |11>,  |↓_e> = |00>
En ambos sectores, en la base {|↑>, |↓>}:
    H_α = [[ ε  + ω0 b†b ,  J D(Λ)        ],
           [ J D(Λ)†     ,  ε' + ω0 b†b   ]]
    impar: ε = ε̃1,        ε' = ε̃2,  J = t
    par:   ε = ε̃1 + ε̃2,   ε' = 0,   J = Δ

Ansatz por rama (|r, η> = D(η) S(r)|0>):
    |φ↑> = N↑ [ |r↑, η↑> + κ↑ |r↓, η↓ + Λ> ]
    |φ↓> = N↓ [ |r↓, η↓> + κ↓ |r↑, η↑ − Λ> ]
    D(η) = exp[η (b† − b)],  S(r) = exp[r (b² − b†²)/2]
Vector de parámetros: x = (η↑, η↓, r↑, r↓, κ↑, κ↓).

La minimización usa únicamente el funcional de energía cerrado (todos los elementos de Fock):
    E = (A↑ + A↓)/2 − sqrt[ ((A↑ − A↓)/2)² + T² ],
    A↑ = ε  + ω0 [η↑² + sinh²r↑ + κ↑²((η↓+Λ)² + sinh²r↓) + 2κ↑ G↑] / (1 + κ↑² + 2κ↑ F),
    A↓ = ε' + ω0 [η↓² + sinh²r↓ + κ↓²((η↑−Λ)² + sinh²r↑) + 2κ↓ G↓] / (1 + κ↓² + 2κ↓ F),
    T  = J [κ↑ + κ↓ + (1 + κ↑κ↓) F] / sqrt[(1 + κ↑² + 2κ↑F)(1 + κ↓² + 2κ↓F)],
con F y G↑, G↓ cerrados (ver _F_G). En δ_α = 0 (ε = ε') se usa la reducción por paridad de
Rabi η↑ = −η↓ = η/2, r↑ = r↓ = r, κ↑ = κ↓ = κ y el funcional reducido E(η, r, κ).
El vector en Fock solo se construye con los parámetros óptimos (fidelidad, pesos, tests).

Figuras: parámetros óptimos y fidelidad / error relativo por sector (sufijo _PAPS),
guardadas en Figures/anstaz/.
"""

import io
import os
import shutil
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import factorial, eval_genlaguerre
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

#  PARÁMETROS

N_JOBS   = -1                  # núcleos para joblib (-1 = todos)

omega0   = 1.0
eps1_val = 0.25
eps2_val = 0.25
cv_list  = [0.5, 1.0, 1.5]     # J_α/ω0  (t = Δ = cv en las figuras)
KAP_MAX  = 1.0                 # κ ∈ [0, KAP_MAX] (redundancia κ -> 1/κ: basta κ <= 1)

# Figuras: eje Λ = √(λ₁² + λ₂²), δλ = 0
LAM_F2   = np.linspace(0.025, 4.0*np.sqrt(2), 400)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', 'anstaz')

#  ESTILO (usa LaTeX si está disponible; si no, mathtext con fuente cm)
_BASE_RC = {
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
}


def _latex_available():
    """True si matplotlib puede renderizar con usetex (latex + dvipng)."""
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
    has_tex = _latex_available()
    rc = dict(_BASE_RC)
    if has_tex:
        rc.update({'text.usetex': True,
                   'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
                   'font.family': 'serif',
                   'font.serif': ['Computer Modern']})
    else:
        rc.update({'text.usetex': False,
                   'mathtext.fontset': 'cm',
                   'font.family': 'serif',
                   'font.serif': ['cmr10', 'DejaVu Serif'],
                   'axes.formatter.use_mathtext': True,
                   'axes.unicode_minus': False})
    plt.rcParams.update(rc)
    print('[estilo] ' + ('LaTeX disponible: usetex' if has_tex
                         else 'LaTeX no disponible: mathtext (Computer Modern)'))
    return has_tex


#  FOCK: desplazamiento exacto (para el hamiltoniano exacto)
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


def _Mof_Lam(Lam):
    return int(min(170, max(50, np.ceil(4.0*Lam*Lam) + 25)))


def Lam_to_lam12(Lam, dlam):
    """λ₁, λ₂ con λ₁² + λ₂² = Λ² y λ₁ − λ₂ = δλ. Devuelve (nan, nan) si no existe."""
    disc = 2.0*Lam**2 - dlam**2
    if disc < 0:
        return np.nan, np.nan
    lam1 = 0.5*(dlam + np.sqrt(disc))
    return lam1, lam1 - dlam


#  FOCK: estados del ansatz (solo con los parámetros óptimos)
class FockOps:
    """Generadores reales antisimétricos en Fock truncado (dim = M+1):
        G1 = b† − b             ->  D(η) = exp(η G1)
        G2 = (b² − b†²)/2       ->  S(r) = exp(r G2)
    exp(x G) se evalúa exactamente con la diagonalización de H = iG (hermítica):
        exp(x G) = V exp(−i x w) V†.
    """

    def __init__(self, M):
        self.M    = M
        self.dim  = M + 1
        self.nvec = np.arange(self.dim, dtype=float)
        b  = np.diag(np.sqrt(np.arange(1, self.dim, dtype=float)), 1)
        bd = b.T
        self.b = b
        gens = {1: bd - b,
                2: 0.5*(b @ b - bd @ bd)}
        self.eig = {}
        for k, G in gens.items():
            w, V = eigh(1j*G)
            self.eig[k] = (w, V, V.conj().T)

    def apply(self, k, x, v):
        """exp(x G_k) v."""
        if x == 0.0:
            return v
        w, V, Vh = self.eig[k]
        return (V @ (np.exp(-1j*x*w) * (Vh @ v))).real

    def state(self, eta, r):
        """|r, η> = D(η) S(r) |0>."""
        v = np.zeros(self.dim)
        v[0] = 1.0
        v = self.apply(2, r, v)
        v = self.apply(1, eta, v)
        return v


_OPS_CACHE = {}


def get_ops(M):
    if M not in _OPS_CACHE:
        _OPS_CACHE[M] = FockOps(M)
    return _OPS_CACHE[M]


#  SECTORES 
def sector_params(sector, lam1, lam2, eps1, eps2, t, Delta):
    """(ε, ε', J) del bloque en la base {|↑>, |↓>}."""
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    if sector == 'odd':        # |↑_o> = |10>, |↓_o> = |01>
        return e1t, e2t, t
    else:                      # |↑_e> = |11>, |↓_e> = |00>
        return e1t + e2t, 0.0, Delta


#  FUNCIONAL DE ENERGÍA 
def _F_G(x, Lam):
    """F = <r↑,η↑|D(Λ)|r↓,η↓>, G↑ = <r↑,η↑|b†b|r↓,η↓+Λ>, G↓ = <r↓,η↓|b†b|r↑,η↑−Λ>."""
    eu, ed, ru, rd = x[0], x[1], x[2], x[3]
    Dl = Lam + ed - eu                                   # Λ + η↓ − η↑
    ch = np.cosh(ru - rd)
    F  = np.exp(-Dl**2/(np.exp(-2.0*ru) + np.exp(-2.0*rd)))/np.sqrt(ch)
    su, sd = np.sinh(ru), np.sinh(rd)
    au = Dl*np.exp(rd)*su/ch                             # Δ e^{r↓} sinh r↑ / cosh(r↑ − r↓)
    ad = Dl*np.exp(ru)*sd/ch                             # Δ e^{r↑} sinh r↓ / cosh(r↑ − r↓)
    Gu = F*((eu - au)*(ed + Lam + ad) + su*sd/ch)
    Gd = F*((ed + ad)*(eu - Lam - au) + su*sd/ch)
    return F, Gu, Gd


def closed_ABC(x, eps, epsp, J, Lam):
    """A↑, A↓, T del funcional  (x = (η↑, η↓, r↑, r↓, κ↑, κ↓))."""
    eu, ed, ru, rd, ku, kd = x[:6]
    F, Gu, Gd = _F_G(x, Lam)
    nu = 1.0 + ku*ku + 2.0*ku*F                          # N↑^{-2}
    nd = 1.0 + kd*kd + 2.0*kd*F                          # N↓^{-2}
    if nu < 1e-12 or nd < 1e-12:                         # |φ_s> = 0: no es un estado
        return np.nan, np.nan, np.nan
    A = eps  + omega0*(eu*eu + np.sinh(ru)**2 + ku*ku*((ed + Lam)**2 + np.sinh(rd)**2)
                       + 2.0*ku*Gu)/nu
    B = epsp + omega0*(ed*ed + np.sinh(rd)**2 + kd*kd*((eu - Lam)**2 + np.sinh(ru)**2)
                       + 2.0*kd*Gd)/nd
    T = J*(ku + kd + (1.0 + ku*kd)*F)/np.sqrt(nu*nd)
    return A, B, T


def energy_closed(x, eps, epsp, J, Lam):
    """Funcional general."""
    A, B, T = closed_ABC(x, eps, epsp, J, Lam)
    if not np.isfinite(A):
        return 1e6
    return 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + T*T)


def energy_closed_sym(xs, eps, epsp, J, Lam):
    """Funcional reducido en δ_α = 0, E(η, r, κ),
    E = ε + {ω0[η_n²/4 + (1+κ²) sinh²r + κ²(Λ − η_n/2)² + 2κF(Λ²/4 − (Λ−η_n)² e^{4r}/4 + sinh²r)]
             − J |2κ + (1+κ²)F|} / (1 + κ² + 2κF),   F = exp[−(Λ−η_n)² e^{2r}/2]."""
    eta_n, r, kap = 2.0*xs[0], xs[1], xs[2]
    F   = np.exp(-0.5*(Lam - eta_n)**2*np.exp(2.0*r))
    sh2 = np.sinh(r)**2
    den = 1.0 + kap**2 + 2.0*kap*F
    if den < 1e-12:                                      # κ = −1 con F = 1: |φ↑> = 0
        return 1e6
    num = (omega0*(0.25*eta_n**2 + (1.0 + kap**2)*sh2 + kap**2*(Lam - 0.5*eta_n)**2
                   + 2.0*kap*F*(0.25*Lam**2 - 0.25*(Lam - eta_n)**2*np.exp(4.0*r) + sh2))
           - abs(J)*abs(2.0*kap + (1.0 + kap**2)*F))
    return eps + num/den


def var_states(x, Lam, ops):
    """|φ↑>, |φ↓> normalizados en Fock (solo con los parámetros óptimos)."""
    eu, ed, ru, rd, ku, kd = x[:6]
    pu = ops.state(eu, ru) + ku*ops.state(ed + Lam, rd)
    pd = ops.state(ed, rd) + kd*ops.state(eu - Lam, ru)
    return pu/np.linalg.norm(pu), pd/np.linalg.norm(pd)


def energy_fock(x, eps, epsp, J, Lam, ops):
    """El mismo funcional evaluado en Fock (solo para el test T2)."""
    pu, pd = var_states(x, Lam, ops)
    A = eps  + omega0*(pu @ (ops.nvec*pu))
    B = epsp + omega0*(pd @ (ops.nvec*pd))
    T = J*(pu @ ops.apply(1, Lam, pd))
    return 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + T*T)


def weights_2x2(A, B, C):
    """(c↑, c↓) del menor autovalor de [[A, C], [C, B]].
    Usa (C, E−A) o (E−B, C), la de mayor norma (evita 0/0)."""
    E  = 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + C*C)
    v1 = np.array([C, E - A])
    v2 = np.array([E - B, C])
    v  = v1 if np.hypot(*v1) >= np.hypot(*v2) else v2
    return v / np.hypot(*v)


#  MINIMIZACIÓN
KAP_SEEDS = (0.15, 0.5, 0.0)


def _seeds(Lam, n_par, sector, warm=False):

    h = 0.5*Lam
    old = [(0, 0, 0, 0), (-h, h, -0.1, -0.1), (h, -h, -0.1, -0.1), (-h, h, 0, 0),
           (0, 0, -0.1, -0.1), (-Lam, 0, -0.1, -0.1), (0, Lam, -0.1, -0.1),
           (h, -h, 0, 0), (h, -h, -0.2, -0.2)]
    if sector == 'even':
        old = [(b, a, rb, ra) for (a, b, ra, rb) in old]
    kaps = KAP_SEEDS[:2] if warm else KAP_SEEDS
    s4 = [tuple(float(v) for v in sd) for sd in old]
    if n_par == 6:                                   # general
        return [sd + (k, k) for sd in s4 for k in kaps]
    s2 = list(dict.fromkeys((0.5*(a - b), 0.5*(ra + rb)) for (a, b, ra, rb) in s4))
    return [sd + (k,) for sd in s2 for k in kaps]   # simétrico


def _bounds(Lam, n_par):
    be, br = (-Lam - 1.0, Lam + 1.0), (-1.5, 1.5)
    bk = (0.0, KAP_MAX)
    return {3: [be, br, bk], 6: [be, be, br, br, bk, bk]}[n_par]


def _run_seeds(fun, args, Lam, n_par, x_warm, sector):
    seeds = _seeds(Lam, n_par, sector, warm=x_warm is not None)
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = _bounds(Lam, n_par)
    lo = np.array([b[0] for b in bnd])
    hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0  = np.clip(np.asarray(sd, float), lo, hi)
        res = minimize(fun, x0, args=args, method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


def _sym_to_full(xs):
    """(η, r, κ) -> (η, −η, r, r, κ, κ): paridad de Rabi para δ_α = 0."""
    eta, r, kap = xs[0], xs[1], xs[2]
    return np.array([eta, -eta, r, r, kap, kap])


def minimize_sector(sector, eps, epsp, J, Lam, x_warm=None):
    """Devuelve (E, x) con x = (η↑, η↓, r↑, r↓, κ↑, κ↓).
    Si δ_α = (ε − ε')/2 = 0, se impone la paridad de Rabi Ŝ_α de las notas:
    η↓ = −η↑, r↓ = r↑, κ↓ = κ↑, y se minimiza el funcional reducido E(η, r, κ)."""
    if abs(eps - epsp) < 1e-12:
        xw = None if x_warm is None else (x_warm[0], x_warm[2], x_warm[4])
        E, xs = _run_seeds(energy_closed_sym, (eps, epsp, J, Lam), Lam, 3, xw, sector)
        return E, _sym_to_full(xs)
    xw = None if x_warm is None else tuple(x_warm[:6])
    return _run_seeds(energy_closed, (eps, epsp, J, Lam), Lam, 6, xw, sector)


#  EXACTO (un modo acoplado, convención de las notas)
def exact_ground(eps, epsp, J, Lam, M):
    n1 = M + 1
    DL = disp_matrix(M, Lam)
    nd = np.arange(n1)*omega0
    H  = np.zeros((2*n1, 2*n1))
    H[:n1, :n1] = np.diag(eps + nd)
    H[n1:, n1:] = np.diag(epsp + nd)
    H[:n1, n1:] = J*DL
    H[n1:, :n1] = J*DL.T
    w, V = eigh(H)
    return w[0], V[:, 0]


#  BARRIDOS
def sweep_Lambda(sector, cv, Lam_arr):
    """Barrido en Λ con continuación (warm start). t = Δ = cv."""
    n    = len(Lam_arr)
    X    = np.full((6, n), np.nan)
    Ev   = np.full(n, np.nan)
    Eex  = np.full(n, np.nan)
    Efk  = np.full(n, np.nan)
    fid  = np.full(n, np.nan)
    rel  = np.full(n, np.nan)
    leak = np.full(n, np.nan)
    cu   = np.full(n, np.nan)
    cd   = np.full(n, np.nan)
    warm = None
    for k, Lam in enumerate(Lam_arr):
        lam1, lam2 = Lam_to_lam12(Lam, 0.0)
        if np.isnan(lam1):
            continue
        M = _Mof_Lam(Lam)
        eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, cv, cv)
        E, x = minimize_sector(sector, eps, epsp, J, Lam, warm)
        warm = x
        X[:, k] = x
        Ev[k]   = E
        ops = get_ops(M)
        A, B, C = closed_ABC(x, eps, epsp, J, Lam)
        c1, c2  = weights_2x2(A, B, C)
        pu, pd  = var_states(x, Lam, ops)
        E0, psi_ex = exact_ground(eps, epsp, J, Lam, M)
        psi_var = np.concatenate([c1*pu, c2*pd])
        Eex[k]  = E0
        Efk[k]  = energy_fock(x, eps, epsp, J, Lam, ops)
        fid[k]  = abs(psi_ex @ psi_var)**2
        rel[k]  = abs(E - E0)/abs(E0) if abs(E0) > 1e-12 else np.nan
        leak[k] = max(np.sum(pu[-5:]**2), np.sum(pd[-5:]**2))
        cu[k], cd[k] = c1, c2
    return dict(X=X, Evar=Ev, Eex=Eex, Efock=Efk, fid=fid, rel=rel, leak=leak, cu=cu, cd=cd)


def compute_fig2():
    _OPS_CACHE.clear()
    keys = [(s, cv) for cv in cv_list for s in ('odd', 'even')]
    res  = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(sweep_Lambda)(s, cv, LAM_F2) for s, cv in keys)
    return dict(zip(keys, res))


#  FIGURA: parámetros variacionales
C_A    = plt.cm.seismic(0.12)    # azul  (rama ↓)
C_B    = plt.cm.seismic(0.88)    # rojo  (rama ↑)
C_ODD  = 'black'
C_EVEN = plt.cm.seismic(0.95)


def _plot_param(ax, x, y, sym, ls, color):
    """Grafica y (o −y si y es mayoritariamente negativo) con la etiqueta correspondiente."""
    yy = y[np.isfinite(y)]
    s  = -1.0 if (yy.size and np.mean(yy) < 0) else 1.0
    lab = rf'$-{sym}$' if s < 0 else rf'${sym}$'
    ax.plot(x, s*y, ls, color=color, lw=1.0, label=lab)


def plot_fig2(R, suffix):
    """Una figura por sector y por J_α/ω0."""
    Lam_arr = LAM_F2
    ylim_by_cv = {0.5: 0.35, 1.0: 0.6, 1.5: 0.8}

    # (símbolo, índice en x, estilo, color); x = (η↑, η↓, r↑, r↓, κ↑, κ↓)
    curves = {
        # sector par: índice 1 = |↑_e> = |11>, índice 2 = |↓_e> = |00>
        'even': [(r'\beta_1', 0, '-',  C_B), (r'\xi_1', 2, '--', C_B), (r'\kappa_1', 4, '-.', C_B),
                 (r'\beta_2', 1, '-',  C_A), (r'\xi_2', 3, '--', C_A), (r'\kappa_2', 5, '-.', C_A)],
        # sector impar (δ_o = 0, simétrico): rama |↑_o> = |10>
        'odd':  [(r'\alpha', 0, '-', 'green'), (r'r', 2, '--', 'green'), (r'\kappa', 4, '-.', 'green')],
    }

    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        for cv in cv_list:
            pstr = rf"$\delta\lambda={0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f},\ J_\alpha/\omega_0={cv}$"
            for sector in ('even', 'odd'):
                X = R[(sector, cv)]['X']
                fig, ax = plt.subplots(figsize=(5, 4))
                for sym, i, ls, col in curves[sector]:
                    _plot_param(ax, Lam_arr, X[i], sym, ls, col)
                ax.axvline(0.5*np.sqrt(2), color='k', lw=1.0, ls=':', alpha=0.5)
                ax.set_xlim(Lam_arr[0], Lam_arr[-1])      # mismo rango en Λ que la fidelidad
                ax.set_ylim(0.0, ylim_by_cv[cv])
                ax.set_xlabel(r'$\Lambda$')
                ax.set_ylabel(r'Variational parameters')
                ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                               right=True, left=True)
                ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                          borderpad=0.6, handlelength=2.2, labelspacing=0.4)
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f'Optimal_parameters_iso_s{cv}_{sector}{suffix}.png'),
                            bbox_inches='tight', transparent=False, dpi=300)
                plt.close(fig)
                print(f'Optimal parameters ({sector}), ' + pstr)


# ══════════════════════════════════════════════════════════════════
#  FIGURA 3: fidelidad y error relativo respecto al estado exacto
# ══════════════════════════════════════════════════════════════════
def _sci_axis(ax, data, side, color, percent=False):
    """Etiquetas del eje y con una cifra decimal y el multiplicador ×10^e
    (y %, si aplica) en la parte superior del eje."""
    vals = np.concatenate([np.ravel(d) for d in data])
    vals = vals[np.isfinite(vals)]
    vmax = np.max(np.abs(vals)) if vals.size else 0.0
    e = int(np.floor(np.log10(vmax))) if vmax > 0 else 0
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/10.0**e:.1f}'))
    tag = (rf'\times 10^{{{e}}}' if e != 0 else '') + (r'\,\%' if percent else '')
    if tag:
        x, ha = (0.0, 'left') if side == 'left' else (1.0, 'right')
        ax.text(x, 1.02, '$' + tag + '$', transform=ax.transAxes,
                color=color, ha=ha, va='bottom')


def plot_fig3(R, suffix):
    Lam_arr = LAM_F2
    LS = {0.5: '-', 1.0: '--', 1.5: ':'}
    fid_o_err = {cv: 1.0 - R[('odd', cv)]['fid']  for cv in cv_list}
    fid_e_err = {cv: 1.0 - R[('even', cv)]['fid'] for cv in cv_list}
    rel_o = {cv: R[('odd', cv)]['rel']  for cv in cv_list}
    rel_e = {cv: R[('even', cv)]['rel'] for cv in cv_list}

    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        fig, (ax_o1, ax_o2) = plt.subplots(1, 2, figsize=(10, 4))

        # Panel (a): fidelidad
        ax_e1 = ax_o1.twinx()
        for cv in cv_list:
            ax_o1.plot(Lam_arr, fid_o_err[cv], LS[cv], color=C_ODD,  lw=1.4)
            ax_e1.plot(Lam_arr, fid_e_err[cv], LS[cv], color=C_EVEN, lw=1.4)
        ax_o1.set_xlabel(r'$\Lambda$')
        ax_o1.set_ylabel(r'$1-|\langle\psi_{\mathrm{exact}}|\psi_{\mathrm{var}}\rangle|^2$',
                         color=C_ODD)
        ax_o1.tick_params(axis='y', colors=C_ODD, direction='in')
        ax_o1.tick_params(axis='x', direction='in')
        ax_o1.set_xlim(Lam_arr[0], Lam_arr[-1])
        ax_e1.tick_params(axis='y', colors=C_EVEN, direction='in')
        ax_e1.spines['right'].set_color(C_EVEN)
        ax_e1.spines['left'].set_color(C_ODD)
        ax_o1.text(0.95, 0.95, '(a)', transform=ax_o1.transAxes,
                   ha='right', va='top', fontweight='bold')
        ax_o1.set_xticks([0, 1, 2, 3, 4, 5])
        _sci_axis(ax_o1, [fid_o_err[cv] for cv in cv_list], 'left',  C_ODD)
        _sci_axis(ax_e1, [fid_e_err[cv] for cv in cv_list], 'right', C_EVEN)

        # Panel (b): error relativo
        ax_e2 = ax_o2.twinx()
        for cv in cv_list:
            ax_o2.plot(Lam_arr, 100*rel_o[cv], LS[cv], color=C_ODD,  lw=1.4)
            ax_e2.plot(Lam_arr, 100*rel_e[cv], LS[cv], color=C_EVEN, lw=1.4)
        ax_o2.set_xlabel(r'$\Lambda$')
        ax_o2.set_ylabel(r'$|E_{\mathrm{var}}-E_{\mathrm{exact}}|/|E_{\mathrm{exact}}|$',
                         color=C_ODD)
        ax_o2.tick_params(axis='y', colors=C_ODD, direction='in')
        ax_o2.tick_params(axis='x', direction='in')
        ax_o2.set_xlim(Lam_arr[0], Lam_arr[-1])
        ax_e2.tick_params(axis='y', colors=C_EVEN, direction='in')
        ax_e2.spines['right'].set_color(C_EVEN)
        ax_e2.spines['left'].set_color(C_ODD)
        ax_o2.text(0.95, 0.95, '(b)', transform=ax_o2.transAxes,
                   ha='right', va='top', fontweight='bold')
        ax_o2.set_xticks([0, 1, 2, 3, 4, 5])
        _sci_axis(ax_o2, [100*rel_o[cv] for cv in cv_list], 'left',  C_ODD,  percent=True)
        _sci_axis(ax_e2, [100*rel_e[cv] for cv in cv_list], 'right', C_EVEN, percent=True)

        # Leyenda compartida
        handles = [mlines.Line2D([], [], color='gray', ls=LS[cv], lw=1.4,
                                 label=rf'$J_\alpha/\omega_0={cv/omega0}$') for cv in cv_list]
        handles += [mlines.Line2D([], [], color=C_ODD,  lw=1.4, label='odd'),
                    mlines.Line2D([], [], color=C_EVEN, lw=1.4, label='even')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.97),
                   ncol=len(handles), frameon=False, borderpad=0.6, handlelength=1.5,
                   labelspacing=0.3, columnspacing=1.2)

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)      # espacio para los multiplicadores ×10^e
        plt.savefig(os.path.join(OUT_DIR, f'Fidelidad_y_error_relativo_por_sector{suffix}.png'),
                    bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig)

    print(r'Fidelity by sector, ' + rf'omega_0={omega0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f}')
    print(r'Relative error by sector, ' + rf'omega_0={omega0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f}')


#  TESTS IMPRESOS
def _hr(title):
    print('\n' + '='*72 + f'\n {title}\n' + '='*72)


def tests_pre():
    _hr('TESTS PREVIOS')
    rng = np.random.default_rng(1234)
    ops = get_ops(170)

    # T2: funcional cerrado general vs el mismo funcional evaluado en Fock, puntos aleatorios
    for sector in ('odd', 'even'):
        for Lam in (1.0, 2.5, 4.0):
            lam1, lam2 = Lam_to_lam12(Lam, 0.0)
            eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, 1.0, 1.0)
            d = []
            for _ in range(100):
                x6 = np.array([rng.uniform(-Lam, Lam), rng.uniform(-Lam, Lam),
                               rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
                               rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)])
                d.append(abs(energy_fock(x6, eps, epsp, J, Lam, ops)
                             - energy_closed(x6, eps, epsp, J, Lam)))
            print(f'[T2] {sector:4s}  Λ={Lam:3.1f}  max|E_Fock - E_cerrada| = {max(d):.2e}')

    # T2b: funcional reducido E(η, r, κ) vs funcional general en la reducción simétrica
    d = []
    for _ in range(200):
        Lam = rng.uniform(0.2, 5.0)
        xs  = (rng.uniform(-Lam, Lam), rng.uniform(-0.8, 0.3), rng.uniform(-0.9, 0.9))
        eps = rng.uniform(-3.0, 1.0)
        d.append(abs(energy_closed_sym(xs, eps, eps, 1.0, Lam)
                     - energy_closed(_sym_to_full(xs), eps, eps, 1.0, Lam)))
    print(f'[T2b] max|E_reducido(η,r,κ) - E_general(simétrico)| = {max(d):.2e}')

    # T2c: paridad de Rabi: con la reducción simétrica, |φ↓> = Π|φ↑>
    Pi = (-1.0)**np.arange(ops.dim)
    for xs in ((0.3, -0.1, 0.2), (0.6, -0.2, 0.1)):
        pu, pd = var_states(_sym_to_full(xs), 2.5, ops)
        print(f'[T2c] (η,r,κ)={xs}:  ||φ↓ - Π φ↑|| = {np.linalg.norm(pd - Pi*pu):.2e}')


def tests_post(R):
    _hr('TESTS POSTERIORES')
    for (sector, cv), d in R.items():
        gap = d['Evar'] - d['Eex']
        print(f'[T5] {sector:4s} J/ω0={cv}:  min(E_var - E_ex) = {np.nanmin(gap):+.2e}'
              f'   violaciones de la cota variacional = {int(np.sum(gap < -1e-9))}')
        print(f'[T6] {sector:4s} J/ω0={cv}:  máx|E_cerrada - E_Fock| en el óptimo = '
              f'{np.nanmax(np.abs(d["Evar"] - d["Efock"])):.2e}')
        print(f'[T7] {sector:4s} J/ω0={cv}:  máx. peso en los 5 últimos niveles de Fock = {np.nanmax(d["leak"]):.2e}')
        nk = int(np.sum(np.abs(d['X'][4:6]) > KAP_MAX - 1e-6))
        neg = np.any(d['X'][4:6] < 1e-6, axis=0)
        print(f'[T7] {sector:4s} J/ω0={cv}:  κ ∈ [{np.nanmin(d["X"][4:6]):+.3f}, {np.nanmax(d["X"][4:6]):+.3f}]'
              f'   puntos con |κ| en la cota ({KAP_MAX}) = {nk}   puntos con κ en la cota inferior (0) = {int(np.sum(neg))}'
              + (f'   Λ = {np.array2string(LAM_F2[neg], precision=3, separator=", ")}' if neg.any() else ''))
        if sector == 'odd':
            print(f'[T8] odd  J/ω0={cv}:  máx|c↑ + c↓| (δ_o = 0) = {np.nanmax(np.abs(d["cu"] + d["cd"])):.2e}')

#Ejecucion
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    has_tex = setup_style()
    tests_pre()

    _hr('CORRIDA: (P+AP)S, funcional cerrado')
    print('Figuras 2 y 3: barridos en Λ ...')
    R = compute_fig2()
    plot_fig2(R, '_PAPS')
    plot_fig3(R, '_PAPS')

    tests_post(R)
    print(f'\nFiguras guardadas en: {OUT_DIR}')
