# -*- coding: utf-8 -*-
"""
Ansatz variacional con trisqueezing para la cadena de Kitaev de dos sitios
con cavidades locales (base de Lang-Firsov, modos colectivos).

Convención de pseudoespín (notas):
    sector impar:  |↑_o> = |10>,  |↓_o> = |01>
    sector par:    |↑_e> = |11>,  |↓_e> = |00>
En ambos sectores, en la base {|↑>, |↓>}:
    H_α = [[ ε  + ω0 b†b ,  J D(Λ)        ],
           [ J D(Λ)†     ,  ε' + ω0 b†b   ]]
    impar: ε = ε̃1,        ε' = ε̃2,  J = t
    par:   ε = ε̃1 + ε̃2,   ε' = 0,   J = Δ

Ansatz por rama s = ↑, ↓:   |φ_s> = D(η_s) S(r_s) T(τ_s) |0>
    D(η) = exp[η (b† − b)],  S(r) = exp[r (b² − b†²)/2],  T(τ) = exp[τ (b†³ − b³)]
Vector de parámetros: x = (η↑, η↓, r↑, r↓, τ↑, τ↓).

Los tres operadores se construyen a mano en el espacio de Fock truncado
(dimensión M+1) y se exponencian exactamente vía diagonalización de su
generador antihermítico.

Corridas (en este orden con MODE = 'all'):
    1) τ = 0 fijo  -> prueba: debe reproducir las figuras originales (sufijo _tau0)
    2) trisqueezing (τ libre)
    3) convergencia del variacional con trisqueezing en el corte de Fock N
Todas las figuras se guardan en Resultados_variational_trisqueezing/.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import factorial, eval_genlaguerre
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
MODE     = 'all'               # 'tau0' | 'trisq' | 'all'  ('all' corre primero τ = 0)
N_JOBS   = 128                  # núcleos para joblib (-1 = todos)

omega0   = 1.0
eps1_val = 0.25
eps2_val = 0.25
cv_list  = [0.5, 1.0, 1.5]     # J_α/ω0  (t = Δ = cv en las figuras 2 y 3)
N_CONV   = [40, 80, 120, 170]  # cortes de Fock para el test de convergencia
TAU_MAX  = 0.08                # cota de |τ|: para |τ| ≳ 0.1, T(τ)|0> no converge con el corte
                               # de Fock (cola de ley de potencias, ver test T3b)


# Figura 1: plano (λ/t, Δ/t), λ₁ = λ₂ = λ
T_F1     = 1.0
LAM_F1   = np.linspace(0.0, 1.5, 120)
DELTA_F1 = np.linspace(0.0, 1.5, 120)

# Figuras 2, 3 y convergencia: eje Λ = √(λ₁² + λ₂²), δλ = 0
LAM_F2   = np.linspace(0.01, 4.0*np.sqrt(2), 150)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Resultados_variational_trisqueezing')

# ══════════════════════════════════════════════════════════════════
#  ESTILO (usa LaTeX si está disponible; si no, mathtext con fuente cm)
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


# ══════════════════════════════════════════════════════════════════
#  FOCK: desplazamiento exacto (para el hamiltoniano exacto)
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


def _Mof_Lam(Lam):
    return int(min(170, max(50, np.ceil(4.0*Lam*Lam) + 25)))


def Lam_to_lam12(Lam, dlam):
    """λ₁, λ₂ con λ₁² + λ₂² = Λ² y λ₁ − λ₂ = δλ. Devuelve (nan, nan) si no existe."""
    disc = 2.0*Lam**2 - dlam**2
    if disc < 0:
        return np.nan, np.nan
    lam1 = 0.5*(dlam + np.sqrt(disc))
    return lam1, lam1 - dlam


# ══════════════════════════════════════════════════════════════════
#  FOCK: operadores del ansatz construidos a mano
# ══════════════════════════════════════════════════════════════════
class FockOps:
    """Generadores reales antisimétricos en Fock truncado (dim = M+1):
        G1 = b† − b             ->  D(η) = exp(η G1)
        G2 = (b² − b†²)/2       ->  S(r) = exp(r G2)
        G3 = b†³ − b³           ->  T(τ) = exp(τ G3)
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
                2: 0.5*(b @ b - bd @ bd),
                3: bd @ bd @ bd - b @ b @ b}
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

    def state(self, eta, r, tau):
        """|φ> = D(η) S(r) T(τ) |0>."""
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
#  SECTORES (convención de las notas)
# ══════════════════════════════════════════════════════════════════
def sector_params(sector, lam1, lam2, eps1, eps2, t, Delta):
    """(ε, ε', J) del bloque en la base {|↑>, |↓>}."""
    e1t = eps1 - lam1**2 * omega0
    e2t = eps2 - lam2**2 * omega0
    if sector == 'odd':        # |↑_o> = |10>, |↓_o> = |01>
        return e1t, e2t, t
    else:                      # |↑_e> = |11>, |↓_e> = |00>
        return e1t + e2t, 0.0, Delta


# ══════════════════════════════════════════════════════════════════
#  FUNCIONAL DE ENERGÍA
# ══════════════════════════════════════════════════════════════════
def _unpack(x, fix_tau0):
    if fix_tau0:
        eu, ed, ru, rd = x
        tu = td = 0.0
    else:
        eu, ed, ru, rd, tu, td = x
    return eu, ed, ru, rd, tu, td


def var_quantities(x, eps, epsp, J, Lam, ops, fix_tau0):
    """E, A, B, C, |φ↑>, |φ↓> para el ansatz D S T |0>."""
    eu, ed, ru, rd, tu, td = _unpack(x, fix_tau0)
    pu = ops.state(eu, ru, tu)
    pd = ops.state(ed, rd, td)
    mu = pu @ (ops.nvec * pu)                 # <m>_↑
    md = pd @ (ops.nvec * pd)                 # <m>_↓
    F  = pu @ ops.apply(1, Lam, pd)           # <φ↑| D(Λ) |φ↓>
    A  = eps  + omega0*mu
    B  = epsp + omega0*md
    C  = J*F
    E  = 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + C*C)
    return E, A, B, C, pu, pd


def energy_numeric(x, eps, epsp, J, Lam, ops, fix_tau0):
    return var_quantities(x, eps, epsp, J, Lam, ops, fix_tau0)[0]


def energy_analytic(x, eps, epsp, J, Lam):
    """Funcional cerrado del ansatz D S |0> (τ = 0), convención de las notas.
    Solo se usa como referencia en los tests."""
    eu, ed, ru, rd = x
    mu = eu*eu + np.sinh(ru)**2
    md = ed*ed + np.sinh(rd)**2
    F  = (np.exp(-(Lam + ed - eu)**2 / (np.exp(-2*ru) + np.exp(-2*rd)))
          / np.sqrt(np.cosh(rd - ru)))
    A  = eps  + omega0*mu
    B  = epsp + omega0*md
    C  = J*F
    return 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + C*C)


def _energy_analytic_sym(xs, eps, epsp, J, Lam):
    return energy_analytic(_sym_to_full(xs)[:4], eps, epsp, J, Lam)


def weights_2x2(A, B, C):
    """(c↑, c↓) del menor autovalor de [[A, C], [C, B]].
    Usa (C, E−A) o (E−B, C), la de mayor norma (evita 0/0)."""
    E  = 0.5*(A + B) - np.sqrt(0.25*(A - B)**2 + C*C)
    v1 = np.array([C, E - A])
    v2 = np.array([E - B, C])
    v  = v1 if np.hypot(*v1) >= np.hypot(*v2) else v2
    return v / np.hypot(*v)


# ══════════════════════════════════════════════════════════════════
#  MINIMIZACIÓN
# ══════════════════════════════════════════════════════════════════
def _seeds(Lam, n_par, sector):
    """Semillas de la versión original. En el sector par se intercambian las ramas
    (↑ <-> ↓) para pasar de la convención vieja (bloque 1 = |00>) a la de las notas
    (|↑_e> = |11>). Para el ansatz simétrico (δ_α = 0) se usa la proyección
    η = (η↑ − η↓)/2, r = (r↑ + r↓)/2 de esas mismas semillas."""
    h = 0.5*Lam
    old = [(0, 0, 0, 0), (-h, h, -0.1, -0.1), (h, -h, -0.1, -0.1), (-h, h, 0, 0),
           (0, 0, -0.1, -0.1), (-Lam, 0, -0.1, -0.1), (0, Lam, -0.1, -0.1),
           (h, -h, 0, 0), (h, -h, -0.2, -0.2)]
    if sector == 'even':
        old = [(b, a, rb, ra) for (a, b, ra, rb) in old]
    s4 = [tuple(float(v) for v in sd) for sd in old]
    if n_par == 4:                                   # general, τ = 0
        return s4
    if n_par == 6:                                   # general, τ libre
        s6 = [sd + (0.0, 0.0) for sd in s4]
        for tt in (0.02, -0.02):
            s6 += [(h, -h, -0.1, -0.1, tt, -tt),
                   (0.0, 0.0, -0.1, -0.1, tt, -tt),
                   (h, -h, -0.1, -0.1, tt, tt)]
        return s6
    s2 = list(dict.fromkeys((0.5*(a - b), 0.5*(ra + rb)) for (a, b, ra, rb) in s4))
    if n_par == 2:                                   # simétrico, τ = 0
        return s2
    s3 = [sd + (0.0,) for sd in s2]                  # simétrico, τ libre
    for tt in (0.02, -0.02):
        s3 += [(h, -0.1, tt), (0.0, -0.1, tt)]
    return s3


def _bounds(Lam, n_par):
    be, br, bt = (-Lam - 1.0, Lam + 1.0), (-1.5, 1.5), (-TAU_MAX, TAU_MAX)
    return {2: [be, br], 3: [be, br, bt], 4: [be, be, br, br],
            6: [be, be, br, br, bt, bt]}[n_par]


def _run_seeds(fun, args, Lam, n_par, x_warm, sector):
    seeds = _seeds(Lam, n_par, sector)
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    bnd = _bounds(Lam, n_par)
    lo = np.array([b[0] for b in bnd])
    hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        x0  = np.clip(np.asarray(sd, float), lo, hi)
        res = minimize(fun, x0, args=args, method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 800})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


def _sym_to_full(xs):
    """(η, r[, τ]) -> (η, −η, r, r, τ, −τ): condición de simetría para δ_α = 0."""
    eta, r = xs[0], xs[1]
    tau = xs[2] if len(xs) == 3 else 0.0
    return np.array([eta, -eta, r, r, tau, -tau])


def energy_numeric_sym(xs, eps, epsp, J, Lam, ops):
    return var_quantities(_sym_to_full(xs), eps, epsp, J, Lam, ops, False)[0]


def minimize_sector(sector, eps, epsp, J, Lam, M, fix_tau0, x_warm=None):
    """Devuelve (E, x) con x = (η↑, η↓, r↑, r↓, τ↑, τ↓) (τ = 0 si fix_tau0).
    Si δ_α = (ε − ε')/2 = 0, se impone la simetría Ŝ_α de las notas:
    η↓ = −η↑, r↓ = r↑, τ↓ = −τ↑ (sin ella, el gradiente numérico rompe la simetría
    y la minimización puede caer en mínimos asimétricos que el código original,
    con gradiente analítico exacto, nunca exploraba)."""
    ops = get_ops(M)
    if abs(eps - epsp) < 1e-12:
        n_par = 2 if fix_tau0 else 3
        xw = None
        if x_warm is not None:
            xw = (x_warm[0], x_warm[2]) if fix_tau0 else (x_warm[0], x_warm[2], x_warm[4])
        E, xs = _run_seeds(energy_numeric_sym, (eps, epsp, J, Lam, ops),
                           Lam, n_par, xw, sector)
        return E, _sym_to_full(xs)
    n_par = 4 if fix_tau0 else 6
    xw = None if x_warm is None else tuple(x_warm[:n_par])
    E, x = _run_seeds(energy_numeric, (eps, epsp, J, Lam, ops, fix_tau0),
                      Lam, n_par, xw, sector)
    if fix_tau0:
        x = np.concatenate([x, [0.0, 0.0]])
    return E, x


# ══════════════════════════════════════════════════════════════════
#  EXACTO (un modo acoplado, convención de las notas)
# ══════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════
#  BARRIDOS
# ══════════════════════════════════════════════════════════════════
def sweep_Lambda(sector, cv, Lam_arr, fix_tau0, M_fixed=None, with_exact=True):
    """Barrido en Λ con continuación (warm start). t = Δ = cv."""
    n    = len(Lam_arr)
    X    = np.full((6, n), np.nan)
    Ev   = np.full(n, np.nan)
    Eex  = np.full(n, np.nan)
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
        M = M_fixed if M_fixed is not None else _Mof_Lam(Lam)
        eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, cv, cv)
        E, x = minimize_sector(sector, eps, epsp, J, Lam, M, fix_tau0, warm)
        warm = x
        X[:, k] = x
        Ev[k]   = E
        if with_exact:
            ops = get_ops(M)
            _, A, B, C, pu, pd = var_quantities(x, eps, epsp, J, Lam, ops, False)
            c1, c2 = weights_2x2(A, B, C)
            E0, psi_ex = exact_ground(eps, epsp, J, Lam, M)
            psi_var = np.concatenate([c1*pu, c2*pd])
            Eex[k]  = E0
            fid[k]  = abs(psi_ex @ psi_var)**2
            rel[k]  = abs(E - E0)/abs(E0) if abs(E0) > 1e-12 else np.nan
            leak[k] = max(np.sum(pu[-5:]**2), np.sum(pd[-5:]**2))
            cu[k], cd[k] = c1, c2
    return dict(X=X, Evar=Ev, Eex=Eex, fid=fid, rel=rel, leak=leak, cu=cu, cd=cd)


def _odd_row_f1(fix_tau0):
    E = np.empty(len(LAM_F1))
    w = None
    for k, lam in enumerate(LAM_F1):
        Lam = np.sqrt(2.0)*lam
        eps, epsp, J = sector_params('odd', lam, lam, eps1_val, eps2_val, T_F1, 0.0)
        E[k], w = minimize_sector('odd', eps, epsp, J, Lam, _Mof_Lam(Lam), fix_tau0, w)
    return E


def _even_row_f1(Delta, fix_tau0):
    E = np.empty(len(LAM_F1))
    w = None
    for k, lam in enumerate(LAM_F1):
        Lam = np.sqrt(2.0)*lam
        eps, epsp, J = sector_params('even', lam, lam, eps1_val, eps2_val, T_F1, Delta)
        E[k], w = minimize_sector('even', eps, epsp, J, Lam, _Mof_Lam(Lam), fix_tau0, w)
    return E


def compute_fig1(fix_tau0):
    _OPS_CACHE.clear()
    jobs = ([delayed(_odd_row_f1)(fix_tau0)]
            + [delayed(_even_row_f1)(D, fix_tau0) for D in DELTA_F1])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    E_odd  = res[0]
    E_even = np.array(res[1:])               # (n_Delta, n_lam)
    return E_even - E_odd[None, :]


def compute_fig2(fix_tau0):
    _OPS_CACHE.clear()
    keys = [(s, cv) for cv in cv_list for s in ('odd', 'even')]
    res  = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(sweep_Lambda)(s, cv, LAM_F2, fix_tau0) for s, cv in keys)
    return dict(zip(keys, res))


def compute_convergence():
    _OPS_CACHE.clear()
    keys = [(s, cv, N) for N in N_CONV for cv in cv_list for s in ('odd', 'even')]
    res  = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(sweep_Lambda)(s, cv, LAM_F2, False, M_fixed=N, with_exact=False)
        for s, cv, N in keys)
    return dict(zip(keys, res))


# ══════════════════════════════════════════════════════════════════
#  FIGURA 1: paridad fermiónica en el plano (λ, Δ)
# ══════════════════════════════════════════════════════════════════
def plot_fig1(dE, suffix, has_tex):
    print("δE: min=%.4f  max=%.4f" % (np.nanmin(dE), np.nanmax(dE)))
    lam_arr, Delta_arr = LAM_F1, DELTA_F1
    param_str = rf"$t={T_F1:.1f},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.4f},\ \omega_0={omega0}$"
    with plt.rc_context({'xtick.direction': 'out', 'ytick.direction': 'out'}):
        extent = [lam_arr[0], lam_arr[-1], Delta_arr[0], Delta_arr[-1]]
        xlabel = r'$g/t$'
        ylabel = r'$\Delta/t$'
        norm_p = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(np.tanh(dE), origin='lower', aspect='auto',
                       extent=extent, cmap='seismic', norm=norm_p)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\tanh(\delta E)$')

        ax.contour(lam_arr, Delta_arr, dE, levels=[0.0], colors='k',
                   linewidths=2.0, alpha=1.0, linestyles='--')
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if has_tex:
            txt_e = r'\begin{center} Even GS \\ $|\psi_e\rangle$ \end{center}'
            txt_o = r'\begin{center} Odd GS \\ $|\psi_o\rangle$ \end{center}'
        else:
            txt_e = 'Even GS\n' + r'$|\psi_e\rangle$'
            txt_o = 'Odd GS\n' + r'$|\psi_o\rangle$'

        ax.text(1.1, 1.1, txt_e, fontsize=20, color='0.15', ha='center', va='center',
                multialignment='center',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none',
                          pad=5, boxstyle='round,pad=0.25'))
        ax.text(0.4, 0.4, txt_o, fontsize=20, color='black', ha='center', va='center',
                multialignment='center',
                bbox=dict(facecolor='white', alpha=0.25, edgecolor='none',
                          pad=5, boxstyle='round,pad=0.25'))
        ax.text(0.15, 0.95, '(a)', transform=ax.transAxes, ha='right', va='top',
                fontweight='bold', color='white')

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'Parity_Splitting_aniso{suffix}.png'),
                    bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig)
    print(r'Parity Splitting $\delta E = E^e_0 - E^o_0$  (anisotrópico)')
    print(param_str)


# ══════════════════════════════════════════════════════════════════
#  FIGURA 2: parámetros variacionales
# ══════════════════════════════════════════════════════════════════
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


def plot_fig2(R, suffix, fix_tau0):
    """Una figura por sector y por J_α/ω0."""
    Lam_arr = LAM_F2
    ylim_by_cv = {0.5: 0.35, 1.0: 0.6, 1.5: 0.8}
    xlim_by_cv = {0.5: 4.0, 1.0: 4.5, 1.5: 5.0}

    # (símbolo, índice en x, estilo, color); x = (η↑, η↓, r↑, r↓, τ↑, τ↓)
    curves = {
        # sector par: índice 1 = |↑_e> = |11>, índice 2 = |↓_e> = |00>
        'even': [(r'\beta_1', 0, '-',  C_B), (r'\xi_1', 2, '--', C_B),
                 (r'\beta_2', 1, '-',  C_A), (r'\xi_2', 3, '--', C_A)],
        # sector impar (δ_o = 0, simétrico): rama |↑_o> = |10>
        'odd':  [(r'\alpha', 0, '-', 'green'), (r'r', 2, '--', 'green')],
    }
    if not fix_tau0:
        curves['even'] += [(r'\tau_1', 4, ':', C_B), (r'\tau_2', 5, ':', C_A)]
        curves['odd']  += [(r'\tau', 4, ':', 'green')]

    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        for cv in cv_list:
            pstr = rf"$\delta\lambda={0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f},\ J_\alpha/\omega_0={cv}$"
            for sector in ('even', 'odd'):
                X = R[(sector, cv)]['X']
                fig, ax = plt.subplots(figsize=(5, 4))
                for sym, i, ls, col in curves[sector]:
                    _plot_param(ax, Lam_arr, X[i], sym, ls, col)
                ax.axvline(0.5*np.sqrt(2), color='k', lw=1.0, ls=':', alpha=0.5)
                ax.set_xlim(Lam_arr[0], xlim_by_cv[cv])
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
                   ncol=len(handles), frameon=False, borderpad=0.6, handlelength=2.2,
                   labelspacing=0.4, columnspacing=1.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)      # espacio para los multiplicadores ×10^e
        plt.savefig(os.path.join(OUT_DIR, f'Fidelidad_y_error_relativo_por_sector{suffix}.png'),
                    bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig)

    print(r'Fidelity by sector, ' + rf'omega_0={omega0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f}')
    print(r'Relative error by sector, ' + rf'omega_0={omega0},\ \varepsilon_1={eps1_val},\ \varepsilon_2={eps2_val:.3f}')


# ══════════════════════════════════════════════════════════════════
#  FIGURA 4: convergencia del variacional con trisqueezing en N de Fock
# ══════════════════════════════════════════════════════════════════
def plot_convergence(conv):
    Lam_arr = LAM_F2
    LSN = {40: ':', 80: '-.', 120: '--', 170: '-'}
    with plt.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, cv, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
            for N in N_CONV:
                ax.plot(Lam_arr, conv[('odd',  cv, N)]['Evar'], LSN[N], color=C_ODD,  lw=1.4)
                ax.plot(Lam_arr, conv[('even', cv, N)]['Evar'], LSN[N], color=C_EVEN, lw=1.4)
            ax.set_xlabel(r'$\Lambda$')
            ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                           right=True, left=True)
            # zona libre a la izquierda, por encima de la leyenda de (a)
            ax.text(0.05, 0.80, lab, transform=ax.transAxes,
                    ha='left', va='top', fontsize=20)
            ax.text(0.05, 0.66, rf'$J_\alpha/\omega_0={cv/omega0}$', transform=ax.transAxes,
                    ha='left', va='top', fontsize=20)
            ax.set_xlim(Lam_arr[0], Lam_arr[-1])
        axes[0].set_ylabel(r'$E_{\mathrm{var}}/\omega_0$')

        # leyenda en dos columnas dentro de (a): N en la primera, odd/even en la segunda
        blank = mlines.Line2D([], [], color='none', label=' ')
        handles = [mlines.Line2D([], [], color='gray', ls=LSN[N], lw=1.4,
                                 label=rf'$N={N}$') for N in N_CONV]
        handles += [mlines.Line2D([], [], color=C_ODD,  lw=1.4, label='odd'),
                    mlines.Line2D([], [], color=C_EVEN, lw=1.4, label='even')]
        handles += [blank]*(len(N_CONV) - 2)
        axes[0].legend(handles=handles, loc='lower left', ncol=2, frameon=True,
                       edgecolor='black', framealpha=0.0, fancybox=False, fontsize=14,
                       borderpad=0.4, handlelength=2.0, labelspacing=0.25,
                       columnspacing=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'Convergencia_N_Fock_trisqueezing.png'),
                    bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  TESTS IMPRESOS
# ══════════════════════════════════════════════════════════════════
def _hr(title):
    print('\n' + '='*72 + f'\n {title}\n' + '='*72)


def tests_pre():
    _hr('TESTS PREVIOS')

    # T1: unitariedad (ortogonalidad) y realidad de exp(x G_k) truncados
    for M in (40, 170):
        ops = get_ops(M)
        for k, x in ((1, 2.0), (2, 0.5), (3, 0.05)):
            w, V, Vh = ops.eig[k]
            U = V @ np.diag(np.exp(-1j*x*w)) @ Vh
            err_u = np.linalg.norm(U.conj().T @ U - np.eye(ops.dim))
            err_i = np.max(np.abs(U.imag))
            print(f'[T1] M={M:3d}  G{k}  x={x:4.2f}  ||U^T U - 1|| = {err_u:.2e}   max|Im U| = {err_i:.2e}')

    # T2: funcional numérico (τ=0) vs funcional analítico, puntos aleatorios
    rng = np.random.default_rng(1234)
    ops = get_ops(170)
    for sector in ('odd', 'even'):
        for Lam in (1.0, 2.5, 4.0):
            lam1, lam2 = Lam_to_lam12(Lam, 0.0)
            eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, 1.0, 1.0)
            d = []
            for _ in range(100):
                x4 = np.array([rng.uniform(-Lam, Lam), rng.uniform(-Lam, Lam),
                               rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)])
                d.append(abs(energy_numeric(x4, eps, epsp, J, Lam, ops, True)
                             - energy_analytic(x4, eps, epsp, J, Lam)))
            print(f'[T2] {sector:4s}  Λ={Lam:3.1f}  max|E_num(τ=0) - E_analítica| = {max(d):.2e}')

    # T3: identidades con τ ≠ 0 (M = 170)
    Pi = (-1.0)**np.arange(ops.dim)
    for (eta, r, tau) in ((0.7, -0.3, 0.05), (-1.2, 0.4, -TAU_MAX), (2.0, -0.6, TAU_MAX)):
        v   = ops.state(eta, r, tau)
        vT  = ops.state(0.0, 0.0, tau)
        nT  = vT @ (ops.nvec*vT)
        mb  = v @ (ops.b @ v)
        mn  = v @ (ops.nvec*v)
        mf  = eta**2 + np.sinh(r)**2 + nT*np.cosh(2*r)
        off = np.sum(vT[np.arange(ops.dim) % 3 != 0]**2)
        par = np.linalg.norm(Pi*v - ops.state(-eta, r, -tau))
        print(f'[T3] (η,r,τ)=({eta:+.2f},{r:+.2f},{tau:+.2f})  '
              f'|<b>-η|={abs(mb-eta):.2e}  |<m>-fórmula|={abs(mn-mf):.2e}  '
              f'peso fuera de |3k>={off:.2e}  ||Π|τ,r,η> - |-τ,r,-η>||={par:.2e}  n_T={nT:.4f}')

    # T3b: convergencia de T(τ)|0> con el corte de Fock (n_T y peso en los 10 últimos niveles)
    for tau in (0.02, 0.05, TAU_MAX, 0.15, 0.30):
        line = []
        for M in N_CONV:
            o  = get_ops(M)
            vT = o.state(0.0, 0.0, tau)
            line.append(f'M={M}: n_T={vT @ (o.nvec*vT):.5f} cola={np.sum(vT[-10:]**2):.1e}')
        print(f'[T3b] τ={tau:4.2f}  ' + '  '.join(line))

    # T4: óptimo numérico (τ=0) vs óptimo analítico
    for sector in ('odd', 'even'):
        for Lam in (0.5, 1.5, 2.5, 3.5, 4.5):
            lam1, lam2 = Lam_to_lam12(Lam, 0.0)
            eps, epsp, J = sector_params(sector, lam1, lam2, eps1_val, eps2_val, 1.0, 1.0)
            En, _ = minimize_sector(sector, eps, epsp, J, Lam, _Mof_Lam(Lam), True)
            if abs(eps - epsp) < 1e-12:    # misma reducción simétrica que minimize_sector
                Ea, _ = _run_seeds(_energy_analytic_sym, (eps, epsp, J, Lam), Lam, 2, None, sector)
            else:
                Ea, _ = _run_seeds(energy_analytic, (eps, epsp, J, Lam), Lam, 4, None, sector)
            print(f'[T4] {sector:4s}  Λ={Lam:3.1f}  E_num={En:+.10f}  E_an={Ea:+.10f}  dif={En-Ea:+.2e}')


def tests_post(runs, conv=None):
    _hr('TESTS POSTERIORES')
    for tag, R in runs.items():
        for (sector, cv), d in R.items():
            gap = d['Evar'] - d['Eex']
            print(f'[T5] {tag:5s} {sector:4s} J/ω0={cv}:  min(E_var - E_ex) = {np.nanmin(gap):+.2e}'
                  f'   violaciones de la cota variacional = {int(np.sum(gap < -1e-9))}')
            print(f'[T7] {tag:5s} {sector:4s} J/ω0={cv}:  máx. peso en los 5 últimos niveles de Fock = {np.nanmax(d["leak"]):.2e}')
            if tag == 'trisq':
                nb = int(np.sum(np.abs(d['X'][4:6]) > TAU_MAX - 1e-6))
                print(f'[T7] {tag:5s} {sector:4s} J/ω0={cv}:  máx|τ| = {np.nanmax(np.abs(d["X"][4:6])):.4f}'
                      f'   puntos con τ en la cota (±{TAU_MAX}) = {nb}')
            if sector == 'odd':
                print(f'[T8] {tag:5s} odd  J/ω0={cv}:  máx|c↑ + c↓| (δ_o = 0) = {np.nanmax(np.abs(d["cu"] + d["cd"])):.2e}')
    if 'tau0' in runs and 'trisq' in runs:
        for key in runs['trisq']:
            diff = runs['trisq'][key]['Evar'] - runs['tau0'][key]['Evar']
            print(f'[T6] {key[0]:4s} J/ω0={key[1]}:  máx(E_DST - E_DS) = {np.nanmax(diff):+.2e}'
                  f'   violaciones = {int(np.sum(diff > 1e-9))}   mejora máx. = {-np.nanmin(diff):.2e}')
    if conv is not None:
        Nref = max(N_CONV)
        for cv in cv_list:
            for sector in ('odd', 'even'):
                ref = conv[(sector, cv, Nref)]['Evar']
                line = '  '.join(f'N={N}: {np.nanmax(np.abs(conv[(sector, cv, N)]["Evar"] - ref)):.2e}'
                                 for N in N_CONV if N != Nref)
                print(f'[T9] {sector:4s} J/ω0={cv}:  máx|E_N - E_{Nref}|  ->  {line}')


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
def run(fix_tau0, has_tex):
    tag    = 'tau0' if fix_tau0 else 'trisq'
    suffix = '_tau0' if fix_tau0 else ''
    _hr(f'CORRIDA: {"τ = 0 (prueba)" if fix_tau0 else "trisqueezing"}')

    print('Figura 1: splitting de paridad ...')
    dE = compute_fig1(fix_tau0)
    plot_fig1(dE, suffix, has_tex)

    print('Figuras 2 y 3: barridos en Λ ...')
    R = compute_fig2(fix_tau0)
    plot_fig2(R, suffix, fix_tau0)
    plot_fig3(R, suffix)
    print(f'Corrida {tag} lista.')
    return R


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    has_tex = setup_style()
    tests_pre()

    runs, conv = {}, None
    if MODE in ('tau0', 'all'):
        runs['tau0'] = run(True, has_tex)
    if MODE in ('trisq', 'all'):
        runs['trisq'] = run(False, has_tex)
        _hr('CONVERGENCIA EN N DE FOCK (trisqueezing)')
        conv = compute_convergence()
        plot_convergence(conv)

    tests_post(runs, conv)
    print(f'\nFiguras guardadas en: {OUT_DIR}')
