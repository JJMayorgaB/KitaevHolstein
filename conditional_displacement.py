# -*- coding: utf-8 -*-
"""
Desplazamiento condicionado a una rama de pseudoespín, exacto vs variacional.

Bloque en detuning cero (δ_α = 0), base {|↑>, |↓>}:
    H = [[ ω0 b†b ,  J D(Λ) ],
         [ J D(Λ)† , ω0 b†b ]]
Observables en la rama ↑:
    <b†>_↑    = <φ↑| b† |φ↑> / <φ↑|φ↑>
    Var_↑(x)  = <x²>_↑ − <x>_↑²,   x = (b + b†)/√2
Variacional: |φ↑> = D(η/2) S(r) T(τ)|0>  ->  <b†>_↑ = η/2 (ansatz D S y D S T).
Variacional (P+AP)S: |φ↑> = N[|r, η/2> + κ|r, Λ − η/2>],  |φ↓> = P|φ↑>  (paridad de Rabi).
Variacional (P+AP)ST: |φ↑> = N[|τ, r, η/2> + κ|−τ, r, Λ − η/2>] (el AP hereda τ↓ = −τ).
Exacto: bloque ↑ del vector propio del estado fundamental.
Figuras en Resultados_conditional_displacement/.
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
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import factorial, eval_genlaguerre
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
N_JOBS  = -1
omega0  = 1.0
cv_list = [0.5, 1.0, 1.5]                         # J_α/ω0
LAM_ARR = np.linspace(0.01, 4.0*np.sqrt(2), 400)  # Λ
TAU_MAX = 0.08
N_CHUNKS = 40                                     # tramos de Λ por cada J (paralelización)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Resultados_conditional_displacement')


# ══════════════════════════════════════════════════════════════════
#  ESTILO
# ══════════════════════════════════════════════════════════════════
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
    plt.rcParams.update({
        'text.usetex': HAS_LATEX,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
        'font.size': 20,
        'axes.labelsize': 22,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 16,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.1,
        'axes.axisbelow': True,
    })
    return HAS_LATEX


# ══════════════════════════════════════════════════════════════════
#  FOCK
# ══════════════════════════════════════════════════════════════════
def disp_matrix(M_ph, alpha):
    """<m'|D(alpha)|m>, alpha real"""
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


class FockOps:
    """G1 = b†−b (D), G2 = (b²−b†²)/2 (S), G3 = b†³−b³ (T)."""

    def __init__(self, M):
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
        v = np.zeros(self.dim); v[0] = 1.0
        v = self.apply(3, tau, v)
        v = self.apply(2, r, v)
        return self.apply(1, eta, v)


_OPS = {}


def get_ops(M):
    if M not in _OPS:
        _OPS[M] = FockOps(M)
    return _OPS[M]


# ══════════════════════════════════════════════════════════════════
#  VARIACIONAL (δ = 0, simétrico) Y EXACTO
# ══════════════════════════════════════════════════════════════════
def energy_var(x, Lam, J, ops, fix_tau0):
    eta, r = x[0], x[1]
    tau = 0.0 if fix_tau0 else x[2]
    pu = ops.state(+0.5*eta, r, +tau)
    pd = ops.state(-0.5*eta, r, -tau)
    return omega0*(pu @ (ops.nvec*pu)) - J*(pu @ ops.apply(1, Lam, pd))


def minimize_var(Lam, J, M, fix_tau0, x_warm=None):
    ops  = get_ops(M)
    base = [(0.0, 0.0), (0.0, -0.1), (Lam, 0.0), (Lam, -0.25), (0.5*Lam, -0.1)]
    bnd  = [(-Lam - 1.0, Lam + 1.0), (-1.0, 0.0)]
    if fix_tau0:
        seeds, n_par = list(base), 2
    else:
        seeds = [sd + (0.0,) for sd in base]
        seeds += [sd + (tt,) for sd in [(Lam, -0.25), (0.5*Lam, -0.1)] for tt in (0.02, -0.02)]
        bnd += [(-TAU_MAX, TAU_MAX)]
        n_par = 3
    if x_warm is not None:
        seeds = [tuple(x_warm[:n_par])] + seeds
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        res = minimize(energy_var, np.clip(np.asarray(sd, float), lo, hi),
                       args=(Lam, J, ops, fix_tau0), method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    if fix_tau0:
        bestx = np.concatenate([bestx, [0.0]])
    return bestE, bestx


def state_paps(x, Lam, ops):
    """(P+AP)S, rama ↑: N[|r, η/2> + κ|r, Λ − η/2>] (normalizado).
    Si x tiene 4 entradas, (P+AP)ST: N[|τ, r, η/2> + κ|−τ, r, Λ − η/2>]."""
    eta, r, kap = x[0], x[1], x[2]
    tau = x[3] if len(x) > 3 else 0.0
    v = ops.state(0.5*eta, r, tau) + kap*ops.state(Lam - 0.5*eta, r, -tau)
    return v/np.linalg.norm(v)


def energy_paps(x, Lam, J, ops):
    """δ = 0, simetría de Rabi: |φ↓> = P|φ↑>, c↓ = −c↑."""
    pu  = state_paps(x, Lam, ops)
    par = (-1.0)**ops.nvec
    return omega0*(pu @ (ops.nvec*pu)) - J*(pu @ ops.apply(1, Lam, par*pu))


def minimize_paps(Lam, J, M, x_warm=None, with_tau=False, x_s=None):
    """with_tau=False: (P+AP)S (η, r, κ);  with_tau=True: (P+AP)ST (η, r, κ, τ).
    x_s: óptimo de (P+AP)S en el mismo Λ, usado como semilla para (P+AP)ST."""
    ops = get_ops(M)
    bnd = [(-1.0, 2.0*Lam + 1.0), (-1.0, 0.0), (0.0, 1.0)]
    seeds = [(e, r, k) for e in (0.0, Lam, 2.0*Lam) for r in (0.0, -0.2) for k in (0.1, 0.6)]
    if x_warm is None:
        seeds += [(e, r, k) for e in (0.5*Lam, 1.5*Lam) for r in (-0.1, -0.3) for k in (0.05, 0.3, 0.9)]
    if with_tau:
        bnd += [(-TAU_MAX, TAU_MAX)]
        seeds = [sd + (t,) for sd in seeds for t in (0.0, 0.03)]
        if x_s is not None:
            seeds = [tuple(x_s[:3]) + (t,) for t in (0.0, 0.02, -0.02)] + seeds
    if x_warm is not None:
        seeds = [tuple(x_warm)] + seeds
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        res = minimize(energy_paps, np.clip(np.asarray(sd, float), lo, hi),
                       args=(Lam, J, ops), method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
        if res.fun < bestE:
            bestE, bestx = res.fun, res.x
    return bestE, bestx


def conditional_moments(phi, b):
    """<b†>, Var(x) condicionados al bloque phi (vector real, no normalizado)."""
    nrm = phi @ phi
    bphi = b @ phi
    mean_b = (phi @ bphi)/nrm                                   # = <b> = <b†> (real)
    x = (b + b.T)/np.sqrt(2.0)
    xphi = x @ phi
    mean_x = (phi @ xphi)/nrm
    var_x  = (xphi @ xphi)/nrm - mean_x**2
    return mean_b, var_x


def sweep(J, Lam_arr=LAM_ARR):
    """Barrido con continuación (arranque en caliente) sobre Lam_arr."""
    n = len(Lam_arr)
    out = {k: np.full(n, np.nan) for k in
           ('b_ex', 'b_ex_dn', 'var_ex', 'b_ds', 'var_ds', 'b_dst', 'var_dst',
            'tau', 'E_ex', 'E_ds', 'E_dst', 'b_paps', 'var_paps', 'E_paps', 'kappa',
            'b_papst', 'var_papst', 'E_papst', 'tau_papst')}
    w_ds = w_dst = w_paps = w_papst = None
    for k, Lam in enumerate(Lam_arr):
        M   = _Mof_Lam(Lam)
        ops = get_ops(M)
        n1  = M + 1
        # exacto
        DL = disp_matrix(M, Lam)
        nd = np.arange(n1)*omega0
        H  = np.zeros((2*n1, 2*n1))
        H[:n1, :n1] = np.diag(nd); H[n1:, n1:] = np.diag(nd)
        H[:n1, n1:] = J*DL;        H[n1:, :n1] = J*DL.T
        w, V = eigh(H)
        vec  = V[:, 0]
        out['E_ex'][k] = w[0]
        out['b_ex'][k], out['var_ex'][k] = conditional_moments(vec[:n1], ops.b)
        out['b_ex_dn'][k], _             = conditional_moments(vec[n1:], ops.b)
        # variacional D S |0>
        E, x = minimize_var(Lam, J, M, True, w_ds); w_ds = x
        out['E_ds'][k] = E
        out['b_ds'][k], out['var_ds'][k] = conditional_moments(ops.state(0.5*x[0], x[1], 0.0), ops.b)
        # variacional D S T |0>
        E, x = minimize_var(Lam, J, M, False, w_dst); w_dst = x
        out['E_dst'][k], out['tau'][k] = E, x[2]
        out['b_dst'][k], out['var_dst'][k] = conditional_moments(ops.state(0.5*x[0], x[1], x[2]), ops.b)
        # variacional (P+AP) S
        E, x = minimize_paps(Lam, J, M, w_paps); w_paps = x
        out['E_paps'][k], out['kappa'][k] = E, x[2]
        out['b_paps'][k], out['var_paps'][k] = conditional_moments(state_paps(x, Lam, ops), ops.b)
        # variacional (P+AP) S T
        E, x = minimize_paps(Lam, J, M, w_papst, with_tau=True, x_s=w_paps); w_papst = x
        out['E_papst'][k], out['tau_papst'][k] = E, x[3]
        out['b_papst'][k], out['var_papst'][k] = conditional_moments(state_paps(x, Lam, ops), ops.b)
    return out


# ══════════════════════════════════════════════════════════════════
#  FIGURAS
# ══════════════════════════════════════════════════════════════════
C_EX, C_DS, C_DST = 'black', plt.cm.seismic(0.88), plt.cm.seismic(0.12)
C_PAPS = 'tab:green'
C_PAPST = 'tab:orange'


def _legend(fig):
    h = [mlines.Line2D([], [], color=C_EX,  ls='-',  lw=1.6, label='exact'),
         mlines.Line2D([], [], color=C_DS,  ls='--', lw=1.6, label=r'$\hat D\hat S|0\rangle$'),
         mlines.Line2D([], [], color=C_DST, ls=':',  lw=1.8, label=r'$\hat D\hat S\hat T|0\rangle$'),
         mlines.Line2D([], [], color=C_PAPS, ls='-.', lw=1.6, label=r'$(\mathrm{P}+\mathrm{AP})\hat S$'),
         mlines.Line2D([], [], color=C_PAPST, ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.6,
                       label=r'$(\mathrm{P}+\mathrm{AP})\hat S\hat T$')]
    fig.legend(handles=h, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
               borderpad=0.5, handlelength=2.4, columnspacing=1.6, ncol=len(h),
               loc='lower center', bbox_to_anchor=(0.5, 1.0))   # fuera de los paneles, una fila


def plot_mean(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, J, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
        d = R[J]
        ax.plot(LAM_ARR, d['b_ex'],  '-',  color=C_EX,  lw=1.6)
        ax.plot(LAM_ARR, d['b_ds'],  '--', color=C_DS,  lw=1.6)
        ax.plot(LAM_ARR, d['b_dst'], ':',  color=C_DST, lw=1.8)
        ax.plot(LAM_ARR, d['b_paps'], '-.', color=C_PAPS, lw=1.6)
        ax.plot(LAM_ARR, d['b_papst'], ls=(0, (6, 2, 1, 2, 1, 2)), color=C_PAPST, lw=1.6)
        ax.set_xlabel(r'$\Lambda$')
        ax.set_xlim(LAM_ARR[0], LAM_ARR[-1])
        ymax = max(np.nanmax(d['b_ex']), np.nanmax(d['b_ds']), np.nanmax(d['b_dst']), np.nanmax(d['b_paps']),
                   np.nanmax(d['b_papst']))
        ax.set_ylim(0.0, 1.25*ymax)                       # espacio para la etiqueta del panel
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.text(0.05, 0.95, lab + rf'  $J_\alpha/\omega_0={J}$', transform=ax.transAxes,
                ha='left', va='top')
    axes[0].set_ylabel(r'$\langle\hat b^\dagger\rangle_\uparrow$')
    plt.tight_layout()
    _legend(fig)
    fig.savefig(os.path.join(OUT_DIR, 'conditional_displacement.png'),
                bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def plot_var(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, J, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
        d = R[J]
        ax.plot(LAM_ARR, d['var_ex'],  '-',  color=C_EX,  lw=1.6)
        ax.plot(LAM_ARR, d['var_ds'],  '--', color=C_DS,  lw=1.6)
        ax.plot(LAM_ARR, d['var_dst'], ':',  color=C_DST, lw=1.8)
        ax.plot(LAM_ARR, d['var_paps'], '-.', color=C_PAPS, lw=1.6)
        ax.plot(LAM_ARR, d['var_papst'], ls=(0, (6, 2, 1, 2, 1, 2)), color=C_PAPST, lw=1.6)
        ax.axhline(0.5, color='gray', lw=0.8, ls='-.')
        ax.set_xlabel(r'$\Lambda$')
        ax.set_xlim(LAM_ARR[0], LAM_ARR[-1])
        span = max(np.nanmax(d['var_ex']), np.nanmax(d['var_ds']), np.nanmax(d['var_dst']),
                   np.nanmax(d['var_paps']), np.nanmax(d['var_papst'])) - 0.5
        ax.set_ylim(0.5 - 0.03*span, 0.5 + 1.3*span)     # espacio para la etiqueta del panel
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.text(0.05, 0.95, lab + rf'  $J_\alpha/\omega_0={J}$', transform=ax.transAxes,
                ha='left', va='top')
    axes[0].set_ylabel(r'$\mathrm{Var}_\uparrow(\hat x)$')
    plt.tight_layout()
    _legend(fig)
    fig.savefig(os.path.join(OUT_DIR, 'conditional_variance.png'),
                bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def _plot_relerr(R, key, ylabel, fname):
    """Error absoluto |X_var − X_ex| de cada ansatz, misma estructura 1×3."""
    LS_PAPST = (0, (6, 2, 1, 2, 1, 2))
    styles = [('ds', '--', C_DS, 1.6), ('dst', ':', C_DST, 1.8),
              ('paps', '-.', C_PAPS, 1.6), ('papst', LS_PAPST, C_PAPST, 1.6)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, J, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
        d = R[J]
        errs = []
        for name, ls, c, lw in styles:
            e = np.abs(d[f'{key}_{name}'] - d[f'{key}_ex'])
            errs.append(e)
            ax.plot(LAM_ARR, e, ls=ls, color=c, lw=lw)
        hi = np.nanmax(np.concatenate(errs))
        ax.set_ylim(0.0, 1.25*hi)                       # espacio para la etiqueta del panel
        ax.set_xlabel(r'$\Lambda$')
        ax.set_xlim(LAM_ARR[0], LAM_ARR[-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.text(0.05, 0.95, lab + rf'  $J_\alpha/\omega_0={J}$', transform=ax.transAxes,
                ha='left', va='top')
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    _legend_var(fig)
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def _legend_var(fig):
    """Leyenda de los cuatro ansatz variacionales (sin el exacto, que es la referencia)."""
    h = [mlines.Line2D([], [], color=C_DS,  ls='--', lw=1.6, label=r'$\hat D\hat S|0\rangle$'),
         mlines.Line2D([], [], color=C_DST, ls=':',  lw=1.8, label=r'$\hat D\hat S\hat T|0\rangle$'),
         mlines.Line2D([], [], color=C_PAPS, ls='-.', lw=1.6, label=r'$(\mathrm{P}+\mathrm{AP})\hat S$'),
         mlines.Line2D([], [], color=C_PAPST, ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.6,
                       label=r'$(\mathrm{P}+\mathrm{AP})\hat S\hat T$')]
    fig.legend(handles=h, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
               borderpad=0.5, handlelength=2.4, columnspacing=1.6, ncol=len(h),
               loc='lower center', bbox_to_anchor=(0.5, 1.0))


def plot_mean_relerr(R):
    _plot_relerr(R, 'b',
                 r'$|\langle\hat b^\dagger\rangle_\uparrow^{\mathrm{var}}-\langle\hat b^\dagger\rangle_\uparrow^{\mathrm{ex}}|$',
                 'conditional_displacement_abserr.png')


def plot_var_relerr(R):
    _plot_relerr(R, 'var',
                 r'$|\mathrm{Var}_\uparrow^{\mathrm{var}}-\mathrm{Var}_\uparrow^{\mathrm{ex}}|$',
                 'conditional_variance_abserr.png')


# ══════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    # cada J se parte en N_CHUNKS tramos de Λ independientes (cada tramo arranca con todas
    # las semillas y sigue con continuación dentro del tramo); los tramos se unen al final
    chunks = np.array_split(np.arange(len(LAM_ARR)), N_CHUNKS)
    tasks  = [(J, idx) for J in cv_list for idx in chunks]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(sweep)(J, LAM_ARR[idx]) for J, idx in tasks)
    R = {}
    for J in cv_list:
        parts = [r for (Jt, _), r in zip(tasks, res) if Jt == J]
        R[J] = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    print('\nTESTS')
    for J in cv_list:
        d = R[J]
        print(f'[D1] J/ω0={J}:  máx|<b>_↑ + <b>_↓| exacto = {np.nanmax(np.abs(d["b_ex"] + d["b_ex_dn"])):.2e}')
        print(f'[D2] J/ω0={J}:  violaciones E_DS < E_ex: {int(np.sum(d["E_ds"] - d["E_ex"] < -1e-9))}'
              f'   E_DST < E_ex: {int(np.sum(d["E_dst"] - d["E_ex"] < -1e-9))}'
              f'   E_DST > E_DS: {int(np.sum(d["E_dst"] - d["E_ds"] > 1e-9))}'
              f'   E_PAPS < E_ex: {int(np.sum(d["E_paps"] - d["E_ex"] < -1e-9))}'
              f'   E_PAPS > E_DS: {int(np.sum(d["E_paps"] - d["E_ds"] > 1e-9))}'
              f'   E_PAPST < E_ex: {int(np.sum(d["E_papst"] - d["E_ex"] < -1e-9))}'
              f'   E_PAPST > E_PAPS: {int(np.sum(d["E_papst"] - d["E_paps"] > 1e-9))}')
        jump_ex  = np.nanmax(np.abs(np.diff(d['b_ex'])))
        jump_ds  = np.nanmax(np.abs(np.diff(d['b_ds'])))
        jump_dst = np.nanmax(np.abs(np.diff(d['b_dst'])))
        jump_paps = np.nanmax(np.abs(np.diff(d['b_paps'])))
        jump_papst = np.nanmax(np.abs(np.diff(d['b_papst'])))
        print(f'[D3] J/ω0={J}:  máx|Δ<b†>_↑| entre puntos vecinos (ΔΛ={LAM_ARR[1]-LAM_ARR[0]:.3f}):'
              f' exacto={jump_ex:.3e}  DS={jump_ds:.3e}  DST={jump_dst:.3e}  (P+AP)S={jump_paps:.3e}  (P+AP)ST={jump_papst:.3e}'
              f'   (Λ del salto DS={LAM_ARR[np.nanargmax(np.abs(np.diff(d["b_ds"])))]:.3f},'
              f' DST={LAM_ARR[np.nanargmax(np.abs(np.diff(d["b_dst"])))]:.3f})')
        print(f'[D4] J/ω0={J}:  máx|τ| = {np.nanmax(np.abs(d["tau"])):.4f}')
        print(f'[D5] J/ω0={J}:  (P+AP)S  κ ∈ [{np.nanmin(d["kappa"]):.3f}, {np.nanmax(d["kappa"]):.3f}]'
              f'   máx(E_DS − E_PAPS) = {np.nanmax(d["E_ds"] - d["E_paps"]):.3e}')
        print(f'[D6] J/ω0={J}:  (P+AP)ST  máx|τ| = {np.nanmax(np.abs(d["tau_papst"])):.4f}'
              f'   máx(E_PAPS − E_PAPST) = {np.nanmax(d["E_paps"] - d["E_papst"]):.3e}')

    plot_mean(R)
    plot_var(R)
    plot_mean_relerr(R)
    plot_var_relerr(R)
    print(f'\nFiguras guardadas en: {OUT_DIR}')
