"""
Desplazamiento condicionado a una rama de pseudoespín, exacto vs variacional (P+AP)S.
El vector en Fock solo se construye con los parámetros óptimos, para Var_↑(x) y los tests.
Exacto: bloque ↑ del vector propio del estado fundamental.
Figuras en Figures/conditional/.
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


#  PARÁMETROS
N_JOBS  = -1
omega0  = 1.0
eps_tilde = 0.0                                   # ε̃_+
cv_list = [0.5, 1.0, 1.5]                         # J_α/ω0
LAM_ARR = np.linspace(0.01, 4.0*np.sqrt(2), 400)  # Λ
KAP_MAX = 1.0                                     # κ ∈ [0, KAP_MAX]
N_CHUNKS = 40                                     # tramos de Λ por cada J (paralelización)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', 'conditional')


#  ESTILO
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

#  FOCK (solo para el exacto y para los momentos del estado óptimo)
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
    """G1 = b†−b (D), G2 = (b²−b†²)/2 (S)."""

    def __init__(self, M):
        self.dim  = M + 1
        self.nvec = np.arange(self.dim, dtype=float)
        b  = np.diag(np.sqrt(np.arange(1, self.dim, dtype=float)), 1)
        bd = b.T
        self.b = b
        gens = {1: bd - b, 2: 0.5*(b @ b - bd @ bd)}
        self.eig = {}
        for k, G in gens.items():
            w, V = eigh(1j*G)
            self.eig[k] = (w, V, V.conj().T)

    def apply(self, k, x, v):
        if x == 0.0:
            return v
        w, V, Vh = self.eig[k]
        return (V @ (np.exp(-1j*x*w) * (Vh @ v))).real

    def state(self, eta, r):
        v = np.zeros(self.dim); v[0] = 1.0
        v = self.apply(2, r, v)
        return self.apply(1, eta, v)


_OPS = {}


def get_ops(M):
    if M not in _OPS:
        _OPS[M] = FockOps(M)
    return _OPS[M]


#  VARIACIONAL (P+AP)S (δ = 0, simétrico) Y EXACTO
def energy_paps(x, Lam, J):
    """Funcional cerrado E(η, r, κ) de (P+AP)S en δ = 0"""
    eta, r, kap = x
    F   = np.exp(-0.5*(Lam - eta)**2*np.exp(2.0*r))
    sh2 = np.sinh(r)**2
    G   = (0.25*Lam**2 - 0.25*(Lam - eta)**2*np.exp(4.0*r) + sh2)*F
    den = 1.0 + kap**2 + 2.0*kap*F
    if den < 1e-12:                 # κ = −1 con F = 1: |φ↑> = 0, no es un estado
        return 1e6
    num = (omega0*(0.25*eta**2 + (1.0 + kap**2)*sh2 + kap**2*(Lam - 0.5*eta)**2 + 2.0*kap*G)
           - J*abs(2.0*kap + (1.0 + kap**2)*F))
    return eps_tilde + num/den


def mean_b_paps(x, Lam):
    """<b†>_↑ cerrado de (P+AP)S"""
    eta, r, kap = x
    F = np.exp(-0.5*(Lam - eta)**2*np.exp(2.0*r))
    return (0.5*eta + kap**2*(Lam - 0.5*eta) + kap*Lam*F)/(1.0 + kap**2 + 2.0*kap*F)


def state_paps(x, Lam, ops):
    """(P+AP)S, rama ↑: N[|r, η/2> + κ|r, Λ − η/2>] (normalizado)."""
    eta, r, kap = x
    v = ops.state(0.5*eta, r) + kap*ops.state(Lam - 0.5*eta, r)
    return v/np.linalg.norm(v)


def energy_paps_fock(x, Lam, J, ops):
    """Misma energía evaluada en Fock (solo para el test D4)."""
    pu  = state_paps(x, Lam, ops)
    par = (-1.0)**ops.nvec
    T   = pu @ ops.apply(1, Lam, par*pu)
    return eps_tilde + omega0*(pu @ (ops.nvec*pu)) - J*abs(T)


def minimize_paps(Lam, J, x_warm=None):
    """(P+AP)S: minimiza E(η, r, κ) cerrado con κ ∈ [0, KAP_MAX]."""
    bnd = [(-1.0, 2.0*Lam + 1.0), (-1.0, 0.0), (0.0, KAP_MAX)]
    seeds = [(e, r, k) for e in (0.0, Lam, 2.0*Lam) for r in (0.0, -0.2) for k in (0.1, 0.6, 0.0)]
    if x_warm is None:
        seeds += [(e, r, k) for e in (0.5*Lam, 1.5*Lam) for r in (-0.1, -0.3) for k in (0.05, 0.3, 0.9)]
    else:
        seeds = [tuple(x_warm)] + seeds
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    bestE, bestx = np.inf, None
    for sd in seeds:
        res = minimize(energy_paps, np.clip(np.asarray(sd, float), lo, hi),
                       args=(Lam, J), method='L-BFGS-B', bounds=bnd,
                       options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
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
           ('b_ex', 'b_ex_dn', 'var_ex', 'E_ex', 'b_paps', 'var_paps', 'E_paps', 'kappa',
            'E_paps_fock', 'tail', 'b_paps_fock')}
    w_paps = None
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
        out['E_ex'][k] = w[0] + eps_tilde
        out['b_ex'][k], out['var_ex'][k] = conditional_moments(vec[:n1], ops.b)
        out['b_ex_dn'][k], _             = conditional_moments(vec[n1:], ops.b)
        # variacional (P+AP) S (funcional cerrado)
        E, x = minimize_paps(Lam, J, w_paps); w_paps = x
        out['E_paps'][k], out['kappa'][k] = E, x[2]
        pu = state_paps(x, Lam, ops)
        out['b_paps_fock'][k], out['var_paps'][k] = conditional_moments(pu, ops.b)
        out['b_paps'][k] = mean_b_paps(x, Lam)
        out['E_paps_fock'][k] = energy_paps_fock(x, Lam, J, ops)
        out['tail'][k] = np.sum(pu[-5:]**2)
    return out


# ══════════════════════════════════════════════════════════════════
#  FIGURAS
# ══════════════════════════════════════════════════════════════════
C_EX   = 'black'
C_PAPS = 'tab:green'


def _legend(ax):
    h = [mlines.Line2D([], [], color=C_EX,   ls='-',  lw=1.6, label='exact'),
         mlines.Line2D([], [], color=C_PAPS, ls='-.', lw=1.6, label=r'$(\mathrm{P}+\mathrm{AP})\hat S$')]
    ax.legend(handles=h, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
              borderpad=0.5, handlelength=1.8, ncol=1, loc='upper right', bbox_to_anchor=(1.0, 0.86))   # panel (a), una columna, bajo la etiqueta


def plot_mean(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, J, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
        d = R[J]
        ax.plot(LAM_ARR, d['b_ex'],   '-',  color=C_EX,   lw=1.6)
        ax.plot(LAM_ARR, d['b_paps'], '-.', color=C_PAPS, lw=1.6)
        ax.set_xlabel(r'$\Lambda$')
        ax.set_xlim(LAM_ARR[0], LAM_ARR[-1])
        ymax = max(np.nanmax(d['b_ex']), np.nanmax(d['b_paps']))
        ax.set_ylim(0.0, 1.35*ymax)                       # espacio para la etiqueta y la leyenda
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.text(0.05, 0.95, lab + rf'  $J_\alpha/\omega_0={J}$', transform=ax.transAxes,
                ha='left', va='top')
    axes[0].set_ylabel(r'$\langle\hat b^\dagger\rangle_\uparrow$')
    plt.tight_layout()
    _legend(axes[0])
    fig.savefig(os.path.join(OUT_DIR, 'conditional_displacement.png'),
                bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def plot_var(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, J, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
        d = R[J]
        ax.plot(LAM_ARR, d['var_ex'],   '-',  color=C_EX,   lw=1.6)
        ax.plot(LAM_ARR, d['var_paps'], '-.', color=C_PAPS, lw=1.6)
        ax.axhline(0.5, color='gray', lw=0.8, ls='-.')
        ax.set_xlabel(r'$\Lambda$')
        ax.set_xlim(LAM_ARR[0], LAM_ARR[-1])
        span = max(np.nanmax(d['var_ex']), np.nanmax(d['var_paps'])) - 0.5
        ax.set_ylim(0.5 - 0.03*span, 0.5 + 1.6*span)     # espacio para la etiqueta y la leyenda
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.text(0.05, 0.95, lab + rf'  $J_\alpha/\omega_0={J}$', transform=ax.transAxes,
                ha='left', va='top')
    axes[0].set_ylabel(r'$\mathrm{Var}_\uparrow(\hat x)$')
    plt.tight_layout()
    _legend(axes[0])
    fig.savefig(os.path.join(OUT_DIR, 'conditional_variance.png'),
                bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def _plot_relerr(R, key, ylabel, fname):
    """Error absoluto |X_var − X_ex| de (P+AP)S, misma estructura 1×3."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, J, lab in zip(axes, cv_list, ['(a)', '(b)', '(c)']):
        d = R[J]
        e = np.abs(d[f'{key}_paps'] - d[f'{key}_ex'])
        ax.plot(LAM_ARR, e, ls='-.', color=C_PAPS, lw=1.6)
        ax.set_ylim(0.0, 1.6*np.nanmax(e))              # espacio para la etiqueta y la leyenda
        ax.set_xlabel(r'$\Lambda$')
        ax.set_xlim(LAM_ARR[0], LAM_ARR[-1])
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.text(0.05, 0.95, lab + rf'  $J_\alpha/\omega_0={J}$', transform=ax.transAxes,
                ha='left', va='top')
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    _legend_var(axes[0])
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def _legend_var(ax):
    """Leyenda del ansatz variacional (sin el exacto, que es la referencia)."""
    h = [mlines.Line2D([], [], color=C_PAPS, ls='-.', lw=1.6, label=r'$(\mathrm{P}+\mathrm{AP})\hat S$')]
    ax.legend(handles=h, frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
              borderpad=0.5, handlelength=1.8, ncol=1, loc='upper right', bbox_to_anchor=(1.0, 0.86))   # panel (a), una columna, bajo la etiqueta


def plot_mean_relerr(R):
    _plot_relerr(R, 'b',
                 r'$|\langle\hat b^\dagger\rangle_\uparrow^{\mathrm{var}}-\langle\hat b^\dagger\rangle_\uparrow^{\mathrm{ex}}|$',
                 'conditional_displacement_abserr.png')


def plot_var_relerr(R):
    _plot_relerr(R, 'var',
                 r'$|\mathrm{Var}_\uparrow^{\mathrm{var}}-\mathrm{Var}_\uparrow^{\mathrm{ex}}|$',
                 'conditional_variance_abserr.png')


#  EJECUCIÓN
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
        print(f'[D2] J/ω0={J}:  min(E_PAPS − E_ex) = {np.nanmin(d["E_paps"] - d["E_ex"]):.2e}'
              f'   violaciones E_PAPS < E_ex: {int(np.sum(d["E_paps"] - d["E_ex"] < -1e-9))}')
        jump_ex   = np.nanmax(np.abs(np.diff(d['b_ex'])))
        jump_paps = np.nanmax(np.abs(np.diff(d['b_paps'])))
        print(f'[D3] J/ω0={J}:  máx|Δ<b†>_↑| entre puntos vecinos (ΔΛ={LAM_ARR[1]-LAM_ARR[0]:.3f}):'
              f' exacto={jump_ex:.3e}  (P+AP)S={jump_paps:.3e}'
              f'   (Λ del salto (P+AP)S={LAM_ARR[np.nanargmax(np.abs(np.diff(d["b_paps"])))]:.3f})')
        print(f'[D4] J/ω0={J}:  máx|E_cerrado − E_Fock| en el óptimo = {np.nanmax(np.abs(d["E_paps"] - d["E_paps_fock"])):.2e}'
              f'   máx. peso en los 5 últimos niveles de Fock = {np.nanmax(d["tail"]):.2e}')
        print(f'[D5] J/ω0={J}:  (P+AP)S  κ ∈ [{np.nanmin(d["kappa"]):.3f}, {np.nanmax(d["kappa"]):.3f}]'
              f'   puntos con |κ| en la cota ({KAP_MAX}) = {int(np.sum(np.abs(d["kappa"]) > KAP_MAX - 1e-6))}')
        low = d['kappa'] < 1e-6
        print(f'[D5] J/ω0={J}:  puntos con κ en la cota inferior (0) = {int(np.sum(low))}'
              + (f'   Λ = {np.array2string(LAM_ARR[low], precision=3, separator=", ")}' if low.any() else ''))
        print(f'[D6] J/ω0={J}:  máx|<b†>_↑ cerrado − <b†>_↑ Fock| = {np.nanmax(np.abs(d["b_paps"] - d["b_paps_fock"])):.2e}')

    plot_mean(R)
    plot_var(R)
    plot_mean_relerr(R)
    plot_var_relerr(R)
    print(f'\nFiguras guardadas en: {OUT_DIR}')
