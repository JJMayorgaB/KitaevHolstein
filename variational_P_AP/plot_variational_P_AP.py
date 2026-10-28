# -*- coding: utf-8 -*-
"""
Figuras del ansatz (P+AP)S (parámetros óptimos; fidelidad y error relativo por sector), a partir
de los datos que guarda variational_P_AP.py en Figures/anstaz/data/ (no recalcula nada).
Las figuras (sufijo _PAPS) se guardan en Figures/anstaz/ con los mismos nombres de antes.
Uso:  python variational_P_AP.py        (datos, una vez)
      python plot_variational_P_AP.py   (figuras, cada vez que se cambie el estilo)
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

warnings.filterwarnings('ignore')

# parámetros de la corrida (los mismos con los que se generaron los datos)
from variational_P_AP import (OUT_DIR, DATA_DIR, LAM_F2, cv_list, omega0, eps1_val, eps2_val,
                              X_NAMES, SCALARS)


#  LECTURA DE LOS DATOS
def load_fig2():
    R = {}
    for cv in cv_list:
        for sector in ('odd', 'even'):
            path = os.path.join(DATA_DIR, f'sweep_{sector}_cv{cv}.txt')
            with open(path, encoding='utf-8') as f:
                names = [ln for ln in f if ln.startswith('#')][-1][1:].split()
            A = np.loadtxt(path, ndmin=2)
            c = {k: A[:, i] for i, k in enumerate(names)}
            d = {k: c[k] for k in SCALARS}
            d['X']    = np.array([c[k] for k in X_NAMES])
            d['Xraw'] = np.array([c[k + '_raw'] for k in X_NAMES])
            R[(sector, cv)] = d
    return R


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


#  EJECUCIÓN
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    has_tex = setup_style()
    R = load_fig2()
    plot_fig2(R, '_PAPS')
    plot_fig3(R, '_PAPS')
    print(f'\nFiguras guardadas en: {OUT_DIR}')
