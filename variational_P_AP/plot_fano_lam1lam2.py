# -*- coding: utf-8 -*-
"""
FIGURA 3 del paper (respuesta fotónica en el plano (λ1, λ2)) a partir de los datos que genera
fano_lam1lam2.py en Figures/fig_fano/data/. Para cada (ε1, ε2):

    PAPS/  : (a) mapa 3D de F1 con los cortes encima; (b) cortes a λ1 = 0.5, 1.0, 1.25 de
             F1 (sólido, eje izquierdo) y (ΔX1)² (punteado, eje derecho), con el ansatz (P+AP)S
    exact/ : la misma figura con la diagonalización exacta del bloque
    diff/  : heatmaps de |F1_PAPS − F1_exact| y |(ΔX1)²_PAPS − (ΔX1)²_exact| en (λ1, λ2)

Mismo estilo que la celda del notebook de la Fig. 3, restringido al primer cuadrante. Las rectas
discontinuas marcan λ_i = √(ε_i/ω0) (ε̃_i = 0), solo si ε_i > 0; con Δ = t, sobre ellas los dos
sectores de paridad son exactamente degenerados. Los ejes de (b) se escalan solos con la
notación ×10^k + offset de la figura original.

Uso:  python plot_fano_lam1lam2.py            (los cuatro (ε1, ε2) de EPS_SETS)
      python plot_fano_lam1lam2.py e1 e2      (solo ese)
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fano_lam1lam2 import (LAM, LAM2_CUT, CUT_LAM1, EPS_SETS, f_map, f_cut, f_val, _tag)
from variational_P_AP import omega0

warnings.filterwarnings('ignore')

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

OUT_DIR = os.path.join(HERE, 'Figures', 'fig_fano')

C_A = plt.cm.seismic(0.05)   # azul
C_B = plt.cm.seismic(0.85)   # rojo
C_C = plt.cm.seismic(0.30)   # rojo medio
CUT_COLORS = [C_A, C_B, C_C]
SCI_FS = 18                  # tamaño de los multiplicadores de (b)
DIFF_FLOOR = 1e-16           # piso de la escala logarítmica de diff/


def _res_lines(e):
    """λ con ε̃ = 0 en [0, 2] (solo si ε > 0)."""
    return [np.sqrt(e/omega0)] if e > 0 and np.sqrt(e/omega0) <= LAM[-1] else []


def _offset_axis(ax, y, side):
    """Eje con la notación de la figura original: valor mostrado = (y − off)/10^k."""
    lo, hi = np.nanmin(y), np.nanmax(y)
    off = np.floor(10.0*lo + 1e-3)/10.0
    rng = max(hi - off, 1e-12)
    k   = int(np.floor(np.log10(rng)))
    scl = 10.0**k
    top = rng/scl
    step = 0.5 if top <= 2.0 else (1.0 if top <= 5.0 else 2.0)
    ax.set_ylim(off - 0.06*rng, hi + 0.06*rng)
    ax.set_yticks(off + scl*np.arange(0.0, top + 1e-9, step))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{(v - off)/scl:.1f}'))
    lab = (rf'\times10^{{{k}}}' if k != 0 else '') + (rf'+{off:g}' if off != 0 else '')
    if lab:
        x, ha = (0.0, 'left') if side == 'left' else (1.0, 'right')
        ax.text(x, 1.01, rf'${lab}$', transform=ax.transAxes, ha=ha, va='bottom', fontsize=SCI_FS)


def fig_fano(e1, e2, method):
    F1  = np.loadtxt(f_map('F1', method, e1, e2))                 # filas = λ2, columnas = λ1
    Fc  = np.loadtxt(f_cut('F1', method, e1, e2))                 # filas = λ1 del corte
    Dc  = np.loadtxt(f_cut('DX1', method, e1, e2))
    LAM1, LAM2 = np.meshgrid(LAM, LAM)

    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1], wspace=0.45)

    # --- (a) mapa 3D de F1 con los cortes encima ---
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax_a.plot_surface(LAM1, LAM2, F1, cmap='inferno', linewidth=0, antialiased=True, alpha=0.9)
    cb = fig.colorbar(surf, ax=ax_a, fraction=0.03, pad=0.08, shrink=0.6)
    t_lo, t_hi = np.ceil(10.0*F1.min() - 1e-6)/10.0, np.floor(10.0*F1.max() + 1e-6)/10.0
    if t_hi > t_lo:
        cb.set_ticks([t_lo, t_hi])
        cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    else:
        cb.set_ticks([F1.min(), F1.max()])
        cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cb.ax.set_title(r'$F_1$', pad=8)
    for i, v in enumerate(CUT_LAM1):
        ax_a.plot(np.full_like(LAM2_CUT, v), LAM2_CUT, Fc[i], '-', color=CUT_COLORS[i], lw=2.0)
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_yticks([0, 1, 2])
    ax_a.set_xlabel(r'$\lambda_1$', labelpad=10)
    ax_a.set_ylabel(r'$\lambda_2$', labelpad=10)
    ax_a.view_init(elev=30, azim=-60)
    ax_a.set_box_aspect((1, 1, 1))
    ax_a.text2D(0.96, 0.97, '(a)', transform=ax_a.transAxes, ha='right', va='top')

    # --- (b) cortes de F1 (izq, sólido) y ΔX1² (der, punteado) ---
    ax_b  = fig.add_subplot(gs[0, 1])
    ax_b2 = ax_b.twinx()
    for i in range(len(CUT_LAM1)):
        ax_b.plot(LAM2_CUT, Fc[i], '-', color=CUT_COLORS[i], lw=2.0)
        ax_b2.plot(LAM2_CUT, Dc[i], ':', color=CUT_COLORS[i], lw=2.0)
    for x in _res_lines(e2):
        ax_b.axvline(x=x, color='k', linestyle='--', lw=1.0, alpha=0.75)
    ax_b.set_xlabel(r'$\lambda_2$')
    ax_b.set_ylabel(r'$F_1$')
    ax_b2.set_ylabel(r'$(\Delta\hat{X}_1)^2$')
    ax_b.set_xlim(LAM2_CUT.min(), LAM2_CUT.max())
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_box_aspect(1)
    ax_b2.set_box_aspect(1)
    ax_b.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, size=5, width=1.5)
    ax_b2.tick_params(axis='y', direction='in', right=True, size=5, width=1.5)
    ax_b.text(0.95, 0.95, '(b)', transform=ax_b.transAxes, ha='right', va='top')
    _offset_axis(ax_b, Fc, 'left')
    _offset_axis(ax_b2, Dc, 'right')

    handles = [mlines.Line2D([], [], color=CUT_COLORS[i], lw=2.0, label=rf'$\lambda_1 = {v}$')
               for i, v in enumerate(CUT_LAM1)]
    handles += [mlines.Line2D([], [], color='k', ls='-', lw=2.0, label=r'$F_1$'),
                mlines.Line2D([], [], color='k', ls=':', lw=2.0, label=r'$(\Delta\hat{X}_1)^2$')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 0.96),
               ncol=len(handles), frameon=False,
               borderpad=0.2, handlelength=1.5, labelspacing=0.4, columnspacing=0.8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def fig_diff(e1, e2):
    dF = np.abs(np.loadtxt(f_map('F1', 'PAPS', e1, e2)) - np.loadtxt(f_map('F1', 'exact', e1, e2)))
    dD = np.abs(np.loadtxt(f_map('DX1', 'PAPS', e1, e2)) - np.loadtxt(f_map('DX1', 'exact', e1, e2)))
    print(f'  max |ΔF1| = {dF.max():.2e},  max |Δ(ΔX1)²| = {dD.max():.2e}')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, lab, tag in zip(axs, (dF, dD),
                               (r'$|F_1^{\rm PAPS} - F_1^{\rm exact}|$',
                                r'$|(\Delta\hat X_1)^2_{\rm PAPS} - (\Delta\hat X_1)^2_{\rm exact}|$'),
                               ('(a)', '(b)')):
        dd = np.maximum(d, DIFF_FLOOR)
        im = ax.imshow(dd, origin='lower', aspect='auto', cmap='viridis',
                       norm=LogNorm(vmin=max(dd.min(), DIFF_FLOOR), vmax=max(dd.max(), 10*DIFF_FLOOR)),
                       extent=[LAM[0], LAM[-1], LAM[0], LAM[-1]])
        for x in _res_lines(e1):
            ax.axvline(x=x, color='w', linestyle='--', lw=1.0, alpha=0.75)
        for y in _res_lines(e2):
            ax.axhline(y=y, color='w', linestyle='--', lw=1.0, alpha=0.75)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([1, 2])                                     # un solo 0 en la esquina
        ax.set_xlim(LAM[0], LAM[-1])
        ax.set_ylim(LAM[0], LAM[-1])
        ax.set_xlabel(r'$\lambda_1$')
        ax.set_ylabel(r'$\lambda_2$')
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True,
                       color='white', size=9, width=2.2)
        ax.text(0.05, 0.95, tag, transform=ax.transAxes, ha='left', va='top', color='white')
        cb = fig.colorbar(im, ax=ax, pad=0.03)
        cb.set_label(lab, fontsize=18)
        cb.ax.tick_params(labelsize=16)
    fig.tight_layout()
    return fig


def _save(fig, sub, name):
    d = os.path.join(OUT_DIR, sub)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f'{name}.png'), bbox_inches='tight', transparent=False, dpi=300)
    fig.savefig(os.path.join(d, f'{name}.svg'), bbox_inches='tight', transparent=False)
    print(f'  Figura guardada en: {os.path.join(d, name)}.png / .svg')


def make_all(e1, e2, show=False):
    need = [f_map(k, m, e1, e2) for k in ('F1', 'DX1') for m in ('PAPS', 'exact')] \
         + [f_cut(k, m, e1, e2) for k in ('F1', 'DX1') for m in ('PAPS', 'exact')]
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        print('  Faltan datos (corre primero fano_lam1lam2.py), se omite:\n    '
              + '\n    '.join(os.path.basename(f) for f in miss))
        return
    if os.path.exists(f_val(e1, e2)):
        R = np.atleast_2d(np.loadtxt(f_val(e1, e2)))
        print(f'  Chequeo con el laboratorio: max |F1_exact − F1_lab| = {np.max(np.abs(R[:, 3] - R[:, 4])):.2e}, '
              f'max |F1_PAPS − F1_lab| = {np.max(np.abs(R[:, 2] - R[:, 4])):.2e}')
    tag = _tag(e1, e2)
    for method in ('PAPS', 'exact'):
        fig = fig_fano(e1, e2, method)
        _save(fig, method, f'Fano_F1_DX1_3D_y_cortes_{method}_{tag}')
        if show:
            plt.show()
        plt.close(fig)
    fig = fig_diff(e1, e2)
    _save(fig, 'diff', f'diff_PAPS_exact_{tag}')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sets, show = EPS_SETS, False
    elif len(sys.argv) == 3:
        sets, show = [(float(sys.argv[1]), float(sys.argv[2]))], True
    else:
        sys.exit('uso: python plot_fano_lam1lam2.py [e1 e2]')
    for e1, e2 in sets:
        print(f'ε1 = {e1:g}, ε2 = {e2:g}')
        make_all(e1, e2, show)
