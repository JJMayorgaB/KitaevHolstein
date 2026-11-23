# -*- coding: utf-8 -*-
"""
FIGURA 3 (nueva) del paper, a partir de datos ya calculados (no recalcula nada):

    (a) cortes a λ1 = 0.5, 1.0, 1.25 de F1 (sólido, eje izquierdo) y (ΔX1)² (discontinuo, eje derecho)
        vs λ2: líneas = (P+AP)S, círculos abiertos = F1 exacto, cuadrados abiertos = (ΔX1)² exacto.
        Datos de fano_lam1lam2.py (Figures/fig_fano/data/), ε1 = ε2 = 0.25, Δ = t = ω0.
        Mismas escalas que la figura original: F1 ×10^-1 + 1, (ΔX1)² ×10^-1 + 0.5.
    (b) <X1 X2> (marco recentrado) del estado base en el plano (θ, Δ/t), con la frontera δE = 0.
        Datos de X1X2_theta_delta.py (Figures/fig_X1X2/data/), ε1 = ε2 = 0.25, Λ/√2 = 1.

Los cortes de (a) no usan rojo ni azul (en (b) son la paridad fermiónica).

Uso:  python plot_fig3_fano_X1X2.py          figura con la paleta CUT_SCHEME
      python plot_fig3_fano_X1X2.py test     una figura por cada paleta de CUT_SCHEMES (prueba de colores)
      python plot_fig3_fano_X1X2.py test inferno   solo las paletas cuyo nombre contiene 'inferno'
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fano_lam1lam2 as FA
import X1X2_theta_delta as XX

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 33,
    'axes.labelsize': 33,
    'axes.titlesize': 33,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
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

OUT_DIR = os.path.join(HERE, 'Figures', 'paper_fig3')
NAME    = 'fig3_fano_X1X2_PAPS'

# parámetros de cada panel
E1_A, E2_A = 0.25, 0.25                       # (a)
E1_B, E2_B, L_B = 0.25, 0.25, 1.0             # (b)

# ── escalas del panel (a), las de la figura original: valor mostrado = (y - offset)/scale ──
F1_OFF,  F1_SCL  = 1.0, 1e-1
DX1_OFF, DX1_SCL = 0.5, 1e-1
SCI_FS   = 22                                 # tamaño de los multiplicadores
EX_EVERY = 25                                 # submuestreo de los marcadores exactos (400 -> 16)
MS       = 6                                  # tamaño de los marcadores

# paletas de los cortes λ1 = 0.5, 1.0, 1.25 (sin rojo ni azul)
CUT_SCHEMES = {
    'viridis':     [plt.cm.viridis(0.10), plt.cm.viridis(0.45), plt.cm.viridis(0.78)],
    'green-ochre': ['#1b7837', '#b8860b', '#4d4d4d'],
    'grays':       ['0.0', '0.40', '0.65'],
    'cividis':     [plt.cm.cividis(0.05), plt.cm.cividis(0.50), plt.cm.cividis(0.90)],
    'greens':      [plt.cm.Greens(0.95), plt.cm.Greens(0.70), plt.cm.Greens(0.45)],
    'teal-brown':  ['#01665e', '#35978f', '#8c510a'],
    # inferno (el colormap del mapa 3D de F1); se evita el extremo amarillo, poco visible sobre blanco
    'inferno-A':   [plt.cm.inferno(0.10), plt.cm.inferno(0.45), plt.cm.inferno(0.75)],
    'inferno-B':   [plt.cm.inferno(0.20), plt.cm.inferno(0.55), plt.cm.inferno(0.80)],
    'inferno-C':   [plt.cm.inferno(0.05), plt.cm.inferno(0.35), plt.cm.inferno(0.65)],
    'inferno-D':   [plt.cm.inferno(0.30), plt.cm.inferno(0.60), plt.cm.inferno(0.85)],
}
CUT_SCHEME = 'inferno-D'
# posición (λ2, F1, alineación) de las etiquetas λ1 = 0.5, 1.0, 1.25 del panel (a)
STYLE_LEG_POS = (0.85, 1.114)                  # (λ2, F1) del borde izquierdo de la leyenda F1 / (ΔX1)²
CUT_LABEL_POS = [(1.36, 1.0616, 'left'), (0.74, 1.1454, 'left'), (1.04, 1.0858, 'left')]
BOUNDARY_COLOR = 'white'                       # color de la frontera δE = 0 en (b)
CMAP_B     = 'seismic_r'                      # (b): <X1 X2> < 0 (impar) rojo, > 0 (par) azul


def _offset_fmt(off, scl):
    return ticker.FuncFormatter(lambda y, _: f'{(y - off)/scl:.1f}')


def _fmt(v):
    """Una cifra decimal como máximo: 0.5 -> '0.5', 1.0 -> '1'."""
    s = f'{abs(v):.1f}'.rstrip('0').rstrip('.')
    return r'$0$' if s == '0' else (r'$-' if v < 0 else r'$') + s + r'$'


def _ticks(ax, col='black'):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True,
                   color=col, size=9, width=2.2)


def panel_a(ax_b, cut_colors):
    lam2 = FA.LAM2_CUT
    Fv = np.loadtxt(FA.f_cut('F1', 'PAPS', E1_A, E2_A))
    Dv = np.loadtxt(FA.f_cut('DX1', 'PAPS', E1_A, E2_A))
    Fe = np.loadtxt(FA.f_cut('F1', 'exact', E1_A, E2_A))
    De = np.loadtxt(FA.f_cut('DX1', 'exact', E1_A, E2_A))
    ax_b2 = ax_b.twinx()                                     # eje derecho para ΔX1²
    sl = slice(EX_EVERY//2, None, EX_EVERY)
    for i in range(len(FA.CUT_LAM1)):
        c = cut_colors[i]
        ax_b.plot(lam2,  Fv[i], '-', color=c, lw=2.0)
        ax_b2.plot(lam2, Dv[i], '--', color=c, lw=2.0)
        ax_b.plot(lam2[sl],  Fe[i][sl], 'o', mfc='none', mec=c, ms=MS, mew=1.3)
        ax_b2.plot(lam2[sl], De[i][sl], 's', mfc='none', mec=c, ms=MS - 0.5, mew=1.3)
    for x in ([np.sqrt(E2_A)] if E2_A > 0 else []):          # ε̃2 = 0
        ax_b.axvline(x=x, color='k', linestyle='--', lw=1.0, alpha=0.75)

    ax_b.set_xlabel(r'$\lambda_2$')
    ax_b.set_ylabel(r'$F_1$')
    ax_b2.set_ylabel(r'$(\Delta\hat{X}_1)^2$')
    ax_b.set_xlim(lam2.min(), lam2.max())
    ax_b.set_xticks([0, 1, 2])
    ax_b.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, size=5, width=1.5)
    ax_b2.tick_params(axis='y', direction='in', right=True, size=5, width=1.5)
    ax_b.text(0.95, 0.95, '(a)', transform=ax_b.transAxes, ha='right', va='top')

    # eje F1: ticks 0.0, 0.5, 1.0, 1.5  (×10^{-1} + 1)
    ax_b.set_ylim(0.99, 1.16)
    ax_b.set_yticks(F1_OFF + F1_SCL*np.array([0.0, 0.5, 1.0, 1.5]))
    ax_b.yaxis.set_major_formatter(_offset_fmt(F1_OFF, F1_SCL))
    ax_b.text(0.0, 1.01, r'$\times10^{-1}+1$', transform=ax_b.transAxes,
              ha='left', va='bottom', fontsize=SCI_FS)
    # eje ΔX1²: ticks 0.0, 0.5, 1.0  (×10^{-1} + 0.5)
    ax_b2.set_ylim(0.495, 0.61)
    ax_b2.set_yticks(DX1_OFF + DX1_SCL*np.array([0.0, 0.5, 1.0]))
    ax_b2.yaxis.set_major_formatter(_offset_fmt(DX1_OFF, DX1_SCL))
    ax_b2.text(1.0, 1.01, r'$\times10^{-1}+0.5$', transform=ax_b2.transAxes,
               ha='right', va='bottom', fontsize=SCI_FS)

    # leyendas: color = λ1 (arriba a la derecha); estilo: sólida + ○ = F1, discontinua + □ = (ΔX1)²
    # (abajo a la izquierda, zona libre); líneas = (P+AP)S, marcadores = exacto
    h1 = [mlines.Line2D([], [], color=cut_colors[i], lw=2.0, label=rf'$\lambda_1 = {v}$')
          for i, v in enumerate(FA.CUT_LAM1)]
    h2 = [mlines.Line2D([], [], color='k', ls='-', lw=1.6, marker='o', mfc='none', ms=MS - 1,
                        label=r'$F_1$'),
          mlines.Line2D([], [], color='k', ls='--', lw=1.6, marker='s', mfc='none', ms=MS - 1.5,
                        label=r'$(\Delta\hat X_1)^2$')]
    # λ1 = v como texto del color de su curva, junto a ella (como |Cat_±> en la Fig. 4)
    for i, v in enumerate(FA.CUT_LAM1):
        x, y, ha = CUT_LABEL_POS[i]
        ax_b.text(x, y, rf'$\lambda_1 = {v}$', color=cut_colors[i], ha=ha, va='center',
                  fontsize=19, transform=ax_b.transData)
    ax_b.legend(handles=h2, loc='center left', bbox_to_anchor=STYLE_LEG_POS, bbox_transform=ax_b.transData,
                frameon=False, fontsize=17, handlelength=1.5, labelspacing=0.2, borderaxespad=0.0)
    return ax_b2


def panel_b(fig, ax):
    _odd = np.loadtxt(XX.f_odd(E1_B, E2_B, L_B))
    th, E_o, C_o = _odd[:, 0], _odd[:, 1], _odd[:, 2]
    E_e = np.loadtxt(XX.f_even('E', E1_B, E2_B, L_B))
    C_e = np.loadtxt(XX.f_even('X1X2', E1_B, E2_B, L_B))
    dE  = E_e - E_o[None, :]
    X12 = np.where(dE < 0.0, C_e, C_o[None, :])
    vmax = np.ceil(np.round(10.0*np.max(np.abs(X12)), 6))/10.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(X12, origin='lower', aspect='auto', cmap=CMAP_B, norm=norm,
                   extent=[th[0], th[-1], XX.DELTA[0], XX.DELTA[-1]])
    if dE.min() < 0.0 < dE.max():
        ax.contour(th, XX.DELTA, dE, levels=[0.0], colors=BOUNDARY_COLOR, linewidths=1.75, linestyles='--')
    ax.set_xticks([0.0, np.pi/4, np.pi/2])
    ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels([r'$1$', r'$2$'])
    ax.set_xlim(th[0], th[-1])
    ax.set_ylim(XX.DELTA[0], XX.DELTA[-1])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\Delta/t$')
    _ticks(ax, col='white')
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, ha='left', va='top', color='white')
    cb = fig.colorbar(im, ax=ax, pad=0.03, ticks=[-vmax, 0.0, vmax])
    cb.ax.set_yticklabels([_fmt(-vmax), _fmt(0.0), _fmt(vmax)])
    cb.set_label(r'$\langle\hat X_1\hat X_2\rangle$')
    cb.ax.tick_params(direction='in')


def main(scheme=CUT_SCHEME, fig_name=NAME, out_dir=OUT_DIR, show=True):
    need = ([FA.f_cut(k, m, E1_A, E2_A) for k in ('F1', 'DX1') for m in ('PAPS', 'exact')]
            + [XX.f_odd(E1_B, E2_B, L_B), XX.f_even('E', E1_B, E2_B, L_B),
               XX.f_even('X1X2', E1_B, E2_B, L_B)])
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        sys.exit('Faltan datos:\n  ' + '\n  '.join(os.path.relpath(f, HERE) for f in miss))

    fig = plt.figure(figsize=(11.5, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.65)
    panel_a(fig.add_subplot(gs[0, 0]), CUT_SCHEMES[scheme])
    panel_b(fig, fig.add_subplot(gs[0, 1]))

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{fig_name}.png'), bbox_inches='tight', transparent=False, dpi=300)
    fig.savefig(os.path.join(out_dir, f'{fig_name}.svg'), bbox_inches='tight', transparent=False)
    print(f'Figura guardada en: {os.path.join(out_dir, fig_name)}.png / .svg')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'test':
        tdir = os.path.join(OUT_DIR, 'color_tests')
        key  = sys.argv[2].lower() if len(sys.argv) > 2 else ''      # p. ej. 'inferno': solo esas paletas
        sel  = [k for k in CUT_SCHEMES if key in k.lower()]
        for sch in sel:
            main(sch, fig_name=f'{NAME}_{sch}', out_dir=tdir, show=False)
        files = [os.path.join(tdir, f'{NAME}_{sch}.png') for sch in sel]
        n = len(files); nrow = int(np.ceil(n/2))
        figo, axs = plt.subplots(nrow, 2, figsize=(20, 4.6*nrow))
        for axo, f, sch in zip(axs.ravel(), files, sel):
            axo.imshow(plt.imread(f)); axo.set_title(sch, fontsize=18); axo.axis('off')
        for axo in axs.ravel()[n:]:
            axo.axis('off')
        figo.tight_layout()
        ov = f'color_tests_overview{"_" + key if key else ""}.png'
        figo.savefig(os.path.join(tdir, ov), dpi=110)
        print(f'Hoja de contacto: {os.path.join(tdir, ov)}')
    else:
        main()
