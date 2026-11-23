# -*- coding: utf-8 -*-
"""
FIGURA 4 (nueva) del paper, a partir de los datos de wigner_cats.py (Figures/paper_fig4/data/) y
de δ_W de variational_P_AP_cat.py (Figures/cat/data/cat_PAPS/):

    (a) |Cat_->, Λ/√2 = 2     (b) |Cat_+>, Λ/√2 = 3
        mapa W(x, p) variacional (x = √2 Re α, p = √2 Im α, W(x, p) = W(α)/2, |W| ≤ 1/π);
        debajo el corte W(x, 0) y a la derecha W(0, p):
        línea = variacional, círculos = exacto
    (c) δ_W vs Λ/√2: línea = variacional, círculos = exacto
Colores: |Cat_+> naranja y |Cat_-> verde en los tres paneles; mapa de W verde/amarillo-naranja. figsize = (10, 4).

Uso:  python plot_wigner_cats.py     (genera las tres combinaciones de colores de SCHEMES)
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wigner_cats import PANELS, X_MAP, P_MAP, f_map, f_cuts, f_neg

warnings.filterwarnings('ignore')

# Estilo: tamaños pensados para figsize = (10, 4) (dos columnas de PRB, ancho completo)
FS_LAB, FS_TICK, FS_PANEL, FS_LEG = 22, 19, 19, 15
FS_INSET = 15                       # textos dentro de los mapas (a), (b)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': FS_LAB,
    'axes.labelsize': FS_LAB,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_LEG,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
})

OUT_DIR = os.path.join(HERE, 'Figures', 'paper_fig4')
NAME    = 'wigner_cats_PAPS'
# Sin rojo ni azul (reservados a la paridad fermiónica en el paper).
# Mapa de W: negativo verde, positivo amarillo-naranja. Los gatos toman el color del signo de
# W(0): |Cat_+> (W(0) > 0) naranja, |Cat_-> (W(0) < 0) verde.
CMAP_W  = LinearSegmentedColormap.from_list(
    'green_yellow_orange', [(0.0, '#00441b'), (0.25, '#41ab5d'), (0.5, '#ffffff'),
                            (0.72, '#fec44f'), (1.0, '#cc4c02')])
C_PLUS, C_MINUS = '#d95f02', '#1b7837'                           # |Cat_+> naranja, |Cat_-> verde
# Cuadraturas del paper, X = (a + a†)/√2:  x = √2 Re α, p = √2 Im α,  W(x, p) = W(α)/2 (∫W dx dp = 1).
SQ2     = np.sqrt(2.0)
WMAX    = 1.0/np.pi                                              # |W(x, p)| ≤ 1/π
X_LIM   = 6.3                                                    # |x| ≤ X_LIM en los mapas y cortes
X_TICKS = [-6, -3, 0, 3, 6]                                      # ticks de x (±3: lóbulos de |Cat_+>)
NEG_EVERY = 5                                                    # submuestreo de los círculos de (c)


def _tick(ax, **kw):
    ax.tick_params(direction='in', top=True, right=True, size=3.5, width=0.8, **kw)


def _load_cut(f):
    d = np.loadtxt(f)
    return d[:, 0], d[:, 1]


def main(cmap=None, c_plus=None, c_minus=None, fig_name=NAME, out_dir=OUT_DIR, show=True,
         x_lim=None, x_ticks=None):
    x_lim   = X_LIM if x_lim is None else x_lim
    x_ticks = X_TICKS if x_ticks is None else x_ticks
    cmap    = CMAP_W if cmap is None else cmap
    C_PLUS  = globals()['C_PLUS'] if c_plus is None else c_plus
    C_MINUS = globals()['C_MINUS'] if c_minus is None else c_minus
    need = [f_map(L, s) for L, s in PANELS] + [f for L, s in PANELS for k in ('var', 'exact')
                                                for f in f_cuts(k, L, s)] \
         + [f_neg('exact'), f_neg('variational')]
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        sys.exit('Faltan datos (corre primero wigner_cats.py):\n  '
                 + '\n  '.join(os.path.relpath(f, HERE) for f in miss))

    fig = plt.figure(figsize=(10, 4))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.34)
    # bloque (a)-(b): mismo ancho que outer[0], pero baja hasta la altura de la etiqueta Λ/√2 de (c)
    # (la colorbar ya no lleva etiqueta debajo), para dar más alto a (a) y (b)
    pos = outer[0].get_position(fig)
    gl = gridspec.GridSpec(
        7, 2, figure=fig, left=pos.x0, right=pos.x1, bottom=pos.y0 - 0.10, top=pos.y1,
        width_ratios=[4.0, 1.0], wspace=0.06,
        height_ratios=[1.0, 0.42, 0.36, 1.0, 0.42, 0.75, 0.10], hspace=0.08)
    norm = TwoSlopeNorm(vmin=-WMAX, vcenter=0.0, vmax=WMAX)
    ext  = [SQ2*X_MAP[0], SQ2*X_MAP[-1], SQ2*P_MAP[0], SQ2*P_MAP[-1]]

    for k, ((L, sign), lab) in enumerate(zip(PANELS, ('(a)', '(b)'))):
        r0  = 3*k
        col = C_PLUS if sign > 0 else C_MINUS
        name = r'|\mathrm{Cat}_+\rangle' if sign > 0 else r'|\mathrm{Cat}_-\rangle'
        W = 0.5*np.loadtxt(f_map(L, sign))                        # W(α) -> W(x, p)
        xv, cRe_v = _load_cut(f_cuts('var', L, sign)[0])
        pv, cIm_v = _load_cut(f_cuts('var', L, sign)[1])
        xe, cRe_e = _load_cut(f_cuts('exact', L, sign)[0])
        pe, cIm_e = _load_cut(f_cuts('exact', L, sign)[1])
        xv, pv, xe, pe = SQ2*xv, SQ2*pv, SQ2*xe, SQ2*pe            # α -> (x, p)
        cRe_v, cIm_v, cRe_e, cIm_e = 0.5*cRe_v, 0.5*cIm_v, 0.5*cRe_e, 0.5*cIm_e

        # mapa
        ax = fig.add_subplot(gl[r0, 0])
        im = ax.imshow(W, origin='lower', aspect='auto', extent=ext, cmap=cmap, norm=norm,
                       interpolation='bilinear')
        ax.axhline(0.0, color='k', ls='--', lw=0.7)
        ax.axvline(0.0, color='k', ls='--', lw=0.7)
        ax.set_xlim(-x_lim, x_lim); ax.set_ylim(ext[2], ext[3])
        ax.set_yticks([-2, 0, 2])
        ax.set_ylabel(r'$p$', labelpad=2)
        ax.set_xticks(x_ticks)
        _tick(ax, labelbottom=False)
        ax.text(0.025, 0.93, rf'{lab} ${name}$', transform=ax.transAxes, ha='left', va='top',
                fontsize=FS_INSET)
        ax.text(0.025, 0.07, rf'$\Lambda/\sqrt{{2}} = {L:g}$', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=FS_INSET - 1)

        # corte inferior: W(x, 0)
        axb = fig.add_subplot(gl[r0 + 1, 0], sharex=ax)
        axb.plot(xv, cRe_v, '-', color=col, lw=1.3)
        axb.plot(xe, cRe_e, 'o', mfc='none', mec=col, ms=3.5, mew=0.8)
        axb.axhline(0.0, color='0.6', lw=0.5)
        axb.set_ylim(-0.4, 0.4)
        axb.set_yticks([-0.3, 0.3])
        axb.set_yticklabels([r'$-0.3$', r'$0.3$'])
        axb.set_ylabel(r'$W$', labelpad=2)
        _tick(axb, labelbottom=(k == 1))                          # x solo debajo del corte de (b)
        if k == 1:
            axb.set_xlabel(r'$x$', labelpad=2)
        axb.tick_params(axis='y', labelsize=FS_TICK - 3)

        # corte lateral: W(0, p), eje W invertido
        axr = fig.add_subplot(gl[r0, 1], sharey=ax)
        axr.plot(cIm_v, pv, '-', color=col, lw=1.3)
        axr.plot(cIm_e, pe, 'o', mfc='none', mec=col, ms=3.5, mew=0.8)
        axr.axvline(0.0, color='0.6', lw=0.5)
        axr.set_xlim(0.4, -0.4)
        axr.set_xticks([0.3, -0.3])
        axr.set_xticklabels([r'$0.3$', r'$-0.3$'])
        axr.set_xlabel(r'$W$', labelpad=1)
        _tick(axr, labelleft=False)
        axr.tick_params(axis='x', labelsize=FS_TICK - 3)

    # colorbar común, horizontal, bajo (b)
    cax = fig.add_subplot(gl[6, 0])
    cb  = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-0.3, 0.0, 0.3])
    cb.ax.set_xticklabels([r'$-0.3$', r'$0$', r'$0.3$'])
    cb.ax.tick_params(direction='in', size=3, width=0.8, labelsize=FS_TICK)
    cax.text(1.04, 0.5, r'$W(x, p)$', transform=cax.transAxes, ha='left', va='center',
             fontsize=FS_LAB)                                     # etiqueta a la derecha de la barra

    # (c) δ_W
    axc = fig.add_subplot(outer[1])
    NE, NV = np.loadtxt(f_neg('exact')), np.loadtxt(f_neg('variational'))
    sq2 = np.sqrt(2.0)
    for j, col in ((1, C_PLUS), (2, C_MINUS)):                   # columnas: Lam, δ+, δ−
        axc.plot(NV[:, 0]/sq2, NV[:, j], '-', color=col, lw=1.5)
        axc.plot(NE[::NEG_EVERY, 0]/sq2, NE[::NEG_EVERY, j], 'o', mfc='none', mec=col, ms=4, mew=0.9)
    axc.set_xlim(0.0, 3.5)
    axc.set_ylim(-0.01, 0.33)
    axc.set_xticks([0, 1, 2, 3])
    axc.set_yticks([0.0, 0.1, 0.2, 0.3])
    axc.set_xlabel(r'$\Lambda/\sqrt{2}$')
    axc.set_ylabel(r'$\delta_W$')
    _tick(axc)
    axc.text(0.05, 0.95, '(c)', transform=axc.transAxes, ha='left', va='top', fontsize=FS_PANEL)
    axc.text(0.05, 0.731, r'$|\mathrm{Cat}_-\rangle$', transform=axc.transAxes, color=C_MINUS,
             fontsize=FS_LAB)
    axc.text(0.05, 0.104, r'$|\mathrm{Cat}_+\rangle$', transform=axc.transAxes, color=C_PLUS,
             fontsize=FS_LAB)
    leg = [mlines.Line2D([], [], color='0.4', ls='none', marker='o', mfc='none', ms=5, label='Exact'),
           mlines.Line2D([], [], color='0.4', lw=1.5, label='Variational')]
    # leyenda dentro de Λ/√2 ∈ (0, 2), δ_W ∈ (0.1, 0.2)
    axc.legend(handles=leg, loc='lower left', bbox_to_anchor=(0.0, 0.095), bbox_transform=axc.transData,
               frameon=False, handlelength=0.7, handletextpad=0.35, borderpad=0.0, labelspacing=0.3,
               fontsize=FS_LEG)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{fig_name}.png'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(out_dir, f'{fig_name}.svg'), bbox_inches='tight')
    print(f'Figura guardada en: {os.path.join(out_dir, fig_name)}.png / .svg')
    if show:
        plt.show()
    plt.close(fig)


# Combinaciones de colores: (sufijo, mapa de W, color |Cat_+>, color |Cat_->)
SCHEMES = [
    ('green-orange', CMAP_W, C_PLUS, C_MINUS),                                    # verde / amarillo-naranja
    ('red-blue',     'RdBu', plt.cm.seismic(0.88), plt.cm.seismic(0.12)),         # la original
    ('green-pink',   'PiYG', '#4d9221', '#c51b7d'),                               # rosa (W < 0) / verde (W > 0)
]

if __name__ == '__main__':
    for suf, cmap, cp, cm in SCHEMES:
        main(cmap=cmap, c_plus=cp, c_minus=cm, fig_name=f'{NAME}_{suf}', show=False)
