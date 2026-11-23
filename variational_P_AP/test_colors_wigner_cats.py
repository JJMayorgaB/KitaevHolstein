# -*- coding: utf-8 -*-
"""
PRUEBA DE COLORES de la nueva Fig. 4 (plot_wigner_cats.py): varias combinaciones de mapa de
W(α) y colores de |Cat_+> / |Cat_-> que no usan el rojo/azul de la paridad fermiónica.
Cada variante se guarda en Figures/paper_fig4/color_tests/ y además se arma una hoja de contacto
(color_tests_overview.png) con todas.

Uso:  python test_colors_wigner_cats.py
"""
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_wigner_cats as P

OUT = os.path.join(HERE, 'Figures', 'paper_fig4', 'color_tests')

# mapas propios
NATURE = LinearSegmentedColormap.from_list(       # estilo del ejemplo (b): negativo verde-azul, positivo amarillo-rojo-negro
    'nature_cat', [(0.0, '#0b1a4a'), (0.20, '#1f5fd6'), (0.38, '#33b36b'), (0.5, '#ffffff'),
                   (0.62, '#ffe066'), (0.78, '#f28c28'), (0.90, '#c0281f'), (1.0, '#1a0000')])
YOG = LinearSegmentedColormap.from_list(          # negativo verde, positivo amarillo-naranja
    'green_yellow_orange', [(0.0, '#00441b'), (0.25, '#41ab5d'), (0.5, '#ffffff'),
                            (0.72, '#fec44f'), (1.0, '#cc4c02')])

# (nombre, mapa de W, color |Cat_+>, color |Cat_->)
VARIANTS = [
    ('A_RdBu_teal-ochre',   'RdBu',   '#1B9E77', '#D95F02'),   # mapa actual, gatos Dark2
    ('B_PiYG_ochre-purple', 'PiYG',   '#D95F02', '#5E3C99'),   # rosa-verde (como el ejemplo de F)
    ('C_PRGn_orange-brown', 'PRGn',   '#E08214', '#8C510A'),   # morado-verde
    ('D_BrBG_purple-pink',  'BrBG',   '#6A3D9A', '#C51B7D'),   # marrón-verde azulado
    ('E_GrYO_purple-black', YOG,      '#6A3D9A', '#252525'),   # verde / amarillo-naranja
    ('F_nature_purple-teal', NATURE,  '#7B3294', '#1B9E77'),   # estilo del ejemplo (b)
    ('G_PuOr_teal-pink',    'PuOr_r', '#1B9E77', '#C51B7D'),   # morado-naranja
]

if __name__ == '__main__':
    files = []
    for name, cmap, cp, cm in VARIANTS:
        print(name)
        P.main(cmap=cmap, c_plus=cp, c_minus=cm, fig_name=name, out_dir=OUT, show=False)
        files.append(os.path.join(OUT, f'{name}.png'))
    # hoja de contacto
    n = len(files); ncol = 2; nrow = int(np.ceil(n/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 5.2*nrow))
    for ax, f, (name, *_ ) in zip(axs.ravel(), files, VARIANTS):
        ax.imshow(plt.imread(f)); ax.set_title(name.replace('_', ' '), fontsize=18); ax.axis('off')
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'color_tests_overview.png'), dpi=110)
    print(f'Hoja de contacto: {os.path.join(OUT, "color_tests_overview.png")}')
