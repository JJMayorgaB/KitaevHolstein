# -*- coding: utf-8 -*-
"""
Migración única a la convención con J_α/ω0 en los nombres (los datos ya calculados eran de
J_α/ω0 = 1.0) y a la nueva organización de Figures/cat. No borra nada: solo mueve y renombra.

    python migrar_cv1.py            # muestra lo que haría (no toca nada)
    python migrar_cv1.py --apply    # lo hace

1. Mapas (P+AP)S:  data/cat_state_information_resources_PAPS_sign±1_.../<archivos>  ->  .../cv1.0/<archivos>
2. Cortes en θ:    data/marginal_negativity_vs_theta_{exact,PAPS}_sign±1_{Λ}.txt     ->  ..._sign±1_cv1.0_{Λ}.txt
3. cat_PAPS:       negativity_*.txt, wigner_{exact,var}_lam*.txt, wigner_joint_*_{ReRe,ImIm}.txt  ->  con _cv1.0
4. Figuras sueltas de Figures/cat (png/svg/jpg)  ->  Figures/cat/_old_flat/ (las nuevas se generan
   con los plot_*.py en fidelity/, cat_params/, negativity/, wigner/, info_resources/, checks/)
Los mapas exactos viejos (nL_200) no se tocan: los nuevos (nL_300) van en otra carpeta.
"""
import os
import re
import sys

HERE  = os.path.dirname(os.path.abspath(__file__))
CAT   = os.path.join(HERE, 'Figures', 'cat')
DATA  = os.path.join(CAT, 'data')
APPLY = '--apply' in sys.argv
moves = []

# 1. mapas (P+AP)S: archivos sueltos -> subcarpeta cv1.0
if os.path.isdir(DATA):
    for d in sorted(os.listdir(DATA)):
        full = os.path.join(DATA, d)
        if os.path.isdir(full) and re.match(r'cat_state_information_resources_PAPS_sign[+-]1_', d):
            for f in sorted(os.listdir(full)):
                if os.path.isfile(os.path.join(full, f)):
                    moves.append((os.path.join(full, f), os.path.join(full, 'cv1.0', f)))

# 2. cortes en θ sin etiqueta de J_α/ω0
if os.path.isdir(DATA):
    for f in sorted(os.listdir(DATA)):
        m = re.fullmatch(r'(marginal_negativity_vs_theta_(?:exact|PAPS)_sign[+-]1)_([0-9.]+\.txt)', f)
        if m:
            moves.append((os.path.join(DATA, f), os.path.join(DATA, f'{m.group(1)}_cv1.0_{m.group(2)}')))

# 3. cat_PAPS
CP = os.path.join(DATA, 'cat_PAPS')
if os.path.isdir(CP):
    for f in sorted(os.listdir(CP)):
        if '_cv' in f:
            continue
        m1 = re.fullmatch(r'(negativity_(?:exact|variational)|wigner_(?:exact|var)_lam[0-9.]+?)\.txt', f)
        m2 = re.fullmatch(r'(wigner_joint_.+)_(ReRe|ImIm)\.txt', f)
        if m1:
            moves.append((os.path.join(CP, f), os.path.join(CP, f'{m1.group(1)}_cv1.0.txt')))
        elif m2:
            moves.append((os.path.join(CP, f), os.path.join(CP, f'{m2.group(1)}_cv1.0_{m2.group(2)}.txt')))

# 4. figuras sueltas -> _old_flat
if os.path.isdir(CAT):
    for f in sorted(os.listdir(CAT)):
        if os.path.isfile(os.path.join(CAT, f)) and f.lower().endswith(('.png', '.svg', '.jpg')):
            moves.append((os.path.join(CAT, f), os.path.join(CAT, '_old_flat', f)))

clash = [b for _, b in moves if os.path.exists(b)]
for a, b in moves:
    print(f'{os.path.relpath(a, HERE)}\n    -> {os.path.relpath(b, HERE)}')
print(f'\n{len(moves)} movimientos' + (f', {len(clash)} destinos ya existen (no se tocan)' if clash else ''))

if not APPLY:
    print('Nada se movió. Para aplicar:  python migrar_cv1.py --apply')
    sys.exit(0)
for a, b in moves:
    if os.path.exists(b):
        continue
    os.makedirs(os.path.dirname(b), exist_ok=True)
    os.replace(a, b)
print('Listo.')
