# -*- coding: utf-8 -*-
"""
DATOS de la figura 2 del paper (mapas de paridad fermiónica δE = E_e − E_o) con el ansatz
(P+AP)S de variational_P_AP.py (funcional cerrado, minimize_sector, convención de las notas).
Pensado para correr en el clúster (keldysh); la figura la hace plot_fig2_paritydiagrams.py.

    (a)     plano (g/t, Δ/t), acoplamientos isótropos g1 = g2 = g, ε1 = ε2 = 0.25 t
    (b)–(d) plano (g1/t, g2/t) para Δ/t = 0.5, 1.0, 1.5, con ε1 = t y ε2 = 1.25² t
t = ω0 = 1, así que g = λ ω0 = λ. Mallas de 200 × 200.
En (b)–(d) solo se calcula el primer cuadrante g1, g2 ≥ 0 (el espectro es par en cada g_i,
â_i → −â_i); plot_fig2_paritydiagrams.py reconstruye el plano completo por reflexión.

El sector impar no depende de Δ: se calcula una sola vez por panel y se reutiliza.
Continuación (warm start) a lo largo de cada fila: en (a) a lo largo de g, en (b)–(d) a lo
largo de g2 con g1 fijo. Cada fila es una tarea de joblib.

Uso:
    python fig2_paritydiagrams.py          # todo: (a) y (b)–(d)
    python fig2_paritydiagrams.py a        # solo (a)
    python fig2_paritydiagrams.py bcd      # solo (b)–(d)
Los .txt se guardan en Figures/paper_fig2/data/ y no se recalculan si ya existen
(bórralos para recalcular).
"""

import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from variational_P_AP import minimize_sector, sector_params, omega0   # (P+AP)S, convención de las notas

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
N_JOBS = -1                                  # núcleos para joblib (-1 = todos)
T_VAL  = 1.0                                 # t (= ω0)
n_grid = 50

# (a) plano (g/t, Δ/t), isótropo
EPS_A     = 0.5                             # ε1 = ε2 (en unidades de t)
G_A       = np.linspace(0.0, 1.5, n_grid)       # g/t  (= λ, ω0 = t)
DELTA_A   = np.linspace(0.0, 1.5, n_grid)       # Δ/t

# (b)–(d) plano (g1/t, g2/t)
EPS1_2D   = 0.5                               # ε1 = t
EPS2_2D   = 0.5                             # ε2 = 1.25² t  (ε̃2 = 0 en g2 = 1.25 t)
G_2D      = np.linspace(0.0, 1.5, n_grid)       # g1/t y g2/t, primer cuadrante (incluye g = 0)
DELTA_2D  = [0.5, 1.0, 1.5]                  # Δ/t de (b), (c), (d)

DATA_DIR  = os.path.join(HERE, 'Figures', 'paper_fig2', 'data')


def _lam(g):
    return g/omega0                          # λ = g/ω0


# ══════════════════════════════════════════════════════════════════
#  FILAS (una tarea de joblib cada una, con continuación)
# ══════════════════════════════════════════════════════════════════
def _row_iso(sector, Delta):
    """(a): E del sector a lo largo de g, con g1 = g2 = g."""
    E = np.empty(len(G_A))
    w = None
    for k, g in enumerate(G_A):
        lam = _lam(g)
        Lam = np.sqrt(2.0)*abs(lam)
        eps, epsp, J = sector_params(sector, lam, lam, EPS_A, EPS_A, T_VAL, Delta)
        E[k], w = minimize_sector(sector, eps, epsp, J, Lam, w)
    return E


def _row_2d(sector, g1, Delta):
    """(b)–(d): E del sector a lo largo de g2, con g1 fijo."""
    E = np.empty(len(G_2D))
    w = None
    lam1 = _lam(g1)
    for k, g2 in enumerate(G_2D):
        lam2 = _lam(g2)
        Lam = np.sqrt(lam1**2 + lam2**2)
        eps, epsp, J = sector_params(sector, lam1, lam2, EPS1_2D, EPS2_2D, T_VAL, Delta)
        E[k], w = minimize_sector(sector, eps, epsp, J, Lam, w)
    return E


# ══════════════════════════════════════════════════════════════════
#  ARCHIVOS
# ══════════════════════════════════════════════════════════════════
def f_coord(name):
    return os.path.join(DATA_DIR, f'coord_{name}.txt')


def f_dE_a():
    return os.path.join(DATA_DIR, f'dE_a_iso_eps{EPS_A}_t{T_VAL}'
                                  f'_nG{len(G_A)}_nD{len(DELTA_A)}_PAPS.txt')


def f_Eodd_2d():
    return os.path.join(DATA_DIR, f'Eodd_2d_Q1_eps1{EPS1_2D}_eps2{EPS2_2D}_t{T_VAL}'
                                  f'_n{len(G_2D)}_PAPS.txt')


def f_dE_2d(Delta):
    return os.path.join(DATA_DIR, f'dE_2d_Q1_Delta{Delta}_eps1{EPS1_2D}_eps2{EPS2_2D}_t{T_VAL}'
                                  f'_n{len(G_2D)}_PAPS.txt')


def _save_coords():
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savetxt(f_coord('g_a'),     G_A,      fmt='%.10e', header='(a) g/t (columnas de dE_a)')
    np.savetxt(f_coord('Delta_a'), DELTA_A,  fmt='%.10e', header='(a) Delta/t (filas de dE_a)')
    np.savetxt(f_coord('g_2d_Q1'), G_2D,     fmt='%.10e',
               header='(b)-(d) g1/t (columnas) y g2/t (filas) de dE_2d_Q1, primer cuadrante')


# ══════════════════════════════════════════════════════════════════
#  CÁLCULO
# ══════════════════════════════════════════════════════════════════
def compute_a():
    f = f_dE_a()
    if os.path.exists(f):
        print(f'(a): ya existe {os.path.basename(f)}, se omite')
        return
    print(f'(a): {len(DELTA_A) + 1} filas × {len(G_A)} g ...', flush=True)
    jobs = ([delayed(_row_iso)('odd', 0.0)]                          # el impar no depende de Δ
            + [delayed(_row_iso)('even', D) for D in DELTA_A])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    E_odd  = res[0]
    E_even = np.array(res[1:])
    dE = E_even - E_odd[None, :]
    np.savetxt(f, dE, fmt='%.10e',
               header=(f'deltaE = E_even - E_odd, (P+AP)S | filas = Delta/t ({len(DELTA_A)}, '
                       f'{DELTA_A[0]}..{DELTA_A[-1]}), columnas = g/t ({len(G_A)}, '
                       f'{G_A[0]}..{G_A[-1]}) | eps1 = eps2 = {EPS_A}, t = {T_VAL}, '
                       f'omega0 = {omega0}, g1 = g2 = g'))
    print(f'(a): guardado {os.path.basename(f)}  δE ∈ [{dE.min():.4f}, {dE.max():.4f}]')


def compute_bcd():
    # sector impar: una sola vez (no depende de Δ)
    f_o = f_Eodd_2d()
    if os.path.exists(f_o):
        print(f'(b)-(d): impar leído de {os.path.basename(f_o)}')
        E_odd = np.loadtxt(f_o)
    else:
        print(f'(b)-(d): impar, {len(G_2D)} filas × {len(G_2D)} g2 ...', flush=True)
        res = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_row_2d)('odd', g1, 0.0) for g1 in G_2D)
        E_odd = np.array(res).T                                      # filas = g2, columnas = g1
        np.savetxt(f_o, E_odd, fmt='%.10e',
                   header=(f'E_odd, (P+AP)S | filas = g2/t, columnas = g1/t '
                           f'({len(G_2D)}, {G_2D[0]}..{G_2D[-1]}) | eps1 = {EPS1_2D}, '
                           f'eps2 = {EPS2_2D}, t = {T_VAL}, omega0 = {omega0}'))

    # sector par: uno por Δ
    for D in DELTA_2D:
        f = f_dE_2d(D)
        if os.path.exists(f):
            print(f'(b)-(d) Δ/t = {D}: ya existe {os.path.basename(f)}, se omite')
            continue
        print(f'(b)-(d) Δ/t = {D}: par, {len(G_2D)} filas × {len(G_2D)} g2 ...', flush=True)
        res = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_row_2d)('even', g1, D) for g1 in G_2D)
        E_even = np.array(res).T                                     # filas = g2, columnas = g1
        dE = E_even - E_odd
        np.savetxt(f, dE, fmt='%.10e',
                   header=(f'deltaE = E_even - E_odd, (P+AP)S | filas = g2/t, columnas = g1/t '
                           f'({len(G_2D)}, {G_2D[0]}..{G_2D[-1]}) | Delta/t = {D}, '
                           f'eps1 = {EPS1_2D}, eps2 = {EPS2_2D}, t = {T_VAL}, omega0 = {omega0}'))
        print(f'(b)-(d) Δ/t = {D}: guardado  δE ∈ [{dE.min():.4f}, {dE.max():.4f}]')


if __name__ == '__main__':
    which = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    if which not in ('all', 'a', 'bcd'):
        sys.exit('uso: python fig2_paritydiagrams.py [a | bcd]')
    _save_coords()
    if which in ('all', 'a'):
        compute_a()
    if which in ('all', 'bcd'):
        compute_bcd()
    print(f'\nDatos en: {DATA_DIR}')
