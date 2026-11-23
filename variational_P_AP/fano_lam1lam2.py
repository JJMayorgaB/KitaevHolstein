# -*- coding: utf-8 -*-
"""
DATOS de la figura 3 del paper (respuesta fotónica en el plano (λ1, λ2)): factor de Fano de la
cavidad 1, F1, y varianza de la cuadratura (ΔX1)², en el estado base, con
    [PAPS]  el ansatz (P+AP)S de variational_P_AP.py (minimize_sector, convención de las notas)
    [exact] la diagonalización exacta del bloque H_α de un modo (exact_ground)
Pensado para correr en el clúster (keldysh); las figuras las hace plot_fano_lam1lam2.py.

    Δ = t = ω0 = 1,  g_i = λ_i ω0,  primer cuadrante λ1, λ2 ∈ [0, 2]
    |ε1| = |ε2| = 0.25 ω0: los cuatro signos (EPS_SETS)
    mapa N_GRID × N_GRID y cortes finos a λ1 = 0.5, 0.75, 1.0 (N_CUT puntos en λ2)

Convención de pseudoespín (notas):
    impar: |↑> = |10>, |↓> = |01>;   par: |↑> = |11>, |↓> = |00>
Estado base: el sector de menor energía, δE = E_e − E_o: par si δE < −TOL_DEG, si no impar.
Con Δ = t los dos sectores son EXACTAMENTE degenerados en las rectas ε̃1 = 0 o ε̃2 = 0
(λ_i = √(ε_i/ω0), si ε_i > 0): ahí los bloques impar (ε̃1, ε̃2) y par (ε̃1 + ε̃2, 0) tienen la misma
diagonal salvo el orden. Para que el ruido numérico (|δE| ~ 1e-14) no escoja el sector al azar,
con |δE| ≤ TOL_DEG se toma el impar. Cada método escoge con su propio δE.

OBSERVABLES EN EL LABORATORIO a partir de las componentes ψ_σ = c_σ |φ_σ> del modo b.
Estado de laboratorio:  Σ_σ c_σ |σ> D_b(β_σ)|φ_σ> ⊗ |ζ_B>, con
    impar: b = (λ1 a1 − λ2 a2)/Λ, B = (λ2 a1 + λ1 a2)/Λ;  β↑ = −λ1²/Λ, β↓ = λ2²/Λ, ζ_B = −λ1λ2/Λ
    par:   b = (λ1 a1 + λ2 a2)/Λ, B = (λ2 a1 − λ1 a2)/Λ;  β↑ = −Λ,      β↓ = 0,      ζ_B = 0
En ambos sectores a1 = (λ1 b + λ2 B)/Λ. Como |ζ_B> es coherente, en productos en orden normal
B -> ζ_B, así que sobre el modo b:  a1 -> A1 = (λ1/Λ) b + (λ2/Λ) ζ_B. Con u_σ = A1 D_b(β_σ) ψ_σ:
    <n1>       = Σ_σ ||u_σ||²
    <a1†² a1²> = Σ_σ ||A1 u_σ||²
    F1 = (<a1†² a1²> + <n1> − <n1>²)/<n1>          (F1 = 1 si <n1> = 0: cavidad 1 en el vacío)
    (ΔX1)² = [λ1² Var(X_b) + λ2²/2]/Λ²            (Var X_B = 1/2, Cov(X_b, X_B) = 0)
(los términos cruzados entre σ se anulan porque <↑|↓> = 0). En λ1 = λ2 = 0: F1 = 1, (ΔX1)² = 1/2.

CHEQUEO con el exacto de LABORATORIO (dos cavidades â1, â2 truncadas; n1 y X1 directos) en
VAL_POINTS, para cada (ε1, ε2): compara [PAPS], [exact] y [lab].

Uso:
    python fano_lam1lam2.py                  # los cuatro (ε1, ε2): mapa, cortes y chequeo
    python fano_lam1lam2.py e1 e2            # solo ese (ε1, ε2)
    python fano_lam1lam2.py e1 e2 map|val    # solo mapa+cortes, o solo el chequeo
Los .txt se guardan en Figures/fig_fano/data/ y no se recalculan si ya existen
(bórralos para recalcular).
"""

import os
import sys
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from variational_P_AP import (minimize_sector, sector_params, closed_ABC, weights_2x2,
                              var_states, get_ops, exact_ground, _Mof_Lam, omega0)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════
N_JOBS    = -1                                  # núcleos para joblib (-1 = todos)
T_VAL     = 1.0                                 # t (= ω0)
DELTA_VAL = 1.0                                 # Δ (= t)

EPS_SETS  = [(0.25, 0.25), (0.25, -0.25), (-0.25, 0.25), (-0.25, -0.25)]   # (ε1, ε2)

N_GRID    = 100                                 # malla del mapa (prueba; luego 300)
G_MIN, G_MAX = 0.0, 2.0
LAM       = np.linspace(G_MIN, G_MAX, N_GRID)  # λ1 (columnas) y λ2 (filas)

CUT_LAM1  = [0.5, 0.75, 1.0]                    # cortes de (b)
CUT_SHIFT = 1e-6                                # corte sobre la recta degenerada λ1 = √ε1 -> límite λ1 -> √ε1⁻
N_CUT     = 400
LAM2_CUT  = np.linspace(G_MIN, G_MAX, N_CUT)

VAL_POINTS = [(0.5, 0.3), (1.0, 0.5), (1.25, 1.5), (1.5, 1.0), (2.0, 2.0)]   # (λ1, λ2) del chequeo
N_MIN, N_MAX, N_CONV = 30, 160, 20              # truncación de cada cavidad en el exacto de laboratorio

TOL_DEG    = 1e-9                               # |δE| ≤ TOL_DEG: degenerado -> se toma el impar

DATA_DIR  = os.path.join(HERE, 'Figures', 'fig_fano', 'data')

KINDS   = ('dE', 'F1', 'DX1')                   # cantidades que se guardan
METHODS = ('PAPS', 'exact')


# ══════════════════════════════════════════════════════════════════
#  OBSERVABLES A PARTIR DE LAS COMPONENTES DEL MODO b
# ══════════════════════════════════════════════════════════════════
def _lab_shifts(sector, lam1, lam2, Lam):
    """(β↑, β↓, ζ_B) del estado de laboratorio."""
    if sector == 'odd':
        return -lam1**2/Lam, lam2**2/Lam, -lam1*lam2/Lam
    return -Lam, 0.0, 0.0


def observables(psi_u, psi_d, sector, lam1, lam2, Lam, ops):
    """(F1, (ΔX1)²) en el laboratorio; psi_σ = c_σ|φ_σ> (con su peso), en Fock del modo b."""
    if Lam < 1e-12:
        return 1.0, 0.5
    bu, bd, zB = _lab_shifts(sector, lam1, lam2, Lam)
    b  = ops.b
    Xb = (b + b.T)/np.sqrt(2.0)
    n1 = nn = 0.0
    mX = mX2 = 0.0
    for psi, beta in ((psi_u, bu), (psi_d, bd)):
        v  = ops.apply(1, beta, psi)                     # D_b(β) ψ_σ
        u  = (lam1/Lam)*(b @ v) + (lam2/Lam)*zB*v         # A1 v
        w  = (lam1/Lam)*(b @ u) + (lam2/Lam)*zB*u         # A1² v
        n1 += u @ u
        nn += w @ w
        Xv = Xb @ v
        mX += v @ Xv
        mX2 += Xv @ Xv
    F1 = 1.0 if n1 < 1e-14 else (nn + n1 - n1*n1)/n1
    var_b = mX2 - mX*mX
    DX1 = (lam1**2*var_b + 0.5*lam2**2)/Lam**2
    return F1, DX1


def _point(sector, lam1, lam2, e1, e2, warm):
    """Un sector en (λ1, λ2): (E, F1, DX1) con PAPS y con el exacto del bloque, y x óptimo."""
    Lam = np.hypot(lam1, lam2)
    eps, epsp, J = sector_params(sector, lam1, lam2, e1, e2, T_VAL, DELTA_VAL)
    M   = _Mof_Lam(Lam)
    ops = get_ops(M)
    # (P+AP)S
    E_v, x = minimize_sector(sector, eps, epsp, J, Lam, warm)
    A, B, C = closed_ABC(x, eps, epsp, J, Lam)
    if C == 0.0 and abs(A - B) < 1e-10*max(1.0, abs(A)):
        cu, cd = np.sqrt(0.5), np.sqrt(0.5)             # bloque degenerado: límite J -> 0⁺
    else:
        cu, cd = weights_2x2(A, B, C)
    pu, pd = var_states(x, Lam, ops)
    F_v, D_v = observables(cu*pu, cd*pd, sector, lam1, lam2, Lam, ops)
    # exacto del bloque
    E_x, v = exact_ground(eps, epsp, J, Lam, M)
    F_x, D_x = observables(v[:M + 1], v[M + 1:], sector, lam1, lam2, Lam, ops)
    return (E_v, F_v, D_v), (E_x, F_x, D_x), x


def _off_line(lam, e):
    """Si λ cae exactamente sobre una recta degenerada λ = √(ε/ω0) (con Δ = t, δE = 0 en toda la
    recta), se toma el límite por la izquierda λ − CUT_SHIFT. Con las mallas actuales del mapa
    (linspace(0, 2, N) con N = 100 o 300) ningún punto cae sobre esas rectas (|δE| ≥ 1.5e-5);
    esto solo protege mallas que las contengan."""
    return lam - CUT_SHIFT if (e > 0 and abs(lam - np.sqrt(e/omega0)) < 1e-12) else lam


def _row(lam1, lam2_arr, e1, e2):
    """Fila con λ1 fijo a lo largo de λ2 (continuación en cada sector).
    Devuelve dict[(kind, method)] -> array(len(lam2_arr)) del estado base."""
    lam1     = _off_line(lam1, e1)
    lam2_arr = np.array([_off_line(l2, e2) for l2 in lam2_arr])
    n = len(lam2_arr)
    out = {(k, m): np.empty(n) for k in KINDS for m in METHODS}
    warm = {'odd': None, 'even': None}
    for k, lam2 in enumerate(lam2_arr):
        res = {}
        for s in ('odd', 'even'):
            res[s] = _point(s, lam1, lam2, e1, e2, warm[s])
            warm[s] = res[s][2]
        for im, m in enumerate(METHODS):
            Eo, Fo, Do = res['odd'][im]
            Ee, Fe, De = res['even'][im]
            dE = Ee - Eo
            out[('dE', m)][k]  = dE
            out[('F1', m)][k]  = Fe if dE < -TOL_DEG else Fo
            out[('DX1', m)][k] = De if dE < -TOL_DEG else Do
    return out


# ══════════════════════════════════════════════════════════════════
#  EXACTO DE LABORATORIO (chequeo)
# ══════════════════════════════════════════════════════════════════
def _N_of_lam(lam):
    return int(min(N_MAX, max(N_MIN, np.ceil((abs(lam) + 6.0)**2))))


def lab_ground(sector, lam1, lam2, e1, e2, J, N1, N2):
    """Estado base exacto del bloque de paridad en el laboratorio: (E, F1, (ΔX1)²) directos."""
    confs = [(1, 0), (0, 1)] if sector == 'odd' else [(1, 1), (0, 0)]   # |↑>, |↓>
    def ladder(N):
        return sp.diags(np.sqrt(np.arange(1, N + 1, dtype=float)), 1, format='csr')
    a1, a2 = ladder(N1), ladder(N2)
    I1, I2 = sp.identity(N1 + 1, format='csr'), sp.identity(N2 + 1, format='csr')
    n1op = sp.diags(np.arange(N1 + 1, dtype=float))
    n2op = sp.diags(np.arange(N2 + 1, dtype=float))
    x1, x2 = a1 + a1.T, a2 + a2.T                                       # a + a†
    Hph = omega0*(sp.kron(n1op, I2) + sp.kron(I1, n2op))
    g1, g2 = lam1*omega0, lam2*omega0
    blocks = []
    for (m1, m2) in confs:
        blocks.append((e1*m1 + e2*m2)*sp.identity((N1 + 1)*(N2 + 1)) + Hph
                      + g1*m1*sp.kron(x1, I2) + g2*m2*sp.kron(I1, x2))
    Id = sp.identity((N1 + 1)*(N2 + 1), format='csr')
    H  = sp.bmat([[blocks[0], J*Id], [J*Id, blocks[1]]], format='csr')
    w, V = eigsh(H, k=1, which='SA', tol=1e-12)
    psi = V[:, 0].reshape(2, N1 + 1, N2 + 1)
    nvec = np.arange(N1 + 1, dtype=float)
    X1 = (x1/np.sqrt(2.0)).toarray()
    P1 = np.sum(psi**2, axis=(0, 2))                                    # distribución de n1
    n1, n1sq = P1 @ nvec, P1 @ nvec**2
    F1 = 1.0 if n1 < 1e-14 else (n1sq - n1*n1)/n1
    mX  = sum(np.sum(p*(X1 @ p)) for p in psi)
    mX2 = sum(np.sum((X1 @ p)**2) for p in psi)
    return w[0], F1, mX2 - mX*mX


def _val_point(e1, e2, lam1, lam2):
    out = {}
    for s in ('odd', 'even'):
        (Ev, Fv, Dv), (Ex, Fx, Dx), _ = _point(s, lam1, lam2, e1, e2, None)
        J = T_VAL if s == 'odd' else DELTA_VAL
        N1, N2 = _N_of_lam(lam1), _N_of_lam(lam2)
        El, Fl, Dl = lab_ground(s, lam1, lam2, e1, e2, J, N1, N2)
        _, Fl2, Dl2 = lab_ground(s, lam1, lam2, e1, e2, J, N1 + N_CONV, N2 + N_CONV)
        out[s] = (Ev, Fv, Dv, Ex, Fx, Dx, El, Fl, Dl, Fl2, Dl2)
    # cada método escoge el estado base con su propia energía
    gs = lambda i: 'even' if out['even'][i] - out['odd'][i] < -TOL_DEG else 'odd'
    sv, sx, sl = gs(0), gs(3), gs(6)
    return [lam1, lam2, out[sv][1], out[sx][4], out[sl][7], out[sl][9],
            out[sv][2], out[sx][5], out[sl][8], out[sl][10],
            1.0 if sv == 'even' else 0.0, 1.0 if sl == 'even' else 0.0,
            out['even'][6] - out['odd'][6]]


# ══════════════════════════════════════════════════════════════════
#  ARCHIVOS (todos los parámetros y mallas en el nombre)
# ══════════════════════════════════════════════════════════════════
def _tag(e1, e2):
    return f'eps1{e1:g}_eps2{e2:g}_t{T_VAL:g}_D{DELTA_VAL:g}'


def f_map(kind, method, e1, e2):
    """Mapa del estado base; filas = λ2, columnas = λ1."""
    return os.path.join(DATA_DIR, f'map_{kind}_{method}_{_tag(e1, e2)}'
                                  f'_g{G_MIN:g}-{G_MAX:g}_n{N_GRID}.txt')


def _on_line(v, e1):
    """¿El corte λ1 = v cae sobre la recta ε̃1 = 0 (λ1 = √(ε1/ω0))? Con Δ = t, ahí los dos
    sectores son exactamente degenerados para todo λ2 (H_e ≅ Π H_o Π tras ↑ <-> ↓)."""
    return e1 > 0 and abs(v - np.sqrt(e1/omega0)) < 1e-12


def cut_lam1(v, e1):
    """λ1 con el que se calcula el corte: sobre la recta degenerada se toma el límite
    λ1 -> √ε1⁻ (λ1 = v − CUT_SHIFT), donde el estado base sí está definido y cambia de paridad
    al cruzar λ2 = √ε2, como en la Fig. 2(c) a un lado de la frontera vertical."""
    return v - CUT_SHIFT if _on_line(v, e1) else v


def f_cut(kind, method, e1, e2):
    """Cortes del estado base; filas = λ1 de CUT_LAM1, columnas = λ2 (sufijo 'm': límite por izquierda)."""
    lam = '-'.join(f'{v:g}' + ('m' if _on_line(v, e1) else '') for v in CUT_LAM1)
    return os.path.join(DATA_DIR, f'cut_{kind}_{method}_{_tag(e1, e2)}'
                                  f'_lam1{lam}_g{G_MIN:g}-{G_MAX:g}_n{N_CUT}.txt')


def f_val(e1, e2):
    return os.path.join(DATA_DIR, f'validation_{_tag(e1, e2)}.txt')


# ══════════════════════════════════════════════════════════════════
#  CÁLCULO
# ══════════════════════════════════════════════════════════════════
def compute_maps(sets):
    todo = [(e1, e2) for e1, e2 in sets
            if not all(os.path.exists(f) for k in KINDS for m in METHODS
                       for f in (f_map(k, m, e1, e2), f_cut(k, m, e1, e2)))]
    for e1, e2 in sets:
        if (e1, e2) not in todo:
            print(f'mapa: ya existen los de ε1 = {e1:g}, ε2 = {e2:g}, se omite')
    if not todo:
        return
    jobs, keys = [], []
    for e1, e2 in todo:
        for lam1 in LAM:
            jobs.append(delayed(_row)(lam1, LAM, e1, e2)); keys.append((e1, e2, 'map'))
        for lam1 in CUT_LAM1:
            jobs.append(delayed(_row)(cut_lam1(lam1, e1), LAM2_CUT, e1, e2)); keys.append((e1, e2, 'cut'))
    print(f'mapa: {len(todo)} (ε1, ε2) × ({N_GRID} filas × {N_GRID} λ2 + {len(CUT_LAM1)} cortes × '
          f'{N_CUT} λ2), PAPS + exacto ...', flush=True)
    res = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    os.makedirs(DATA_DIR, exist_ok=True)
    for e1, e2 in todo:
        rows_map = [r for r, k in zip(res, keys) if k == (e1, e2, 'map')]
        rows_cut = [r for r, k in zip(res, keys) if k == (e1, e2, 'cut')]
        hdr = (f'eps1 = {e1}, eps2 = {e2}, t = {T_VAL}, Delta = {DELTA_VAL}, omega0 = {omega0}')
        for k in KINDS:
            for m in METHODS:
                M = np.array([r[(k, m)] for r in rows_map]).T           # filas = λ2, columnas = λ1
                np.savetxt(f_map(k, m, e1, e2), M, fmt='%.12e',
                           header=(f'{k} del estado base ({m}) | filas = lambda2, columnas = lambda1 '
                                   f'({N_GRID}, {G_MIN}..{G_MAX}) | {hdr}'))
                C = np.array([r[(k, m)] for r in rows_cut])               # filas = λ1 del corte
                np.savetxt(f_cut(k, m, e1, e2), C, fmt='%.12e',
                           header=(f'{k} del estado base ({m}) | filas = lambda1 = {CUT_LAM1}, '
                                   f'columnas = lambda2 ({N_CUT}, {G_MIN}..{G_MAX}) | {hdr}'))
        Fv = np.loadtxt(f_map('F1', 'PAPS', e1, e2))
        Fx = np.loadtxt(f_map('F1', 'exact', e1, e2))
        print(f'mapa ε1 = {e1:g}, ε2 = {e2:g}: F1 ∈ [{Fx.min():.4f}, {Fx.max():.4f}] (exacto), '
              f'max |F1_PAPS − F1_exact| = {np.max(np.abs(Fv - Fx)):.2e}')


def compute_val(sets):
    for e1, e2 in sets:
        f = f_val(e1, e2)
        if os.path.exists(f):
            print(f'chequeo: ya existe {os.path.basename(f)}, se omite')
            R = np.atleast_2d(np.loadtxt(f))
        else:
            print(f'chequeo ε1 = {e1:g}, ε2 = {e2:g}: {len(VAL_POINTS)} puntos '
                  f'(exacto de laboratorio con dos cavidades) ...', flush=True)
            R = np.array(Parallel(n_jobs=N_JOBS)(
                delayed(_val_point)(e1, e2, l1, l2) for l1, l2 in VAL_POINTS))
            os.makedirs(DATA_DIR, exist_ok=True)
            np.savetxt(f, R, fmt='%.12e',
                       header=(f'eps1 = {e1}, eps2 = {e2}, t = {T_VAL}, Delta = {DELTA_VAL}\n'
                               'lambda1  lambda2  F1_PAPS  F1_exact  F1_lab  F1_lab(N+N_CONV)  '
                               'DX1_PAPS  DX1_exact  DX1_lab  DX1_lab(N+N_CONV)  '
                               'gs_PAPS(1=even)  gs_lab(1=even)  dE_lab'))
        print(f'\n  ε1 = {e1:g}, ε2 = {e2:g}')
        print('   λ1    λ2   gs     dE_lab     F1_PAPS     F1_exact    F1_lab   |ex−lab|  |PAPS−lab| |lab−lab+|'
              '   DX1_exact   DX1_lab   |ex−lab|')
        for l1, l2, Fv, Fx, Fl, Fl2, Dv, Dx, Dl, Dl2, gv, gl, dEl in R:
            print(f'  {l1:4.2f}  {l2:4.2f}  {"even" if gl else "odd "}  {dEl:+9.1e}  {Fv:10.6f}  {Fx:10.6f}  {Fl:10.6f}'
                  f'  {abs(Fx - Fl):8.1e}  {abs(Fv - Fl):8.1e}  {abs(Fl - Fl2):8.1e}'
                  f'  {Dx:10.6f}  {Dl:10.6f}  {abs(Dx - Dl):8.1e}'
                  + ('' if gv == gl else '   (PAPS escoge otro sector)'))


def _args():
    if len(sys.argv) == 1:
        return EPS_SETS, 'all'
    if len(sys.argv) < 3:
        sys.exit('uso: python fano_lam1lam2.py [e1 e2 [map | val]]')
    e1, e2 = float(sys.argv[1]), float(sys.argv[2])
    which = sys.argv[3].lower() if len(sys.argv) > 3 else 'all'
    if which not in ('all', 'map', 'val'):
        sys.exit('uso: python fano_lam1lam2.py [e1 e2 [map | val]]')
    return [(e1, e2)], which


if __name__ == '__main__':
    sets, which = _args()
    if which in ('all', 'map'):
        compute_maps(sets)
    if which in ('all', 'val'):
        compute_val(sets)
    print(f'\nDatos en: {DATA_DIR}')
