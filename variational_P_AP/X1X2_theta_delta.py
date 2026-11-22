# -*- coding: utf-8 -*-
"""
DATOS: segundo momento cruzado <X1 X2> en el MARCO RECENTRADO, del estado base, en el plano
(θ, Δ/t), con el ansatz (P+AP)S de variational_P_AP.py (funcional cerrado, minimize_sector,
convención de las notas). Pensado para correr en el clúster (keldysh); la figura la hace
plot_X1X2_theta_delta.py.

    λ1 = Λ cos θ,  λ2 = Λ sin θ,  θ ∈ [0, π/2],  Λ = √2 · (Λ/√2)  fijo
    t = ω0 = 1,  g_i = λ_i ω0
    Marco recentrado: cuadraturas medidas respecto de −λ_i/√2,  X_i^(r) = X_i + λ_i/√2

Convención de pseudoespín (notas):
    impar: |↑> = |10>, |↓> = |01>;   par: |↑> = |11>, |↓> = |00>
Para cada punto se guardan los DOS valores (sector par y sector impar) y δE = E_e − E_o;
plot_X1X2_theta_delta.py escoge con δE cuál corresponde al estado base.

En el marco recentrado <X_B>_r = 0, Var X_B = 1/2 y <X_b X_B>_r = <X_b>_r <X_B>_r = 0, así que
    <X1 X2>_r,{e/o} = ± (λ1 λ2/Λ²) [<X_b²>_r − 1/2]        (+ par, − impar)
con <X_b²>_r el segundo momento de la mezcla de las dos componentes, desplazadas en el modo b
por β↑ = −Λ/2 y β↓ = +Λ/2 (estado recentrado), con <X_b>, <X_b²> en Fock (var_states) y los
pesos c↑, c↓ de weights_2x2.

El sector impar no depende de Δ: se calcula una sola vez (continuación a lo largo de θ).
El par: una fila por θ, con continuación a lo largo de Δ. Cada fila es una tarea de joblib.

VALIDACIÓN (independiente del ansatz y de los modos colectivos), en VAL_THETA × VAL_DELTA:
    [1] X1X2_var  : (P+AP)S + fórmula (lo mismo que el mapa)
    [2] X1X2_col  : estado exacto del bloque H_α (un modo, exact_ground) + la misma fórmula
    [3] X1X2_lab  : diagonalización exacta del hamiltoniano de LABORATORIO con las dos
                    cavidades â1, â2 truncadas;
                    <X1^(r) X2^(r)> = <X1 X2> + (λ2/√2)<X1> + (λ1/√2)<X2> + λ1 λ2/2  directo
    [4] X1X2_lab+ : igual que [3] con N1, N2 + N_CONV (convergencia en la truncación)
    [2] vs [3] valida la fórmula; [1] vs [3] valida el ansatz.
    H_lab = Σ_i [ε_i n_i + ω0 a_i† a_i + g_i n_i (a_i + a_i†)] + J (|↑><↓| + h.c.)
    con J = t (impar) o Δ (par).

Uso:
    python X1X2_theta_delta.py                        # ε1 = ε2 = 1, Λ/√2 = 3 (mapa y validación)
    python X1X2_theta_delta.py eps1 eps2 L            # mapa y validación  (L = Λ/√2)
    python X1X2_theta_delta.py eps1 eps2 L map        # solo el mapa
    python X1X2_theta_delta.py eps1 eps2 L val        # solo la validación
Los .txt se guardan en Figures/paper_X1X2/data/ y no se recalculan si ya existen
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
N_JOBS = -1                                   # núcleos para joblib (-1 = todos)
T_VAL  = 1.0                                  # t (= ω0)

N_TH   = 101                                  # puntos en θ ∈ [0, π/2] (impar: incluye θ = π/4)
N_D    = 101                                  # puntos en Δ/t (impar: incluye Δ = 1 en [0, 2])
D_MIN, D_MAX = 0.0, 2.0                       # rango de Δ/t

THETA = np.linspace(0.0, 0.5*np.pi, N_TH)
DELTA = np.linspace(D_MIN, D_MAX, N_D)

# validación
VAL_THETA = [np.pi/8, np.pi/4, 3*np.pi/8]
VAL_DELTA = [0.5, 1.0, 1.5]
N_MIN, N_MAX, N_CONV = 30, 160, 20            # truncación de cada cavidad en el exacto de laboratorio

DATA_DIR = os.path.join(HERE, 'Figures', 'paper_X1X2', 'data')


EPS1_DEF, EPS2_DEF, L_DEF = 1.0, 1.0, 3.0     # valores por defecto si no se pasan argumentos


def _args():
    if len(sys.argv) == 1:
        return EPS1_DEF, EPS2_DEF, L_DEF, 'all'
    if len(sys.argv) < 4:
        sys.exit('uso: python X1X2_theta_delta.py [eps1 eps2 L [map | val]]    (L = Λ/√2)')
    e1, e2, L = (float(v) for v in sys.argv[1:4])
    which = sys.argv[4].lower() if len(sys.argv) > 4 else 'all'
    if which not in ('all', 'map', 'val'):
        sys.exit('uso: python X1X2_theta_delta.py [eps1 eps2 L [map | val]]    (L = Λ/√2)')
    return e1, e2, L, which


def _lam12(Lam, th):
    return Lam*np.cos(th), Lam*np.sin(th)


# ══════════════════════════════════════════════════════════════════
#  <X1 X2>_r A PARTIR DE LAS COMPONENTES DEL MODO b
# ══════════════════════════════════════════════════════════════════
def _Xb(ops):
    return (ops.b + ops.b.T)/np.sqrt(2.0)


def x1x2_from_components(psi_u, psi_d, sector, lam1, lam2, Lam, Xb):
    """psi_u = c↑|φ↑>, psi_d = c↓|φ↓> (componentes del modo b en Lang–Firsov, con su peso).
    Devuelve (<X1 X2>_r, <X_b²>_r) en el marco recentrado.
    Componente σ desplazada por β_σ en el modo b: β↑ = −Λ/2, β↓ = +Λ/2.
    Con D†(β) X_b D(β) = X_b + √2 β:
        <X_b²>_r = Σ_σ [<X_b²>_σ + 2√2 β_σ <X_b>_σ + 2 β_σ² p_σ]
    (<·>_σ sin normalizar, p_σ = (c_σ)²)."""
    X2 = 0.0
    for psi, beta in ((psi_u, -0.5*Lam), (psi_d, 0.5*Lam)):
        Xp = Xb @ psi
        p, m1, m2 = psi @ psi, psi @ Xp, Xp @ Xp
        X2 += m2 + 2.0*np.sqrt(2.0)*beta*m1 + 2.0*beta*beta*p
    sgn = 1.0 if sector == 'even' else -1.0
    return sgn*lam1*lam2/Lam**2*(X2 - 0.5), X2


def x1x2_variational(sector, x, eps, epsp, J, Lam, lam1, lam2, ops, Xb):
    A, B, C = closed_ABC(x, eps, epsp, J, Lam)
    if C == 0.0 and abs(A - B) < 1e-10*max(1.0, abs(A)):
        cu, cd = np.sqrt(0.5), np.sqrt(0.5)     # bloque degenerado (J = 0, δ_α = 0): límite J -> 0⁺
    else:
        cu, cd = weights_2x2(A, B, C)
    pu, pd  = var_states(x, Lam, ops)
    return x1x2_from_components(cu*pu, cd*pd, sector, lam1, lam2, Lam, Xb)


# ══════════════════════════════════════════════════════════════════
#  FILAS DEL MAPA (una tarea de joblib cada una, con continuación)
# ══════════════════════════════════════════════════════════════════
def _row_odd(e1, e2, Lam):
    """Sector impar a lo largo de θ (no depende de Δ): E_o(θ), <X1 X2>_r,o(θ)."""
    ops = get_ops(_Mof_Lam(Lam))
    Xb  = _Xb(ops)
    E, Cv = np.empty(N_TH), np.empty(N_TH)
    w = None
    for k, th in enumerate(THETA):
        lam1, lam2 = _lam12(Lam, th)
        eps, epsp, J = sector_params('odd', lam1, lam2, e1, e2, T_VAL, 0.0)
        E[k], w = minimize_sector('odd', eps, epsp, J, Lam, w)
        Cv[k], _ = x1x2_variational('odd', w, eps, epsp, J, Lam, lam1, lam2, ops, Xb)
    return E, Cv


def _row_even(e1, e2, Lam, th):
    """Sector par a lo largo de Δ, con θ fijo: E_e(Δ), <X1 X2>_r,e(Δ)."""
    ops = get_ops(_Mof_Lam(Lam))
    Xb  = _Xb(ops)
    lam1, lam2 = _lam12(Lam, th)
    E, Cv = np.empty(N_D), np.empty(N_D)
    w = None
    for k, D in enumerate(DELTA):
        eps, epsp, J = sector_params('even', lam1, lam2, e1, e2, T_VAL, D)
        E[k], w = minimize_sector('even', eps, epsp, J, Lam, w)
        Cv[k], _ = x1x2_variational('even', w, eps, epsp, J, Lam, lam1, lam2, ops, Xb)
    return E, Cv


# ══════════════════════════════════════════════════════════════════
#  EXACTO DE LABORATORIO (dos cavidades truncadas)
# ══════════════════════════════════════════════════════════════════
def _N_of_lam(lam):
    return int(min(N_MAX, max(N_MIN, np.ceil((abs(lam) + 6.0)**2))))


def lab_ground(sector, lam1, lam2, e1, e2, J, N1, N2):
    """Estado base exacto del bloque de paridad en el laboratorio y <X1^(r) X2^(r)> directo."""
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
    X1 = (x1/np.sqrt(2.0)).toarray()
    X2 = (x2/np.sqrt(2.0)).toarray()
    m1 = sum(np.sum(p*(X1 @ p)) for p in psi)
    m2 = sum(np.sum(p*(p @ X2)) for p in psi)
    m12 = sum(np.sum(p*(X1 @ p @ X2)) for p in psi)
    return w[0], m12 + lam2/np.sqrt(2.0)*m1 + lam1/np.sqrt(2.0)*m2 + 0.5*lam1*lam2


def _val_point(e1, e2, Lam, th, D, sector):
    lam1, lam2 = _lam12(Lam, th)
    J_D = T_VAL if sector == 'odd' else D
    eps, epsp, J = sector_params(sector, lam1, lam2, e1, e2, T_VAL, J_D)
    M   = _Mof_Lam(Lam)
    ops = get_ops(M)
    Xb  = _Xb(ops)
    # [1] variacional
    E_var, x = minimize_sector(sector, eps, epsp, J, Lam)
    C_var, _ = x1x2_variational(sector, x, eps, epsp, J, Lam, lam1, lam2, ops, Xb)
    # [2] exacto del bloque (un modo) + fórmula
    E_col, v = exact_ground(eps, epsp, J, Lam, M)
    C_col, _ = x1x2_from_components(v[:M + 1], v[M + 1:], sector, lam1, lam2, Lam, Xb)
    # [3], [4] exacto de laboratorio
    N1, N2 = _N_of_lam(lam1), _N_of_lam(lam2)
    E_lab, C_lab = lab_ground(sector, lam1, lam2, e1, e2, J, N1, N2)
    _, C_lab2    = lab_ground(sector, lam1, lam2, e1, e2, J, N1 + N_CONV, N2 + N_CONV)
    return [th, D, 0.0 if sector == 'odd' else 1.0, E_var, E_col, E_lab,
            C_var, C_col, C_lab, C_lab2, N1, N2]


# ══════════════════════════════════════════════════════════════════
#  ARCHIVOS (todos los parámetros y mallas en el nombre)
# ══════════════════════════════════════════════════════════════════
def _tag(e1, e2, L):
    return f'eps1{e1:g}_eps2{e2:g}_L{L:g}_t{T_VAL:g}'


def f_odd(e1, e2, L):
    return os.path.join(DATA_DIR, f'odd_{_tag(e1, e2, L)}_nth{N_TH}_PAPS.txt')


def f_even(kind, e1, e2, L):
    """kind = 'E' (energía) o 'X1X2'; filas = Δ/t, columnas = θ."""
    return os.path.join(DATA_DIR, f'even_{kind}_{_tag(e1, e2, L)}_nth{N_TH}'
                                  f'_D{D_MIN:g}-{D_MAX:g}_nD{N_D}_PAPS.txt')


def f_val(e1, e2, L):
    return os.path.join(DATA_DIR, f'validation_{_tag(e1, e2, L)}_PAPS.txt')


# ══════════════════════════════════════════════════════════════════
#  CÁLCULO
# ══════════════════════════════════════════════════════════════════
def compute_map(e1, e2, L):
    Lam = np.sqrt(2.0)*L
    fo, fE, fC = f_odd(e1, e2, L), f_even('E', e1, e2, L), f_even('X1X2', e1, e2, L)
    if all(os.path.exists(f) for f in (fo, fE, fC)):
        print(f'mapa: ya existen {os.path.basename(fo)} y los del sector par, se omite')
        return
    print(f'mapa: impar ({N_TH} θ) + par ({N_TH} filas × {N_D} Δ), Λ = {Lam:.6f} ...', flush=True)
    jobs = ([delayed(_row_odd)(e1, e2, Lam)]
            + [delayed(_row_even)(e1, e2, Lam, th) for th in THETA])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(jobs)
    E_o, C_o = res[0]
    E_e = np.array([r[0] for r in res[1:]]).T                        # filas = Δ, columnas = θ
    C_e = np.array([r[1] for r in res[1:]]).T
    os.makedirs(DATA_DIR, exist_ok=True)
    hdr = f'eps1 = {e1}, eps2 = {e2}, Lambda/sqrt2 = {L}, Lambda = {Lam}, t = {T_VAL}, omega0 = {omega0}'
    np.savetxt(fo, np.column_stack([THETA, E_o, C_o]), fmt='%.12e',
               header=f'sector impar (no depende de Delta) | {hdr}\ntheta  E_odd  <X1 X2>_r,odd')
    for f, M, what in ((fE, E_e, 'E_even'), (fC, C_e, '<X1 X2>_r,even')):
        np.savetxt(f, M, fmt='%.12e',
                   header=(f'{what} | filas = Delta/t ({N_D}, {D_MIN}..{D_MAX}), columnas = theta '
                           f'({N_TH}, 0..pi/2) | {hdr}'))
    dE = E_e - E_o[None, :]
    print(f'mapa: guardado  δE ∈ [{dE.min():.4f}, {dE.max():.4f}],  '
          f'X1X2_e ∈ [{C_e.min():.4f}, {C_e.max():.4f}],  X1X2_o ∈ [{C_o.min():.4f}, {C_o.max():.4f}]')


def compute_val(e1, e2, L):
    Lam = np.sqrt(2.0)*L
    f = f_val(e1, e2, L)
    if os.path.exists(f):
        print(f'validación: ya existe {os.path.basename(f)}, se omite')
        R = np.loadtxt(f)
    else:
        pts = ([(th, np.nan, 'odd') for th in VAL_THETA]
               + [(th, D, 'even') for th in VAL_THETA for D in VAL_DELTA])
        print(f'validación: {len(pts)} puntos (exacto de laboratorio con dos cavidades) ...', flush=True)
        R = np.array(Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_val_point)(e1, e2, Lam, th, D, s) for th, D, s in pts))
        os.makedirs(DATA_DIR, exist_ok=True)
        np.savetxt(f, R, fmt='%.12e',
                   header=(f'eps1 = {e1}, eps2 = {e2}, Lambda/sqrt2 = {L}, Lambda = {Lam}, t = {T_VAL}\n'
                           'theta  Delta  sector(0=odd,1=even)  E_var  E_col  E_lab  '
                           'X1X2_var  X1X2_col  X1X2_lab  X1X2_lab(N+N_CONV)  N1  N2'))
    print('\n  θ/π     Δ/t   sec      E_var        E_lab      X1X2_var     X1X2_col     X1X2_lab'
          '    |col−lab|   |var−lab|   |lab−lab+|')
    for th, D, s, Ev, Ec, El, Cv, Cc, Cl, Cl2, N1, N2 in R:
        print(f'  {th/np.pi:5.3f}  {"  —  " if s == 0 else f"{D:5.2f}"}  {"odd " if s == 0 else "even"}'
              f'  {Ev:11.6f}  {El:11.6f}  {Cv:+11.6f}  {Cc:+11.6f}  {Cl:+11.6f}'
              f'   {abs(Cc - Cl):9.2e}   {abs(Cv - Cl):9.2e}   {abs(Cl - Cl2):9.2e}')


if __name__ == '__main__':
    e1, e2, L, which = _args()
    if which in ('all', 'map'):
        compute_map(e1, e2, L)
    if which in ('all', 'val'):
        compute_val(e1, e2, L)
    print(f'\nDatos en: {DATA_DIR}')
