# -*- coding: utf-8 -*-
"""
NIVEL 2 — optimizadores globales
=================================
¿La continuación o las semillas de producción se quedan atrapadas en un mínimo local?
Ningún método de este nivel usa las semillas ni el arranque caliente de producción.

Para cada caso (odd3, odd6, even6), J/ω0 y Λ, dentro de las cotas de producción:
  * DE      differential evolution (población vectorizada; cada repetición usa otra
            estrategia: best1bin, rand1bin, currenttobest1bin)
  * DA      dual annealing
  * BH      basin-hopping desde un punto aleatorio, pasos acotados
  * GRID    fuerza bruta en malla (3D fina, 6D gruesa) + refinamiento local de los
            mejores nodos distintos
Todos los resultados se pulen al final con el MISMO L-BFGS-B de producción, y se guarda
también la energía antes de pulir ('E_raw'). La población de DE y la malla se evalúan con
el funcional AD vectorizado, idéntico al de producción (common.py --selftest).

Uso:
    python level2_global.py
    python level2_global.py --cases even6 --nrep 3
Resultados: results/level2/<caso>/J<J>/L<Λ>__{DE,DA,BH,GRID}.json
"""

import time

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping

import common as C

METHODS = ['DE', 'DA', 'BH', 'GRID']


def polish(f, x0, bnd):
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    r = minimize(f, np.clip(x0, lo, hi), method='L-BFGS-B', bounds=bnd,
                 options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
    x = np.clip(r.x, lo, hi)
    return float(f(x)), x, int(r.nfev)


def run_DE(case, J, Lam, f, bnd, rep, ad):
    n = C.npar(case)
    vf = lambda X: ad.Evec(np.asarray(X).T, Lam, J)          # scipy pasa X con forma (n, S)
    strategy = ('best1bin', 'rand1bin', 'currenttobest1bin')[rep % 3]
    r = differential_evolution(vf, bnd, strategy=strategy, popsize=30,
                               maxiter=4000 if n == 6 else 2000, tol=1e-12, atol=0.0,
                               mutation=(0.5, 1.0), recombination=0.9, seed=1000 + rep,
                               polish=False, init='latinhypercube', vectorized=True,
                               updating='deferred')
    return float(f(r.x)), np.asarray(r.x, float), int(r.nfev)


def run_DA(case, J, Lam, f, bnd, rep, ad):
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    r = dual_annealing(f, bnd, maxiter=2000, seed=2000 + rep,
                       minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bnd,
                                         'options': {'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000}})
    return float(f(np.clip(r.x, lo, hi))), np.clip(np.asarray(r.x, float), lo, hi), int(r.nfev)


def run_BH(case, J, Lam, f, bnd, rep, ad):
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    rng = np.random.default_rng(3000 + rep)
    x0 = lo + (hi - lo)*rng.random(len(lo))
    width = 0.25*(hi - lo)

    def step(x):
        return np.clip(x + width*rng.uniform(-1.0, 1.0, len(x)), lo, hi)

    r = basinhopping(f, x0, niter=150, T=0.02, take_step=step, seed=3000 + rep,
                     minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bnd,
                                       'options': {'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000}})
    x = np.clip(np.asarray(r.x, float), lo, hi)
    return float(f(x)), x, int(r.nfev)


def run_GRID(case, J, Lam, f, bnd, rep, ad, n3=(121, 61, 41), n6=9, keep=25):
    n = C.npar(case)
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    if n == 3:
        axes = [np.linspace(lo[i], hi[i], n3[i]) for i in range(3)]
    else:
        axes = [np.linspace(lo[i], hi[i], n6) for i in range(6)]
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, n)
    E = ad.Evec(mesh, Lam, J)
    E[~np.isfinite(E)] = np.inf
    order = np.argsort(E)
    cell = np.array([ax[1] - ax[0] for ax in axes])
    picked = []
    for k in order:
        if len(picked) >= keep or not np.isfinite(E[k]):
            break
        if all(np.max(np.abs(mesh[k] - mesh[j])/cell) > 1.5 for j in picked):
            picked.append(k)
    best = (np.inf, None, 0)
    nfev = len(mesh)
    for k in picked:
        Ep, xp, nf = polish(f, mesh[k], bnd)
        nfev += nf
        if Ep < best[0]:
            best = (Ep, xp, nf)
    return float(E[order[0]]), mesh[order[0]], nfev, best


RUNNERS = {'DE': run_DE, 'DA': run_DA, 'BH': run_BH}


def task_point(case, J, Lam, methods, nrep, overwrite):
    f = C.energy_prod(case, J, Lam)
    bnd = C.bounds(case, Lam)
    ad = C.get_ad(case)
    done = []
    for m in methods:
        path = C.point_path('level2', case, J, Lam, m)
        if not overwrite and C.exists(path):
            continue
        t0 = time.time()
        if m == 'GRID':
            E_raw, x_raw, nfev, (Ep, xp, _) = run_GRID(case, J, Lam, f, bnd, 0, ad)
            reps = [dict(E_raw=E_raw, E=Ep)]
            bestE, bestx, bestraw = Ep, xp, E_raw
        else:
            bestE, bestx, bestraw, nfev, reps = np.inf, None, np.inf, 0, []
            for rep in range(nrep):
                E_raw, x_raw, nf = RUNNERS[m](case, J, Lam, f, bnd, rep, ad)
                Ep, xp, nf2 = polish(f, x_raw, bnd)
                nfev += nf + nf2
                reps.append(dict(E_raw=E_raw, E=Ep))
                if Ep < bestE:
                    bestE, bestx, bestraw = Ep, xp, E_raw
        rec = C.full_record(case, J, Lam, bestx, method=m, level='level2', E_raw=bestraw,
                            reps=reps, nfev=nfev, time=time.time() - t0)
        C.save_json(path, rec)
        done.append(m)
    return (case, J, Lam, done)


def main():
    ap = C.base_parser('Nivel 2: optimizadores globales')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--nrep', type=int, default=3, help='repeticiones (semillas aleatorias) de DE, DA y BH')
    args = ap.parse_args()
    cases, Js, lams, chunk = C.parse_common(args)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    tasks = [(c, J, L, methods, args.nrep, args.overwrite) for c in cases for J in Js for L in lams]
    tasks.sort(key=lambda t: (-C.npar(t[0]), -t[2]))
    tasks = C.take_chunk(tasks, chunk)
    C.run_parallel(task_point, tasks, args.cores, 'nivel 2')


if __name__ == '__main__':
    main()
