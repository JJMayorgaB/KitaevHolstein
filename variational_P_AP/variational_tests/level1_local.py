# -*- coding: utf-8 -*-
"""
NIVEL 1 — optimizadores locales
================================
¿El mínimo depende del optimizador local o del gradiente por diferencias finitas?

Para cada caso (odd3, odd6, even6), J/ω0 y Λ:
  * con gradiente (diferencias finitas, como en producción): L-BFGS-B, SLSQP, trust-constr
  * sin derivadas: Nelder-Mead, Powell, COBYLA
  * 'prod': el minimizador de producción (P._run_seeds / P.minimize_sector) en frío
Cada método arranca desde TODAS las semillas de producción (en 6D los métodos sin
derivadas usan las primeras --dfree-seeds) y se reinicia desde su propio resultado
hasta que la energía deja de bajar. Se guarda el mejor resultado de cada método con
sus cantidades físicas, KKT y hessiano exacto (AD), y la energía de cada semilla.
COBYLA en 6D cuesta ~1.5 ms por evaluación: se corre con --cobyla-seeds semillas, sin
reinicios, en uno de cada --cobyla-every valores de Λ.

Además, por (caso, J), 'prod_warm' reproduce el barrido de producción con arranque
caliente (continuación en Λ), que es lo que generan las figuras.

Uso:
    python level1_local.py                       # todo
    python level1_local.py --cases odd3 --J 1.0  # parte
    python level1_local.py --quick               # prueba rápida
Resultados: results/level1/<caso>/J<J>/L<Λ>__<método>.json
"""

import time

import numpy as np
from scipy.optimize import minimize, Bounds

import common as C

GRAD_METHODS  = ['L-BFGS-B', 'SLSQP', 'trust-constr']
DFREE_METHODS = ['Nelder-Mead', 'Powell', 'COBYLA']
ALL_METHODS   = GRAD_METHODS + DFREE_METHODS


def _once(method, f, x0, bnd, n):
    lo = np.array([b[0] for b in bnd]); hi = np.array([b[1] for b in bnd])
    x0 = np.clip(x0, lo, hi)
    if method == 'L-BFGS-B':
        r = minimize(f, x0, method=method, bounds=bnd,
                     options={'ftol': 1e-15, 'gtol': 1e-11, 'maxiter': 1000})
    elif method == 'SLSQP':
        r = minimize(f, x0, method=method, bounds=bnd,
                     options={'ftol': 1e-15, 'maxiter': 1000})
    elif method == 'trust-constr':
        r = minimize(f, x0, method=method, jac='3-point', bounds=Bounds(lo, hi),
                     options={'gtol': 1e-12, 'xtol': 1e-14, 'maxiter': 3000})
    elif method == 'Nelder-Mead':
        r = minimize(f, x0, method=method, bounds=bnd,
                     options={'xatol': 1e-12, 'fatol': 1e-16, 'maxfev': 6000*n, 'adaptive': True})
    elif method == 'Powell':
        r = minimize(f, x0, method=method, bounds=bnd,
                     options={'xtol': 1e-12, 'ftol': 1e-16, 'maxfev': 6000*n})
    elif method == 'COBYLA':
        r = minimize(f, x0, method=method, bounds=bnd,
                     options={'rhobeg': 0.05, 'tol': 1e-12, 'maxiter': 6000*n})
    else:
        raise ValueError(method)
    x = np.clip(np.asarray(r.x, float), lo, hi)
    return float(f(x)), x, int(getattr(r, 'nfev', 0))


def run_method(method, f, x0, bnd, n, nmax=8):
    """Minimiza y reinicia desde el propio resultado hasta que E deja de bajar."""
    E, x, nf = _once(method, f, x0, bnd, n)
    nrest = 0
    for _ in range(nmax - 1):
        E2, x2, nf2 = _once(method, f, x, bnd, n)
        nf += nf2
        nrest += 1
        if E2 < E - 1e-15:
            E, x = E2, x2
        else:
            break
    return E, x, nf, nrest


def task_point(case, J, Lam, methods, dfree_seeds, overwrite, cobyla6, cobyla_seeds):
    n = C.npar(case)
    f = C.energy_prod(case, J, Lam)
    bnd = C.bounds(case, Lam)
    sds = C.seeds(case, Lam)
    done = []
    for m in methods:
        path = C.point_path('level1', case, J, Lam, m)
        if not overwrite and C.exists(path):
            continue
        if m == 'COBYLA' and n == 6 and not cobyla6:
            continue                              # COBYLA en 6D solo en la submalla (muy lento)
        if m in GRAD_METHODS or n == 3:
            use, nmax = sds, 8
        elif m == 'COBYLA':
            use, nmax = sds[:cobyla_seeds], 1
        else:
            use, nmax = sds[:dfree_seeds], 3
        t0 = time.time()
        bestE, bestx, nf_tot, per_seed = np.inf, None, 0, []
        for sd in use:
            E, x, nf, _ = run_method(m, f, sd, bnd, n, nmax=nmax)
            nf_tot += nf
            per_seed.append(E)
            if E < bestE:
                bestE, bestx = E, x
        rec = C.full_record(case, J, Lam, bestx, method=m, level='level1', nfev=nf_tot,
                            time=time.time() - t0, seed_E=per_seed, n_seeds=len(use))
        C.save_json(path, rec)
        done.append(m)
    # producción en frío
    path = C.point_path('level1', case, J, Lam, 'prod')
    if overwrite or not C.exists(path):
        eps, epsp, Jb = C.block(case, J, Lam)
        t0 = time.time()
        if case == 'odd6':                       # producción no tiene 'odd6': mismo minimizador, 6 parámetros
            E, x = C.P._run_seeds(C.P.energy_closed, (eps, epsp, Jb, Lam), Lam, 6, None, 'odd')
        else:
            E, xf = C.P.minimize_sector(C.sector_of(case), eps, epsp, Jb, Lam)
            x = xf if n == 6 else np.array([xf[0], xf[2], xf[4]])
        C.save_json(path, C.full_record(case, J, Lam, x, method='prod', level='level1',
                                        time=time.time() - t0))
        done.append('prod')
    return (case, J, Lam, done)


def task_warm(case, J, lams, overwrite):
    """Barrido de producción con continuación en Λ (arranque caliente), en orden creciente."""
    tag = 'prod_warm'
    if not overwrite and all(C.exists(C.point_path('level1', case, J, L, tag)) for L in lams):
        return (case, J, 'warm', [])
    warm = None
    for Lam in sorted(lams):
        eps, epsp, Jb = C.block(case, J, Lam)
        t0 = time.time()
        if case == 'odd6':
            xw = None if warm is None else tuple(warm)
            E, x = C.P._run_seeds(C.P.energy_closed, (eps, epsp, Jb, Lam), Lam, 6, xw, 'odd')
            xf = x
        else:
            E, xf = C.P.minimize_sector(C.sector_of(case), eps, epsp, Jb, Lam, warm)
            x = xf if C.npar(case) == 6 else np.array([xf[0], xf[2], xf[4]])
        warm = xf if case != 'odd6' else x
        C.save_json(C.point_path('level1', case, J, Lam, tag),
                    C.full_record(case, J, Lam, x, method=tag, level='level1', time=time.time() - t0))
    return (case, J, 'warm', [tag])


def main():
    ap = C.base_parser('Nivel 1: optimizadores locales')
    ap.add_argument('--methods', default=','.join(ALL_METHODS))
    ap.add_argument('--dfree-seeds', type=int, default=8,
                    help='semillas para Nelder-Mead/Powell/COBYLA en 6D (en 3D se usan todas)')
    ap.add_argument('--cobyla-every', type=int, default=3,
                    help='en 6D, COBYLA solo en uno de cada N valores de Λ (~1.5 ms por evaluación)')
    ap.add_argument('--cobyla-seeds', type=int, default=3, help='semillas de COBYLA en 6D')
    ap.add_argument('--no-warm', action='store_true', help='omite el barrido prod_warm')
    args = ap.parse_args()
    cases, Js, lams, chunk = C.parse_common(args)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    tasks = [(c, J, L, methods, args.dfree_seeds, args.overwrite,
              (k % max(args.cobyla_every, 1) == 0), args.cobyla_seeds)
             for c in cases for J in Js for k, L in enumerate(lams)]
    # los puntos más caros (6D, Λ grande) primero: mejor reparto entre núcleos
    tasks.sort(key=lambda t: (-C.npar(t[0]), -t[2]))
    tasks = C.take_chunk(tasks, chunk)
    C.run_parallel(task_point, tasks, args.cores, 'nivel 1 / puntos')
    if not args.no_warm and chunk[0] == 1:
        wt = [(c, J, lams, args.overwrite) for c in cases for J in Js]
        C.run_parallel(task_warm, wt, args.cores, 'nivel 1 / prod_warm')


if __name__ == '__main__':
    main()
