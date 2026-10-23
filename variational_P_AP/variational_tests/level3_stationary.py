# -*- coding: utf-8 -*-
"""
NIVEL 3 — ecuaciones variacionales (sin optimizador)
=====================================================
Gradiente, hessiano y ∂_Λ∇E exactos por diferenciación automática (common.AD).

  ROOTS  puntos estacionarios: ∇E(x) = 0 resuelto con buscadores de raíces (hybr, y
         Levenberg–Marquardt si falla) desde las semillas de producción + puntos
         aleatorios. También los puntos KKT en la cota κ = 0 (sistema reducido con κ fijo
         y multiplicador ∂E/∂κ ≥ 0). Cada punto se clasifica con los autovalores del
         hessiano (mínimo, mínimo degenerado, silla, máximo) y se marca si está dentro
         del dominio de producción. Mínimo global = el menor mínimo dentro del dominio.

  GFLOW  flujo gradiente  ẋ = −∇E(x)  integrado (BDF, jacobiano exacto) hasta |∇E| < 1e-10.
         κ = s² garantiza κ ≥ 0 sin cotas; no depende de ningún criterio de parada de un
         optimizador.

  DAV    continuación de Davidenko por pseudo-longitud de arco:
             ∇E(x(s), Λ(s)) = 0,   d/ds ∇E = H ẋ + ∂_Λ∇E Λ̇ = 0,
         equivalente a dx/dΛ = −H⁻¹ ∂_Λ∇E fuera de los pliegues, pero también sigue las
         ramas que se doblan. Arranca de todos los mínimos encontrados por ROOTS en unos
         Λ de anclaje y sigue cada rama hacia ambos lados. Registra los cambios de signo
         de autovalores del hessiano (pliegues / bifurcaciones). En cada Λ de la malla se
         guarda el menor estado estable dentro del dominio.

Uso:
    python level3_stationary.py                       # ROOTS + GFLOW por punto, luego DAV
    python level3_stationary.py --parts roots,dav
Resultados: results/level3/<caso>/J<J>/L<Λ>__{ROOTS,GFLOW,DAV}.json
            results/level3/<caso>/J<J>/davidenko.json   (todas las ramas)
"""

import itertools
import time
import zlib

import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp

import common as C

TOL_G   = 1e-9      # |∇E|∞ para aceptar un punto estacionario
TOL_EIG = 1e-9      # |λ| por debajo de esto: dirección degenerada
XBIG    = 60.0      # |x| mayor que esto: el buscador se escapó


# ══════════════════════════════════════════════════════════════════
#  PUNTOS ESTACIONARIOS
# ══════════════════════════════════════════════════════════════════
def _rng(case, J, Lam, salt=0):
    return np.random.default_rng(zlib.crc32(f'{case}|{J:.4f}|{Lam:.4f}|{salt}'.encode()))


def classify(ev):
    ev = np.asarray(ev)
    if ev.size == 0:
        return 'min'
    scale = max(1.0, float(np.max(np.abs(ev))))
    nneg = int(np.sum(ev < -TOL_EIG*scale))
    if nneg == 0:
        return 'min' if ev.min() > TOL_EIG*scale else 'degmin'
    return 'max' if nneg == ev.size else 'saddle'


def solve_stationary(ad, J, Lam, x0, fixed):
    """∇_free E = 0 con las variables 'fixed' (κ) fijas en 0."""
    n = len(x0)
    free = np.array([i for i in range(n) if i not in fixed], int)
    base = np.asarray(x0, float).copy()
    base[list(fixed)] = 0.0

    def full(z):
        x = base.copy(); x[free] = z; return x

    def g(z):
        return ad.grad(full(z), Lam, J)[free]

    def h(z):
        return ad.hess(full(z), Lam, J)[np.ix_(free, free)]

    z = base[free]
    best = None
    for meth in ('hybr', 'lm'):
        try:
            r = root(g, z, jac=h, method=meth, options={'xtol': 1e-14} if meth == 'hybr' else {'xtol': 1e-15, 'ftol': 1e-15})
            zz = np.asarray(r.x, float)
            if not np.all(np.isfinite(zz)) or np.max(np.abs(zz)) > XBIG:
                continue
            # pulido de Newton
            for _ in range(6):
                gz = g(zz)
                if np.max(np.abs(gz)) < 1e-13:
                    break
                try:
                    zz = zz - np.linalg.solve(h(zz), gz)
                except np.linalg.LinAlgError:
                    zz = zz - np.linalg.lstsq(h(zz), gz, rcond=None)[0]
                if not np.all(np.isfinite(zz)) or np.max(np.abs(zz)) > XBIG:
                    break
            if not np.all(np.isfinite(zz)) or np.max(np.abs(zz)) > XBIG:
                continue
            res = float(np.max(np.abs(g(zz))))
            if best is None or res < best[1]:
                best = (zz, res)
            if res < TOL_G:
                break
        except Exception:
            continue
    if best is None or best[1] >= TOL_G:
        return None
    return full(best[0])


def stationary_points(case, J, Lam, nrand):
    ad = C.get_ad(case)
    n = C.npar(case)
    lo, hi = C.lohi(case, Lam)
    rng = _rng(case, J, Lam)
    starts = C.seeds(case, Lam) + [lo + (hi - lo)*rng.random(n) for _ in range(nrand)]
    kidx = C.KAP_IDX[n]
    subsets = [()] + [s for k in range(1, len(kidx) + 1) for s in itertools.combinations(kidx, k)]
    found = []
    for x0 in starts:
        for S in subsets:
            x = solve_stationary(ad, J, Lam, x0, S)
            if x is None:
                continue
            E = ad.E(x, Lam, J)
            if any(abs(E - q['E']) < 1e-9 and np.max(np.abs(x - np.array(q['x']))) < 1e-3 for q in found):
                continue
            g = ad.grad(x, Lam, J)
            H = ad.hess(x, Lam, J)
            free = [i for i in range(n) if i not in S]
            # sin complementariedad estricta (∂E/∂κ ≈ 0 en κ = 0) la curvatura en κ también cuenta
            chk = sorted(free + [i for i in S if abs(g[i]) < 1e-8])
            ev = np.linalg.eigvalsh(H[np.ix_(chk, chk)])
            in_dom = bool(np.all(x >= lo - 1e-8) and np.all(x <= hi + 1e-8))
            kkt = bool(all(g[i] >= -1e-6 for i in S))      # multiplicador ≥ 0 (tolerancia del gradiente FD de producción)
            found.append(dict(x=x.tolist(), E=float(E), fixed=list(S), type=classify(ev),
                              eig=ev.tolist(), in_domain=in_dom, kkt_ok=kkt,
                              gnorm=float(np.max(np.abs(g[free])))))
    found.sort(key=lambda q: q['E'])
    return found, len(starts)*len(subsets)


def is_local_min(q):
    return q['in_domain'] and q['kkt_ok'] and q['type'] in ('min', 'degmin')


def task_roots(case, J, Lam, nrand, overwrite):
    path = C.point_path('level3', case, J, Lam, 'ROOTS')
    if not overwrite and C.exists(path):
        return (case, J, Lam, 'ROOTS', 'skip')
    t0 = time.time()
    found, nsolve = stationary_points(case, J, Lam, nrand)
    mins = [q for q in found if is_local_min(q)]
    if mins:
        best = mins[0]
        rec = C.full_record(case, J, Lam, best['x'], method='ROOTS', level='level3')
    else:                                   # no debería pasar: se deja registro vacío
        rec = dict(case=case, J=float(J), Lam=float(Lam), method='ROOTS', level='level3', E=np.nan)
    rec.update(stationary=found, n_stationary=len(found), n_minima=len(mins),
               minima_E=[q['E'] for q in mins], n_solves=nsolve, time=time.time() - t0)
    C.save_json(path, rec)
    return (case, J, Lam, 'ROOTS', len(found))


# ══════════════════════════════════════════════════════════════════
#  FLUJO GRADIENTE
# ══════════════════════════════════════════════════════════════════
def gradient_flow(ad, case, J, Lam, x0, tmax=1e12, budget=40.0):
    """Integra ẋ = −∇E por tramos t ∈ [t_k, 10 t_k] hasta |∇E| < 10^-10.5, t = tmax o
    agotar 'budget' segundos de cómputo (en los valles casi planos λ_min ~ 1e-8 y el
    flujo necesita t ~ 1e9). Devuelve también si convergió."""
    n = C.npar(case)
    kidx = C.KAP_IDX[n]
    y = np.asarray(x0, float).copy()
    y[kidx] = np.sqrt(np.maximum(y[kidx], 0.0) + 1e-6)   # κ = s² (s > 0 para no quedar en el punto fijo artificial s = 0)

    def x_of(y):
        x = y.copy(); x[kidx] = y[kidx]**2; return x

    def rhs(t, y):
        g = ad.grad(x_of(y), Lam, J)
        d = np.ones(n); d[kidx] = 2.0*y[kidx]
        return -g*d

    def jac(t, y):
        x = x_of(y)
        g = ad.grad(x, Lam, J)
        H = ad.hess(x, Lam, J)
        d = np.ones(n); d[kidx] = 2.0*y[kidx]
        Hy = (d[:, None]*H)*d[None, :]
        Hy[kidx, kidx] += 2.0*g[kidx]
        return -Hy

    def ev_conv(t, y):
        return np.log10(np.max(np.abs(rhs(t, y))) + 1e-300) + 10.5
    ev_conv.terminal = True
    ev_conv.direction = -1

    def ev_escape(t, y):
        return XBIG - np.max(np.abs(y))
    ev_escape.terminal = True

    t0w = time.time()
    t, t_end, nfev, conv = 0.0, 1.0, 0, False
    while t < tmax and time.time() - t0w < budget:
        sol = solve_ivp(rhs, (t, t_end), y, method='BDF', jac=jac, rtol=1e-9, atol=1e-13,
                        events=(ev_conv, ev_escape))
        nfev += int(sol.nfev)
        y, t = sol.y[:, -1], float(sol.t[-1])
        if len(sol.t_events[0]) > 0:
            conv = True
            break
        if len(sol.t_events[1]) > 0 or not sol.success:
            break
        t_end = min(10.0*t_end, tmax)
    return x_of(y), t, nfev, conv


def task_gflow(case, J, Lam, nseeds, budget, overwrite):
    path = C.point_path('level3', case, J, Lam, 'GFLOW')
    if not overwrite and C.exists(path):
        return (case, J, Lam, 'GFLOW', 'skip')
    ad = C.get_ad(case)
    lo, hi = C.lohi(case, Lam)
    t0 = time.time()
    runs = []
    for x0 in C.seeds(case, Lam)[:nseeds]:
        try:
            x, tf, nf, conv = gradient_flow(ad, case, J, Lam, x0, budget=budget)
        except Exception as e:
            runs.append(dict(ok=False, err=str(e)))
            continue
        in_dom = bool(np.all(x >= lo - 1e-8) and np.all(x <= hi + 1e-8))
        runs.append(dict(ok=True, x=x.tolist(), E=ad.E(x, Lam, J), t_final=tf, nfev=nf,
                         converged=conv, in_domain=in_dom))
    good = [q for q in runs if q.get('ok') and q['in_domain']]
    if good:
        best = min(good, key=lambda q: q['E'])
        rec = C.full_record(case, J, Lam, best['x'], method='GFLOW', level='level3',
                            t_final=best['t_final'], converged=best['converged'])
    else:
        rec = dict(case=case, J=float(J), Lam=float(Lam), method='GFLOW', level='level3', E=np.nan)
    rec.update(runs=runs, time=time.time() - t0)
    C.save_json(path, rec)
    return (case, J, Lam, 'GFLOW', len(good))


# ══════════════════════════════════════════════════════════════════
#  CONTINUACIÓN DE DAVIDENKO (pseudo-longitud de arco)
# ══════════════════════════════════════════════════════════════════
def _tangent(ad, x, L, J, t_prev=None, direction=+1):
    H = ad.hess(x, L, J)
    gL = ad.dgrad_dLam(x, L, J)
    Jy = np.hstack([H, gL[:, None]])
    t = np.linalg.svd(Jy)[2][-1]
    if t_prev is not None:
        if t @ t_prev < 0:
            t = -t
    elif t[-1]*direction < 0:
        t = -t
    return t / np.linalg.norm(t)


def _newton_fixed_L(ad, x, L, J, it=30):
    for _ in range(it):
        g = ad.grad(x, L, J)
        if np.max(np.abs(g)) < 1e-12:
            break
        H = ad.hess(x, L, J)
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(H, g, rcond=None)[0]
        x = x - dx
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > XBIG:
            return None
    return x if np.max(np.abs(ad.grad(x, L, J))) < 1e-9 else None


def continue_branch(ad, case, J, x0, L0, direction, lmin, lmax, grid,
                    ds0=0.01, dsmax=0.05, dsmin=1e-7, nmax=40000):
    n = len(x0)
    y = np.concatenate([np.asarray(x0, float), [L0]])
    t = _tangent(ad, y[:n], y[n], J, None, direction)
    ds = ds0
    path, gridpts, events = [], [], []
    ev_prev = np.linalg.eigvalsh(ad.hess(y[:n], y[n], J))
    path.append(dict(L=float(y[n]), E=ad.E(y[:n], y[n], J), lam_min=float(ev_prev.min()),
                     nneg=int(np.sum(ev_prev < -TOL_EIG))))
    status = 'range'
    for step in range(nmax):
        yp = y + ds*t
        yy = yp.copy()
        ok = False
        for it in range(12):
            x, L = yy[:n], yy[n]
            G = ad.grad(x, L, J)
            Fv = np.concatenate([G, [t @ (yy - yp)]])
            Jac = np.vstack([np.hstack([ad.hess(x, L, J), ad.dgrad_dLam(x, L, J)[:, None]]), t[None, :]])
            try:
                dy = np.linalg.solve(Jac, -Fv)
            except np.linalg.LinAlgError:
                dy = np.linalg.lstsq(Jac, -Fv, rcond=None)[0]
            yy = yy + dy
            if not np.all(np.isfinite(yy)) or np.max(np.abs(yy[:n])) > XBIG:
                break
            res = np.max(np.abs(ad.grad(yy[:n], yy[n], J)))
            if res < 1e-11 or (np.linalg.norm(dy) < 1e-10 and res < 1e-9):
                ok = True
                break
        if ok and np.linalg.norm(yy - y) > 3.0*ds:
            ok = False                                  # saltó a otra rama
        if not ok:
            ds *= 0.5
            if ds < dsmin:
                status = 'stalled'
                break
            continue
        L_old, x_old = y[n], y[:n]
        y_new = yy
        x_new, L_new = y_new[:n], y_new[n]
        ev = np.linalg.eigvalsh(ad.hess(x_new, L_new, J))
        nneg_old, nneg_new = int(np.sum(ev_prev < -TOL_EIG)), int(np.sum(ev < -TOL_EIG))
        if nneg_new != nneg_old:
            events.append(dict(L=float(L_new), nneg_from=nneg_old, nneg_to=nneg_new,
                               fold=bool(np.sign(t[n]) != np.sign(_tangent(ad, x_new, L_new, J, t)[n]))))
        # cruces de la malla
        for Lg in grid:
            if (L_old - Lg)*(L_new - Lg) < 0 or (Lg == L_new):
                w = (Lg - L_old)/(L_new - L_old) if L_new != L_old else 1.0
                xg = _newton_fixed_L(ad, x_old + w*(x_new - x_old), Lg, J)
                if xg is not None:
                    evg = np.linalg.eigvalsh(ad.hess(xg, Lg, J))
                    lo, hi = C.lohi(case, Lg)
                    gridpts.append(dict(L=float(Lg), x=xg.tolist(), E=ad.E(xg, Lg, J),
                                        eig=evg.tolist(), type=classify(evg),
                                        in_domain=bool(np.all(xg >= lo - 1e-8) and np.all(xg <= hi + 1e-8))))
        t = _tangent(ad, x_new, L_new, J, t)
        y, ev_prev = y_new, ev
        path.append(dict(L=float(L_new), E=ad.E(x_new, L_new, J), lam_min=float(ev.min()), nneg=nneg_new))
        if it <= 3:
            ds = min(ds*1.5, dsmax)
        if L_new < lmin - 1e-9 or L_new > lmax + 1e-9:
            status = 'range'
            break
    else:
        status = 'nmax'
    return dict(start_L=float(L0), direction=int(direction), status=status, nsteps=len(path),
                path=path[::max(1, len(path)//2000)], grid=gridpts, events=events)


def task_dav(case, J, lams, anchors, nrand, overwrite):
    bpath = C.point_path('level3', case, J, 0.0, 'x').replace('L000.0000__x.json', 'davidenko.json')
    if not overwrite and C.exists(bpath):
        return (case, J, 'DAV', 'skip')
    ad = C.get_ad(case)
    t0 = time.time()
    lmin, lmax = float(min(lams)), float(max(lams))
    starts = []
    for La in anchors:
        La = float(lams[np.argmin(np.abs(np.asarray(lams) - La))])
        rp = C.point_path('level3', case, J, La, 'ROOTS')
        if C.exists(rp):
            st = C.load_json(rp).get('stationary', [])
        else:
            st, _ = stationary_points(case, J, La, nrand)
        # mínimos interiores (κ libre). Los de la cota κ = 0 no son raíces de ∇E completo.
        for q in st:
            if q['type'] in ('min', 'degmin') and q['in_domain'] and not q['fixed']:
                starts.append((La, np.array(q['x'])))
    branches = []
    for La, x0 in starts:
        for d in (+1, -1):
            try:
                branches.append(continue_branch(ad, case, J, x0, La, d, lmin, lmax, list(lams)))
            except Exception as e:
                branches.append(dict(start_L=La, direction=d, status=f'error: {e}', grid=[], path=[], events=[]))
    C.save_json(bpath, dict(case=case, J=float(J), anchors=list(anchors), branches=branches,
                            time=time.time() - t0))
    # por punto de la malla: el menor estado estable dentro del dominio
    for Lam in lams:
        cand = [g for b in branches for g in b['grid']
                if abs(g['L'] - Lam) < 1e-9 and g['in_domain'] and g['type'] in ('min', 'degmin')]
        path = C.point_path('level3', case, J, Lam, 'DAV')
        if not cand:
            C.save_json(path, dict(case=case, J=float(J), Lam=float(Lam), method='DAV',
                                   level='level3', E=np.nan, n_branch_minima=0))
            continue
        cand.sort(key=lambda g: g['E'])
        uniq = []
        for g in cand:
            if not any(abs(g['E'] - u['E']) < 1e-9 and np.max(np.abs(np.array(g['x']) - np.array(u['x']))) < 1e-4 for u in uniq):
                uniq.append(g)
        rec = C.full_record(case, J, Lam, uniq[0]['x'], method='DAV', level='level3')
        rec.update(n_branch_minima=len(uniq), branch_minima_E=[u['E'] for u in uniq])
        C.save_json(path, rec)
    return (case, J, 'DAV', len(branches))


def main():
    ap = C.base_parser('Nivel 3: puntos estacionarios, flujo gradiente y continuación de Davidenko')
    ap.add_argument('--parts', default='roots,gflow,dav')
    ap.add_argument('--nrand3', type=int, default=40, help='arranques aleatorios extra (3 parámetros)')
    ap.add_argument('--nrand6', type=int, default=80, help='arranques aleatorios extra (6 parámetros)')
    ap.add_argument('--gf-seeds', type=int, default=4, help='semillas para el flujo gradiente')
    ap.add_argument('--gf-budget', type=float, default=40.0, help='segundos máximos por trayectoria de flujo gradiente')
    ap.add_argument('--anchors', default='0.5,1.5,2.5,3.5,4.5,5.5', help='Λ de anclaje para Davidenko')
    args = ap.parse_args()
    cases, Js, lams, chunk = C.parse_common(args)
    parts = [p.strip() for p in args.parts.split(',')]
    nr = lambda c: args.nrand3 if C.npar(c) == 3 else args.nrand6
    tasks = []
    if 'roots' in parts:
        tasks += [(task_roots, c, J, L, nr(c), args.overwrite) for c in cases for J in Js for L in lams]
    if 'gflow' in parts:
        tasks += [(task_gflow, c, J, L, args.gf_seeds, args.gf_budget, args.overwrite) for c in cases for J in Js for L in lams]
    tasks.sort(key=lambda t: (-C.npar(t[1]), -t[3]))
    tasks = C.take_chunk(tasks, chunk)
    C.run_parallel(_dispatch, tasks, args.cores, 'nivel 3 / ROOTS + GFLOW')
    if 'dav' in parts and chunk[0] == 1:
        anchors = [float(a) for a in args.anchors.split(',')]
        dt = [(task_dav, c, J, lams, anchors, nr(c), args.overwrite) for c in cases for J in Js]
        C.run_parallel(_dispatch, dt, args.cores, 'nivel 3 / Davidenko')


def _dispatch(fn, *a):
    return fn(*a)


if __name__ == '__main__':
    main()
