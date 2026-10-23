# -*- coding: utf-8 -*-
"""
NIVEL 4 — criterio de comparación
==================================
Lee todo lo que produjeron los niveles 1–3 y aplica el mismo criterio a todos los métodos:

  1. Energía: E_ref(caso, J, Λ) = menor energía encontrada por CUALQUIER método dentro del
     dominio. ΔE = E − E_ref. Coincide si ΔE < TOL_E; 'incompleto' si TOL_E ≤ ΔE < 1e-6;
     'otro mínimo' si ΔE ≥ 1e-6.
  2. Cantidades físicas (fidelidad, ⟨b†⟩↑, o, p±, β̃, pesos): dispersión entre los métodos
     que coinciden en energía.
  3. Parámetros: solo se exigen donde son identificables. La incertidumbre que permite la
     tolerancia en energía es δx_i = sqrt(2 TOL_E (H⁻¹)_ii) (hessiano exacto en el óptimo);
     si δx_i > 1e-3 el parámetro se declara 'no identificable' (dirección plana).
  4. Optimalidad en el óptimo de referencia: gradiente proyectado (KKT), signo de los
     multiplicadores en las cotas, hessiano definido positivo en las variables libres.
  5. Paisaje (nivel 3): número de niveles de energía de los mínimos locales dentro del
     dominio (mínimos a menos de --tol-level cuentan como uno: misma cuenca plana o imágenes
     especulares), mínimos que compiten, eventos de Davidenko (pliegues / bifurcaciones).
  6. Simetría: odd3 (paridad impuesta) frente a odd6 (sin imponerla).
  7. Producción: 'prod' (frío) y 'prod_warm' (continuación, lo que generan las figuras)
     frente a E_ref.

Uso:
    python level4_analysis.py            [--tolE 1e-9]
Salida: results/level4/report.txt, records.csv, fig_*.png
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

import common as C

LEVEL_OF = {'L-BFGS-B': 1, 'SLSQP': 1, 'trust-constr': 1, 'Nelder-Mead': 1, 'Powell': 1,
            'COBYLA': 1, 'prod': 1, 'prod_warm': 1, 'DE': 2, 'DA': 2, 'BH': 2, 'GRID': 2,
            'ROOTS': 3, 'GFLOW': 3, 'DAV': 3}
METHOD_ORDER = list(LEVEL_OF)
PHYS = ['fid', 'bdag_up', 'o_var', 'p_plus', 'beta_var', 'c_up']
RANGES = [('I   (Λ < 1.5)', 0.0, 1.5), ('II  (1.5 ≤ Λ < 3.5)', 1.5, 3.5), ('III (Λ ≥ 3.5)', 3.5, 99.0)]
OUT = os.path.join(C.RESULTS, 'level4')


def load_all():
    recs = []
    for lvl in ('level1', 'level2', 'level3'):
        for p in glob.glob(os.path.join(C.RESULTS, lvl, '*', 'J*', 'L*__*.json')):
            try:
                r = C.load_json(p)
            except Exception:
                continue
            if 'method' not in r or not np.isfinite(r.get('E', np.nan)):
                continue
            recs.append(r)
    return recs


def mirror(x):
    """Paridad de Rabi en δ_α = 0: (η↑, η↓, r↑, r↓, κ↑, κ↓) → (−η↓, −η↑, r↓, r↑, κ↓, κ↑)."""
    x = np.asarray(x, float)
    return np.array([-x[1], -x[0], x[3], x[2], x[5], x[4]])


def canonical(r):
    """En odd6 cada mínimo roto tiene una imagen especular con la MISMA energía. Las
    cantidades que no son invariantes (⟨b†⟩↑, c↑) y los parámetros se comparan en el
    representante con η↑ + η↓ ≥ 0."""
    if r['case'] == 'odd6' and r['x'][0] + r['x'][1] < -1e-6:
        xm = mirror(r['x'])
        r.update(C.physics('odd6', r['J'], r['Lam'], xm))
        r['x'] = xm.tolist()
        r['mirrored'] = True
    return r


def energy_levels(Es, tol):
    """Agrupa energías a menos de tol (misma cuenca plana o imágenes especulares)."""
    lv = []
    for e in sorted(Es):
        if not lv or e - lv[-1] > tol:
            lv.append(e)
    return lv


def in_domain(r):
    lo, hi = C.lohi(r['case'], r['Lam'])
    x = np.asarray(r['x'])
    return bool(np.all(x >= lo - 1e-8) and np.all(x <= hi + 1e-8))


def fmt(v, f='{:.1e}'):
    return '—' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def param_uncertainty(case, J, Lam, x, tolE):
    """δx_i = sqrt(2 tolE (H⁻¹)_ii) en las variables libres (las de cota: 0)."""
    ad = C.get_ad(case)
    lo, hi = C.lohi(case, Lam)
    x = np.asarray(x, float)
    free = ~((x <= lo + 1e-9) | (x >= hi - 1e-9))
    dx = np.zeros(len(x))
    if free.any():
        H = ad.hess(x, Lam, J)[np.ix_(free, free)]
        w, V = np.linalg.eigh(H)
        w = np.maximum(w, 1e-300)
        Hinv_diag = np.sum(V**2/w[None, :], axis=1)
        dx[free] = np.sqrt(2.0*tolE*Hinv_diag)
    return dx


def main():
    ap = argparse.ArgumentParser(description='Nivel 4: comparación de los niveles 1–3')
    ap.add_argument('--tolE', type=float, default=1e-9)
    ap.add_argument('--tolphys', type=float, default=1e-4,
                    help='dispersión admisible de cantidades físicas; con tolE = 1e-9 y λ_min ~ 1e-6 los parámetros\n'
                         'no identificables pueden moverse ~1e-2 sin cambiar E, y ⟨b†⟩↑ depende de ellos')
    ap.add_argument('--tol-level', type=float, default=1e-7,
                    help='dos mínimos a menos de esto son el mismo nivel (misma cuenca plana o imágenes especulares)')
    ap.add_argument('--no-figs', action='store_true')
    args = ap.parse_args()
    tolE = args.tolE
    os.makedirs(OUT, exist_ok=True)

    recs = [canonical(r) for r in load_all() if in_domain(r)]
    if not recs:
        raise SystemExit('No hay resultados en results/level1..3. Corre primero los niveles 1–3.')
    by_pt = defaultdict(dict)
    for r in recs:
        by_pt[(r['case'], round(r['J'], 6), round(r['Lam'], 6))][r['method']] = r

    lines = []
    P = lines.append
    P('=' * 100)
    P(' NIVEL 4 — REPORTE DE LA BATERÍA DE PRUEBAS DEL OPTIMIZADOR VARIACIONAL (P+AP)S')
    P('=' * 100)
    P(f' producción: {os.path.abspath(C.PROD_PATH)}    AD: {C.BACKEND}    tolE = {tolE:.0e}   tolphys = {args.tolphys:.0e}')
    P(' ΔE = E − E_ref,  E_ref = menor energía de todos los métodos en ese (caso, J, Λ)')
    P('')

    # ── referencia por punto
    ref = {}
    rows = []
    for key, d in sorted(by_pt.items()):
        best_m = min(d, key=lambda m: d[m]['E'])
        ref[key] = d[best_m]
        for m, r in d.items():
            dE = r['E'] - d[best_m]['E']
            r['_dE'] = dE
            rows.append([key[0], key[1], key[2], m, LEVEL_OF.get(m, 0), f"{r['E']:.15f}", f'{dE:.3e}',
                         r.get('fid'), r.get('bdag_up'), r.get('o_var'), r.get('p_plus'), r.get('beta_var'),
                         r.get('projgrad'), r.get('lam_min'), r.get('n_neg'), r.get('time'), r.get('nfev')])
    with open(os.path.join(OUT, 'records.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['case', 'J', 'Lam', 'method', 'level', 'E', 'dE', 'fid', 'bdag_up', 'o_var',
                    'p_plus', 'beta_var', 'projgrad', 'lam_min', 'n_neg', 'time_s', 'nfev'])
        w.writerows(rows)

    cases = sorted({k[0] for k in by_pt}, key=lambda c: list(C.CASES).index(c))
    Js = sorted({k[1] for k in by_pt})

    # ── 0. inventario
    P('0. INVENTARIO (puntos con resultado por método)')
    cnt = defaultdict(int)
    for (c, J, L), d in by_pt.items():
        for m in d:
            cnt[(c, m)] += 1
    head = '   ' + f'{"caso":7s}' + ''.join(f'{m[:11]:>12s}' for m in METHOD_ORDER)
    P(head)
    for c in cases:
        P('   ' + f'{c:7s}' + ''.join(f'{cnt.get((c, m), 0):12d}' for m in METHOD_ORDER))
    P('')

    verdict = []

    # ── 1. energía por método
    P('1. ENERGÍA: máx ΔE por método y rango de Λ;  [n_fallos = puntos con ΔE ≥ tolE]')
    for c in cases:
        for J in Js:
            pts = {k: d for k, d in by_pt.items() if k[0] == c and k[1] == J}
            if not pts:
                continue
            P(f'   ── {c}   J/ω0 = {J}   ({len(pts)} valores de Λ)')
            P('      ' + f'{"método":13s}' + ''.join(f'{rn[:3]:>14s}' for rn, _, _ in RANGES)
              + f'{"n_fallos":>10s}{"n_otro_min":>12s}{"t_med[s]":>10s}')
            for m in METHOD_ORDER:
                vals = [(k[2], d[m]['_dE'], d[m].get('time', np.nan)) for k, d in pts.items() if m in d]
                if not vals:
                    continue
                cols = []
                for _, a, b in RANGES:
                    v = [dE for L, dE, _ in vals if a <= L < b]
                    cols.append(fmt(max(v)) if v else '—')
                nf = sum(dE >= tolE for _, dE, _ in vals)
                no = sum(dE >= 1e-6 for _, dE, _ in vals)
                tmed = np.nanmedian([t for _, _, t in vals]) if vals else np.nan
                P('      ' + f'{m:13s}' + ''.join(f'{s:>14s}' for s in cols) + f'{nf:>10d}{no:>12d}{fmt(tmed, "{:.2f}"):>10s}')
                if nf:
                    bad = sorted([(L, dE) for L, dE, _ in vals if dE >= tolE], key=lambda t: -t[1])[:6]
                    P('         peores: ' + ', '.join(f'Λ={L:.2f} (ΔE={dE:.1e})' for L, dE in bad))
            P('')

    # ── 2. cantidades físicas
    P('2. CANTIDADES FÍSICAS: máx dispersión (máx − mín) entre los métodos con ΔE < tolE')
    P('      ' + f'{"caso":7s}{"J":>6s}' + ''.join(f'{q:>12s}' for q in PHYS) + f'{"n_métodos":>11s}')
    worst_phys = 0.0
    for c in cases:
        for J in Js:
            spreads = {q: 0.0 for q in PHYS}
            nm = []
            for k, d in by_pt.items():
                if k[0] != c or k[1] != J:
                    continue
                ok = [r for r in d.values() if r['_dE'] < tolE]
                nm.append(len(ok))
                for q in PHYS:
                    v = [abs(r[q]) if q == 'c_up' else r[q] for r in ok if q in r and r[q] is not None]
                    if len(v) > 1:
                        spreads[q] = max(spreads[q], float(np.max(v) - np.min(v)))
            if not nm:
                continue
            worst_phys = max(worst_phys, max(spreads.values()))
            P('      ' + f'{c:7s}{J:6.2f}' + ''.join(f'{fmt(spreads[q]):>12s}' for q in PHYS) + f'{int(np.median(nm)):>11d}')
    P('')

    # ── 3. parámetros
    P('3. PARÁMETROS: dispersión entre métodos con ΔE < tolE, separando identificables (δx < 1e-3)')
    P('   δx_i = sqrt(2·tolE·(H⁻¹)_ii) en el óptimo de referencia: lo que la energía NO puede fijar')
    nonid = defaultdict(list)
    worst_id = 0.0
    for c in cases:
        names = C.PNAMES[C.npar(c)]
        for J in Js:
            sp_id = 0.0
            sp_all = 0.0
            for k, d in sorted(by_pt.items()):
                if k[0] != c or k[1] != J:
                    continue
                xr = np.asarray(ref[k]['x'])
                dx = param_uncertainty(c, J, k[2], xr, tolE)
                ok = [np.asarray(r['x']) for r in d.values() if r['_dE'] < tolE]
                if len(ok) < 2:
                    continue
                X = np.array(ok)
                spread = X.max(0) - X.min(0)
                ident = dx < 1e-3
                if ident.any():
                    sp_id = max(sp_id, float(np.max(spread[ident])))
                sp_all = max(sp_all, float(np.max(spread)))
                for i in np.where(~ident)[0]:
                    nonid[(c, J, names[i])].append(k[2])
            worst_id = max(worst_id, sp_id)
            P(f'      {c:7s} J={J:4.2f}   máx dispersión identificables = {fmt(sp_id)}   (todas = {fmt(sp_all)})')
    if nonid:
        P('   parámetros NO identificables (dirección plana; no deben compararse ni graficarse sin marca):')
        for (c, J, nm_), Ls in sorted(nonid.items()):
            Ls = sorted(Ls)
            P(f'      {c:7s} J={J:4.2f}  {nm_:8s}  {len(Ls):3d} puntos   Λ ∈ [{Ls[0]:.2f}, {Ls[-1]:.2f}]')
    P('')

    # ── 4. optimalidad del óptimo de referencia
    P('4. OPTIMALIDAD EN EL ÓPTIMO DE REFERENCIA (hessiano y gradiente exactos, AD)')
    worst_pg, nneg_tot, kkt_bad = 0.0, 0, 0
    for c in cases:
        for J in Js:
            ks = [k for k in by_pt if k[0] == c and k[1] == J]
            if not ks:
                continue
            pg = [C.optimality(c, J, k[2], ref[k]['x']) for k in ks]
            mpg = max(q['projgrad'] for q in pg)
            lmn = [q['lam_min'] for q in pg]
            nn = sum(q['n_neg'] > 0 for q in pg)
            kb = sum(not q['kkt_sign_ok'] for q in pg)
            nb = sum(any(q['at_lo']) or any(q['at_hi']) for q in pg)
            worst_pg = max(worst_pg, mpg); nneg_tot += nn; kkt_bad += kb
            P(f'      {c:7s} J={J:4.2f}  máx|∇E proy| = {mpg:.1e}   λ_min ∈ [{min(lmn):.1e}, {max(lmn):.1e}]'
              f'   puntos con λ<0: {nn}   KKT signo mal: {kb}   puntos con alguna cota activa: {nb}')
    P('')

    # ── 5. paisaje
    P('5. PAISAJE DE ENERGÍA (nivel 3)')
    competing = defaultdict(list)
    for c in cases:
        for J in Js:
            ks = sorted(k for k in by_pt if k[0] == c and k[1] == J and 'ROOTS' in by_pt[k])
            if not ks:
                continue
            nlev = []
            for k in ks:
                lv = energy_levels(by_pt[k]['ROOTS'].get('minima_E', []), args.tol_level)
                nlev.append(len(lv))
                if len(lv) >= 2 and lv[1] - lv[0] < 1e-2:
                    competing[(c, J)].append((k[2], lv[1] - lv[0]))
            P(f'      {c:7s} J={J:4.2f}  niveles de energía de mínimos locales en el dominio por Λ '
              f'(a menos de tol_level = un nivel): mín {min(nlev)}, máx {max(nlev)}')
            if competing[(c, J)]:
                cl = competing[(c, J)]
                P(f'         mínimos que compiten (segundo nivel a < 1e-2 del primero): {len(cl)} valores de Λ: '
                  + ', '.join(f'{L:.2f} ({g:.1e})' for L, g in cl[:15]) + (' …' if len(cl) > 15 else ''))
            dav = os.path.join(C.RESULTS, 'level3', c, f'J{J:.3f}', 'davidenko.json')
            if os.path.exists(dav):
                D = C.load_json(dav)
                ev = [e for b in D['branches'] for e in b.get('events', [])]
                st = defaultdict(int)
                for b in D['branches']:
                    st[b['status'].split(':')[0]] += 1
                Ls = sorted({round(e['L'], 2) for e in ev})
                P(f'         Davidenko: {len(D["branches"])} ramas ({dict(st)}); cambios de índice del hessiano en Λ ≈ '
                  + (', '.join(f'{L:.2f}' for L in Ls[:12]) if Ls else 'ninguno'))
    P('')

    # ── 6. simetría
    if 'odd3' in cases and 'odd6' in cases:
        P('6. SIMETRÍA: E_ref(odd6) − E_ref(odd3)  (< 0: la reducción simétrica NO es el mínimo global)')
        for J in Js:
            dd = []
            for k in sorted(by_pt):
                if k[0] == 'odd3' and k[1] == J and ('odd6', J, k[2]) in ref:
                    r3, r6 = ref[k], ref[('odd6', J, k[2])]
                    dd.append((k[2], r6['E'] - r3['E'], r3['fid'], r6['fid'], r6.get('asym', 0.0)))
            if not dd:
                continue
            br = [t for t in dd if t[1] < -tolE]
            P(f'      J={J:4.2f}: puntos con ruptura de simetría: {len(br)} de {len(dd)}'
              + (f';  Λ ∈ [{br[0][0]:.2f}, {br[-1][0]:.2f}],  mín ΔE = {min(t[1] for t in br):.2e},'
                 f'  fidelidad odd3 / odd6 en el peor: {min(br, key=lambda t: t[1])[2]:.4f} / {min(br, key=lambda t: t[1])[3]:.4f}'
                 if br else ''))
        P('')

    # ── 7. producción
    P('7. PRODUCCIÓN FRENTE A E_ref')
    P('   (las figuras usan odd3 y even6; en odd6, prod y prod_warm son el mismo minimizador de producción')
    P('    aplicado al funcional general sin imponer la simetría: no corresponden a ninguna figura)')
    prod_bad = 0
    for c in cases:
        for J in Js:
            for m in ('prod', 'prod_warm'):
                v = [(k[2], d[m]['_dE']) for k, d in by_pt.items() if k[0] == c and k[1] == J and m in d]
                if not v:
                    continue
                nb = sum(dE >= tolE for _, dE in v)
                prod_bad += nb if (m == 'prod_warm' and c != 'odd6') else 0
                worst = max(v, key=lambda t: t[1])
                P(f'      {c:7s} J={J:4.2f}  {m:9s}: máx ΔE = {worst[1]:.1e} (Λ={worst[0]:.2f})   puntos con ΔE ≥ tolE: {nb}/{len(v)}')
    P('')

    # ── veredicto
    P('=' * 100)
    P(' VEREDICTO AUTOMÁTICO')
    P('=' * 100)
    P(f'  • Máx dispersión de cantidades físicas entre métodos convergidos: {worst_phys:.1e} '
      + ('(≤ tolphys: OK)' if worst_phys <= args.tolphys else '(> tolphys: REVISAR)'))
    P(f'  • Máx dispersión de parámetros identificables: {worst_id:.1e}')
    P(f'  • Óptimos de referencia: máx|∇E proy| = {worst_pg:.1e}, puntos con λ_min < 0 = {nneg_tot}, KKT con signo mal = {kkt_bad}')
    P(f'  • prod_warm en odd3 y even6 (lo que generan las figuras) con ΔE ≥ tolE: {prod_bad} puntos')
    if competing:
        P('  • Hay mínimos que compiten (ver sección 5): en esos Λ el resultado depende del método/semillas.')
    P('')
    rep = '\n'.join(lines)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(rep + '\n')
    print(rep)

    if not args.no_figs:
        make_figs(by_pt, cases, Js, tolE)
    print(f'\nReporte: {os.path.join(OUT, "report.txt")}')


def make_figs(by_pt, cases, Js, tolE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20')
    col = {m: cmap(i % 20) for i, m in enumerate(METHOD_ORDER)}
    for c in cases:
        fig, axes = plt.subplots(1, len(Js), figsize=(6*len(Js), 4.5), squeeze=False, sharey=True)
        for ax, J in zip(axes[0], Js):
            for m in METHOD_ORDER:
                v = sorted((k[2], d[m]['_dE']) for k, d in by_pt.items() if k[0] == c and k[1] == J and m in d)
                if v:
                    L, dE = np.array(v).T
                    ax.semilogy(L, np.maximum(dE, 1e-17), '.-', ms=3, lw=0.8, color=col[m], label=m)
            ax.axhline(tolE, color='k', ls=':', lw=1)
            ax.set_title(f'{c},  J/ω0 = {J}')
            ax.set_xlabel('Λ')
        axes[0][0].set_ylabel('E − E_ref')
        axes[0][-1].legend(fontsize=7, ncol=2, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'fig_dE_{c}.png'), dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(Js), figsize=(6*len(Js), 4.5), squeeze=False)
        for ax, J in zip(axes[0], Js):
            for k, d in by_pt.items():
                if k[0] == c and k[1] == J and 'ROOTS' in d:
                    mE = np.array(sorted(d['ROOTS'].get('minima_E', [])))
                    if mE.size:
                        ax.semilogy([k[2]]*mE.size, np.maximum(mE - mE[0], 1e-17), 'o', ms=3,
                                    color='C0' if mE.size == 1 else 'C3')
            ax.set_title(f'{c},  J/ω0 = {J}:  mínimos locales (ROOTS)')
            ax.set_xlabel('Λ'); ax.set_ylabel('E_min,i − E_min,0')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'fig_minima_{c}.png'), dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(Js), figsize=(6*len(Js), 4.5), squeeze=False, sharey=True)
        for ax, J in zip(axes[0], Js):
            v = sorted((k[2], C.optimality(c, J, k[2], min(d.values(), key=lambda r: r['E'])['x'])['lam_min'])
                       for k, d in by_pt.items() if k[0] == c and k[1] == J)
            if v:
                L, lm = np.array(v).T
                ax.semilogy(L, np.abs(lm), 'k.-', ms=3)
            ax.set_title(f'{c},  J/ω0 = {J}:  λ_min del hessiano')
            ax.set_xlabel('Λ')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'fig_lammin_{c}.png'), dpi=140)
        plt.close(fig)

    if 'odd3' in cases and 'odd6' in cases:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for J in Js:
            v = []
            for k, d in sorted(by_pt.items()):
                if k[0] == 'odd3' and k[1] == J and ('odd6', J, k[2]) in by_pt:
                    e3 = min(r['E'] for r in d.values())
                    d6 = by_pt[('odd6', J, k[2])]
                    r6 = min(d6.values(), key=lambda r: r['E'])
                    r3 = min(d.values(), key=lambda r: r['E'])
                    v.append((k[2], e3 - r6['E'], r3['fid'], r6['fid']))
            if v:
                L, de, f3, f6 = np.array(v).T
                axes[0].semilogy(L, np.maximum(de, 1e-17), '.-', label=f'J={J}')
                axes[1].plot(L, f3, '-', label=f'odd3 J={J}')
                axes[1].plot(L, f6, '--', label=f'odd6 J={J}')
        axes[0].set_title('E_ref(odd3) − E_ref(odd6)'); axes[0].set_xlabel('Λ'); axes[0].legend()
        axes[1].set_title('fidelidad del óptimo'); axes[1].set_xlabel('Λ'); axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig_odd3_vs_odd6.png'), dpi=140)
        plt.close(fig)


if __name__ == '__main__':
    main()
