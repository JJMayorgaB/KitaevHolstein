#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Batería completa de pruebas del optimizador variacional (P+AP)S
#    0. autoprueba (funcional AD = producción)
#    1. nivel 1: optimizadores locales
#    2. nivel 2: optimizadores globales
#    3. nivel 3: puntos estacionarios, flujo gradiente, continuación de Davidenko
#    4. nivel 4: comparación y reporte  -> results/level4/report.txt
#
#  Uso:
#    bash run_all.sh                    # todos los núcleos, mallas completas
#    bash run_all.sh 12                 # 12 núcleos
#    bash run_all.sh 12 --quick         # prueba rápida (5 valores de Λ, J = 1)
#    bash run_all.sh 12 --cases odd3    # opciones extra: se pasan a los niveles 1–3
#  En el cluster, para que siga corriendo al cerrar la sesión:
#    nohup bash run_all.sh 32 > run_all.log 2>&1 &
#  Si se interrumpe, vuelve a lanzarlo igual: cada punto ya calculado se salta.
# ══════════════════════════════════════════════════════════════════════════════
set -eo pipefail
cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
else
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*) PY=python ;;
        *) if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi ;;
    esac
fi

CORES="${1:-$(nproc 2>/dev/null || echo 4)}"
[ $# -gt 0 ] && shift
EXTRA=("$@")

mkdir -p results/logs
run () {
    local name="$1"; shift
    echo ""
    echo "════════════ $name   $(date '+%Y-%m-%d %H:%M:%S')"
    "$PY" "$@" 2>&1 | tee "results/logs/$name.log"
}

T0=$(date +%s)
run selftest common.py --selftest
run level1   level1_local.py      --cores "$CORES" "${EXTRA[@]}"
run level2   level2_global.py     --cores "$CORES" "${EXTRA[@]}"
run level3   level3_stationary.py --cores "$CORES" "${EXTRA[@]}"
run level4   level4_analysis.py
echo ""
echo "Todo listo en $(( ($(date +%s) - T0)/60 )) min.   Reporte: results/level4/report.txt"
