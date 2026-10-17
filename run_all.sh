#!/usr/bin/env bash
# Implementation candidate: full-checkout/real-data integration is not yet verified.
# Uses only run_jev_study.py. Missing G0, CUDA, dependencies or inputs stop the run.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
MODE=run
case "${1:-}" in
  --help|-h)
    cat <<'HELP'
Usage: bash run_all.sh [--check|--resume|--status]
Run from the complete PT01-jev-phm checkout after applying integration.patch.
Default interpreter: .venv/bin/python (JEV_PYTHON may explicitly name another).
Default output: src/outputs/jev-phm/run-real-20261017-001 (JEV_RUN may name a new run).
The completed G0 file must be bound in src/S02_Configs/real/common.yaml first.
--check  validates inputs and the qualified record map without model training.
--resume reuses finished fits/evaluations after configuration checks; it does NOT
         resume an interrupted optimizer or silently restart a failed fit.
--status shows artifact presence only, not scientific completion or qualification.
No downloads, synthetic input, automatic retries, CPU or checkpoint fallback.
All run directories share the fixed approved-study budget; a changed run root
cannot reset it. Incomplete jobs require manual technical recovery, not a retry.
HELP
    exit 0;;
  --check) MODE=check;;
  --resume) MODE=resume;;
  --status) MODE=status;;
  '') ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2;exit 2;;
esac
PYTHON=${JEV_PYTHON:-$ROOT/.venv/bin/python}
RUN=${JEV_RUN:-$ROOT/src/outputs/jev-phm/run-real-20261017-001}
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
ENTRY="$ROOT/src/S03_Scripts/run_jev_study.py"
COMMON_MODEL=src/S02_Configs/real/semantic.yaml
CONFIGS=(
  src/S02_Configs/real/rf.yaml
  src/S02_Configs/real/signal.yaml
  src/S02_Configs/real/indexed.yaml
  src/S02_Configs/real/semantic.yaml
  src/S02_Configs/real/permuted.yaml
  src/S02_Configs/real/token.yaml
)
if [[ "$MODE" == status ]]; then
  printf 'Artifact presence only; no outcome has been revalidated.\n'
  for path in study.json source_scale.json smoke/result.json summary.csv; do
    [[ -f "$RUN/$path" ]] && state=PRESENT || state=MISSING
    printf '%-8s %s\n' "$state" "$RUN/$path"
  done
  for name in rf signal indexed semantic permuted token; do
    [[ -f "$RUN/fits/$name/selection.json" ]] && state=PRESENT || state=NOT_FINISHED
    printf '%-12s %s\n' "$state" "$name"
  done
  exit 0
fi
[[ -x "$PYTHON" ]] || { printf 'BLOCKED: interpreter does not exist: %s\n' "$PYTHON" >&2;exit 2; }
[[ -f "$ENTRY" && -f AGENTS.md ]] || { echo 'BLOCKED: apply this patch inside the complete repository first.' >&2;exit 2; }
command -v timeout >/dev/null || { echo 'BLOCKED: GNU timeout is required.' >&2;exit 2; }
command -v flock >/dev/null || { echo 'BLOCKED: flock is required for single-run ownership.' >&2;exit 2; }
git rev-parse HEAD
git status --short
[[ -z "$(git status --porcelain -- src/S01_Package src/S02_Configs src/S03_Scripts run_all.sh)" ]] || {
  echo 'BLOCKED: commit the reviewed scientific implementation/configuration before running.' >&2;exit 2;
}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Even preflight/upstream imports have no visible CUDA devices.
export CUDA_VISIBLE_DEVICES=''
"$PYTHON" "$ENTRY" semantic check --config "$COMMON_MODEL" --run-root "$RUN"
[[ "$MODE" == check ]] && exit 0
command -v nvidia-smi >/dev/null || { echo 'BLOCKED: nvidia-smi is unavailable.' >&2;exit 2; }
GPU_UUID=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["runtime"]["gpu_uuid"])' src/S02_Configs/real/common.yaml)
GPU_INDEX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F ',' -v wanted="$GPU_UUID" '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); if ($2==wanted) print $1}')
[[ "$GPU_INDEX" =~ ^[0-9]+$ && "$GPU_INDEX" != 2 ]] || {
  echo 'BLOCKED: configured UUID is absent, ambiguous or identifies physical GPU 2.' >&2;exit 2;
}
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
if [[ "$MODE" == run && -e "$RUN" ]]; then
  echo 'BLOCKED: output exists; use --resume for complete stages, or explicitly select a new JEV_RUN.' >&2
  exit 2
fi
mkdir -p "$RUN/execution"
BUDGET="$ROOT/src/outputs/jev-phm/p1-approved-20261017/execution"
mkdir -p "$BUDGET"
exec 9>"$BUDGET/process.lock"
flock -n 9 || { echo 'BLOCKED: the approved study already has an active process.' >&2;exit 2; }
if [[ -f "$BUDGET/run-root.txt" ]]; then
  [[ "$(cat "$BUDGET/run-root.txt")" == "$RUN" ]] || {
    echo 'BLOCKED: this approved budget is already bound to another run root. Retain its consumed time and failed jobs; changing JEV_RUN does not authorize another 46 fits.' >&2;exit 2;
  }
else
  printf '%s\n' "$RUN" > "$BUDGET/run-root.txt"
fi
USAGE="$BUDGET/usage.tsv"
if [[ -f "$BUDGET/pending.tsv" ]]; then
  echo 'BLOCKED: an interrupted command has an unaccounted resource interval; retain it and resolve actual elapsed use before resuming.' >&2
  exit 2
fi
run_stage() {
  local lane=$1 label=$2 cap=$3;shift 3
  local ceiling=43200 used=0 left started elapsed rc log
  [[ "$lane" == gpu ]] && ceiling=86400
  [[ -f "$USAGE" ]] && used=$(awk -F '\t' -v lane="$lane" '$1==lane {s+=$3} END {print s+0}' "$USAGE")
  left=$((ceiling-used))
  if [[ "$lane" == gpu ]]; then
    local phase_limit phase_used pattern
    case "$label" in
      fit_*) phase_limit=72000;pattern="^fit_";;
      smoke) phase_limit=3600;pattern="^smoke$";;
      verify_source|infer_*|timing_*) phase_limit=3600;pattern="^(verify_source$|infer_|timing_)";;
      *) echo "Unbudgeted GPU operation: $label" >&2;exit 2;;
    esac
    phase_used=0
    [[ -f "$USAGE" ]] && phase_used=$(awk -F '\t' -v pat="$pattern" '$1=="gpu" && $2~pat {s+=$3} END {print s+0}' "$USAGE")
    (( phase_limit-phase_used < left )) && left=$((phase_limit-phase_used))
  fi
  (( left > 0 )) || { echo "BLOCKED: $lane budget exhausted." >&2;exit 2; }
  (( cap > left )) && cap=$left
  log=$(mktemp "$RUN/execution/${label}.XXXXXX.log")
  started=$(date +%s)
  printf '%s\t%s\t%s\n' "$lane" "$label" "$started" > "$BUDGET/pending.tsv"
  printf 'RUN %q ' "$@";printf '\n'
  set +e
  timeout --signal=TERM --kill-after=15 "$cap" "$@" >"$log" 2>&1
  rc=$?
  set -e
  elapsed=$(($(date +%s)-started))
  printf '%s\t%s\t%s\t%s\t%s\n' "$lane" "$label" "$elapsed" "$rc" "$log" >> "$USAGE"
  rm -- "$BUDGET/pending.tsv"
  cat -- "$log"
  if (( rc != 0 )); then
    printf 'FAILED %s, exit=%s. Outputs and logs retained; no automatic retry.\n' "$label" "$rc" >&2
    exit "$rc"
  fi
}
if [[ ! -f "$RUN/source_scale.json" ]]; then
  run_stage cpu prepare 36000 "$PYTHON" "$ENTRY" semantic prepare --config "$COMMON_MODEL" --run-root "$RUN"
fi
if [[ ! -f "$RUN/smoke/result.json" ]]; then
  run_stage gpu smoke 3600 "$PYTHON" "$ENTRY" semantic smoke --config "$COMMON_MODEL" --run-root "$RUN"
fi
for config in "${CONFIGS[@]}"; do
  name=$(basename "$config" .yaml)
  lane=gpu;cap=14400
  [[ "$name" == rf ]] && { lane=cpu;cap=7200; }
  run_stage "$lane" "fit_${name}" "$cap" "$PYTHON" "$ENTRY" semantic fit --config "$config" --run-root "$RUN"
done
# Reload every selected checkpoint and freeze source calibration before MFPT access.
run_stage gpu verify_source 3600 "$PYTHON" "$ENTRY" semantic verify-source --config "$COMMON_MODEL" --run-root "$RUN"
for config in "${CONFIGS[@]}"; do
  name=$(basename "$config" .yaml);lane=gpu
  [[ "$name" == rf ]] && lane=cpu
  run_stage "$lane" "infer_${name}" 3600 "$PYTHON" "$ENTRY" semantic infer --config "$config" --run-root "$RUN"
done
for config in "${CONFIGS[@]}"; do
  name=$(basename "$config" .yaml)
  [[ "$name" == permuted ]] && continue  # Same inference graph as semantic; no extra timing claim.
  if [[ ! -f "$RUN/timing/$name/summary.json" ]]; then
    lane=gpu;[[ "$name" == rf ]] && lane=cpu
    run_stage "$lane" "timing_${name}" 600 "$PYTHON" "$ENTRY" semantic timing --config "$config" --run-root "$RUN"
  fi
done
if [[ ! -f "$RUN/summary.csv" || ! -f "$RUN/paired_unit_effects.csv" || ! -f "$RUN/figures/external_unit_effect.svg" || ! -f "$RUN/figures/error_processing_cost.svg" ]]; then
  run_stage cpu summarize 1800 "$PYTHON" "$ENTRY" semantic summarize --config "$COMMON_MODEL" --run-root "$RUN"
fi
printf 'Pipeline returned without error. Inspect qualification, saved predictions and limits; this does not establish method superiority.\n'
