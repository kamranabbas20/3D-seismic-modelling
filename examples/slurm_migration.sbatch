#!/bin/bash
#SBATCH --job-name=sim3d-rtm
#SBATCH --output=logs/rtm-%j.out
#SBATCH --error=logs/rtm-%j.err
#
# ONE task, MANY cores. The parallelism is Numba's `prange` over the outer
# grid axis: shared memory, one process. `--ntasks=32` would launch 32
# independent copies of the whole experiment, each racing the others for the
# same checkpoint files - it is the most expensive way to get a wrong answer.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#
# Deliberately NO --gres=gpu. sim3d has no GPU backend, so a card would sit
# idle for the whole job and, on most clusters, queue longer to do it.
#
# --requeue covers preemption and node failure. It does NOT cover the wall
# clock: a job that hits --time is cancelled, not requeued. --signal asks
# Slurm to send USR1 to the batch shell ten minutes before that happens, and
# the trap below requeues the job itself. Combined with the per-shot
# checkpoints, a job too big for one time slot finishes across several.
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -euo pipefail

# Inputs, in order of precedence: arguments after the script name, then
# environment variables, then these defaults.
#
#   sbatch examples/slurm_migration.sbatch
#   sbatch examples/slurm_migration.sbatch my_survey.yaml
#   sbatch examples/slurm_migration.sbatch my_survey.yaml runs/mine
#   CONFIG=my_survey.yaml sbatch --export=ALL,CONFIG examples/slurm_migration.sbatch
#
# Arguments are the simplest. Prefer them unless your site sets
# --export=NONE, in which case environment variables need naming explicitly
# on the sbatch line as above, or they will not reach the job.
CONFIG="${1:-${CONFIG:-examples/configs/five_layer_600m_dense.yaml}}"
# Must be a shared filesystem, not node-local scratch: the checkpoints are the
# whole point, and a requeued job can land on a different node.
OUT="${2:-${OUT:-$SLURM_SUBMIT_DIR/runs/dense}}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUT"

# --- environment ------------------------------------------------------------
# Replace with whatever your cluster uses.
# module load python/3.11
source .venv/bin/activate

# run_migration.py reads SLURM_CPUS_PER_TASK for its thread count and pins the
# BLAS pools to one thread itself, so nothing needs setting here. Passing
# --threads would override the scheduler, which is usually the wrong way round.

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK} cpus, "\
     "restart #${SLURM_RESTART_COUNT:-0}, started $(date -Is)"
echo "config: $CONFIG"
echo "output: $OUT"

# --- run --------------------------------------------------------------------
on_time_limit() {
    echo "wall clock nearly spent - requeueing $SLURM_JOB_ID; it resumes from"\
         "the last completed shot in $OUT"
    scontrol requeue "$SLURM_JOB_ID"
}
trap on_time_limit USR1

# Backgrounded and waited on deliberately: a trap does not fire while the
# shell is blocked in a foreground child, so `python ...` on its own would
# ignore USR1 until the process it was waiting for had already been killed.
python examples/run_migration.py "$CONFIG" --out "$OUT" &
PYTHON_PID=$!

# `wait` returns as soon as a trapped signal is handled, before the child has
# actually exited, and reports 128+signal when it does. Waiting once would
# therefore look like a crash the moment USR1 arrived - and under `set -e`
# would end the job before Slurm had requeued it. Wait again until the child
# really is gone.
set +e
while true; do
    wait "$PYTHON_PID"
    STATUS=$?
    [ "$STATUS" -gt 128 ] && continue
    break
done
set -e

if [ "$STATUS" -ne 0 ]; then
    echo "run_migration.py exited $STATUS at $(date -Is)"
    exit "$STATUS"
fi
echo "finished cleanly at $(date -Is)"
