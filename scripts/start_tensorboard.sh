#!/bin/bash
#
# Start TensorBoard for the sc_autoencoder_project.
# All TB data is under logs/tensorboard/ with subdirectories per run.
#
# Usage (on the HPC login node):
#   bash scripts/start_tensorboard.sh [port]
#
# If using VS Code Remote SSH:
#   Port is auto-forwarded. Just open http://localhost:<port> in your browser.
#
# If using plain SSH:
#   From your LOCAL machine: ssh -L <port>:localhost:<port> user@login-node
#   Then open http://localhost:<port>
#

PORT=${1:-6006}
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
LOGDIR="$PROJECT_ROOT/logs/tensorboard"

echo "=== TensorBoard Launcher ==="
echo "  Project: $PROJECT_ROOT"
echo "  Log dir: $LOGDIR"
echo "  Port:    $PORT"
echo ""

# Kill any existing TensorBoard on this port
pkill -f "tensorboard.*--port $PORT" 2>/dev/null && echo "  Killed previous TB instance" && sleep 1

# Check if logdir exists and has data
if [ ! -d "$LOGDIR" ]; then
    echo "  ❌ Directory $LOGDIR does not exist."
    echo "  Run an experiment first: sbatch.tinygpu jobs/tinygpu_single.sh"
    exit 1
fi

EVENT_COUNT=$(find "$LOGDIR" -name "events.out.tfevents*" 2>/dev/null | wc -l)
echo "  Event files found: $EVENT_COUNT"

if [ "$EVENT_COUNT" -eq 0 ]; then
    echo ""
    echo "  ⚠️  No TensorBoard event files in $LOGDIR"
    echo "  Subdirectories present:"
    ls -d "$LOGDIR"/*/ 2>/dev/null | while read d; do echo "    $(basename "$d")/"; done
    echo ""
    echo "  Run one of:"
    echo "    sbatch.tinygpu jobs/smoke_test.sh     # quick test (~2 min)"
    echo "    sbatch.tinygpu jobs/tinygpu_single.sh  # full experiment"
    exit 1
fi

echo ""
echo "  Available runs (each appears in TB's left sidebar dropdown):"
for dir in "$LOGDIR"/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "events.out.tfevents*" 2>/dev/null | wc -l)
        name=$(basename "$dir")
        echo "    📁 $name/ ($count event files)"
    fi
done

# Find tensorboard binary
TB_BIN=""
if command -v tensorboard &>/dev/null; then
    TB_BIN="tensorboard"
elif [ -f "$HOME/.local/bin/tensorboard" ]; then
    TB_BIN="$HOME/.local/bin/tensorboard"
fi

if [ -z "$TB_BIN" ]; then
    echo ""
    echo "  ❌ tensorboard not found. Trying python -m tensorboard.main..."
    TB_BIN="python3 -m tensorboard.main"
fi

echo ""
echo "  Starting: $TB_BIN --logdir $LOGDIR --port $PORT"

# Start TensorBoard
# --bind_all: listen on all interfaces (needed for port forwarding)
# --reload_interval: how often to check for new data (seconds)
$TB_BIN --logdir "$LOGDIR" --port "$PORT" --bind_all --reload_interval 15 2>&1 &
TB_PID=$!

# Wait and verify
sleep 4
if kill -0 $TB_PID 2>/dev/null; then
    echo ""
    echo "  ✅ TensorBoard running (PID: $TB_PID)"
    echo ""
    echo "  ┌──────────────────────────────────────────┐"
    echo "  │  Open in browser: http://localhost:$PORT  │"
    echo "  └──────────────────────────────────────────┘"
    echo ""
    echo "  • Click 'Scalars' tab to see loss curves"
    echo "  • Use the 'Runs' sidebar (left) to select/compare runs"
    echo "  • Data auto-refreshes every 15 seconds"
    echo ""
    echo "  To stop: kill $TB_PID"
else
    echo ""
    echo "  ❌ TensorBoard failed to start!"
    echo "  Check if port $PORT is already in use: lsof -i :$PORT"
    echo "  Try manually: $TB_BIN --logdir $LOGDIR --port $PORT --bind_all"
    exit 1
fi
