#!/bin/bash

# PEAR Compression Batch Runner
# Usage: ./run_compression.sh [start|monitor|results|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$1" in
    "start")
        echo "🚀 Starting compression jobs..."
        python batch_compress.py start
        ;;
    "monitor")
        echo "🔍 Starting progress monitor..."
        python batch_compress.py monitor
        ;;
    "results")
        echo "📋 Showing results..."
        python batch_compress.py results
        ;;
    "all")
        echo "🎯 Starting all jobs and monitoring..."
        python batch_compress.py start
        sleep 5
        python batch_compress.py monitor
        ;;
    *)
        echo "PEAR Neural Compression - Batch Processing"
        echo "=========================================="
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start    - Start compression jobs for all datasets"
        echo "  monitor  - Monitor compression progress"  
        echo "  results  - Show final results"
        echo "  all      - Start jobs and begin monitoring"
        echo ""
        echo "Manual tmux commands:"
        echo "  tmux list-sessions              - List all sessions"
        echo "  tmux attach -t compress_<name>  - Attach to specific job"
        echo "  tmux kill-session -t <name>     - Kill a session"
        echo ""
        echo "Dataset files:"
        ls -lh /data2/luzeyi/baseset/PAC/v1_compression_data/ | grep -v "^total"
        ;;
esac