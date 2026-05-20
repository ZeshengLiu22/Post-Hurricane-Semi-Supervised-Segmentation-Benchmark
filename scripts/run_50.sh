#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The shared helper preflights and launches each method through its conda env.
exec "$SCRIPT_DIR/run_ratio_benchmark.sh" 50 "$@"
