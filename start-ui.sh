#!/usr/bin/env bash


set -e
source /AI2/onetrainer-alt/env.sh

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

prepare_runtime_environment

run_python_in_active_env "scripts/train_ui.py" "$@"
