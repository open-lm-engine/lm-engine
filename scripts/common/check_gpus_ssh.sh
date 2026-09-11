#!/bin/bash
# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************
#
# SSHes into each node in a nodelist, one at a time, and runs check_gpu.py
# to verify every GPU on that node can allocate a tensor and do a matmul.
#
# usage: ./check_gpus_ssh.sh <nodelist.txt> [ssh_user] [python_bin]
#
# nodelist.txt: one hostname/IP per line, blank lines and lines starting
#               with '#' are ignored.

set -u

NODELIST="${1:?usage: $0 <nodelist.txt> [ssh_user] [python_bin]}"
SSH_USER="${2:-}"
PYTHON_BIN="${3:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="$SCRIPT_DIR/check_gpu.py"

if [[ ! -f "$NODELIST" ]]; then
    echo "nodelist file not found: $NODELIST" >&2
    exit 1
fi

if [[ ! -f "$CHECK_SCRIPT" ]]; then
    echo "missing $CHECK_SCRIPT" >&2
    exit 1
fi

mapfile -t NODES < <(grep -vE '^\s*(#|$)' "$NODELIST")

if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "no nodes found in $NODELIST" >&2
    exit 1
fi

declare -a FAILED_NODES=()

for node in "${NODES[@]}"; do
    target="$node"
    [[ -n "$SSH_USER" ]] && target="$SSH_USER@$node"

    echo "==================== $node ===================="
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "$PYTHON_BIN" - < "$CHECK_SCRIPT"; then
        echo "[$node] PASS"
    else
        echo "[$node] FAIL"
        FAILED_NODES+=("$node")
    fi
    echo
done

echo "================== summary =================="
echo "checked ${#NODES[@]} node(s), ${#FAILED_NODES[@]} failed"

if [[ ${#FAILED_NODES[@]} -gt 0 ]]; then
    printf 'FAILED: %s\n' "${FAILED_NODES[@]}"
    exit 1
fi

echo "all nodes OK"
