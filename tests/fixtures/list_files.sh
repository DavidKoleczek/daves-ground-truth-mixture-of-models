#!/usr/bin/env bash
# List files in the given directory (defaults to current directory).
target_dir="${1:-.}"
echo "=== Files in ${target_dir} ==="
ls -1 "${target_dir}"
