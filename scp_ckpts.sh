#!/usr/bin/env bash
# Copy ckpts/ contents to another machine, excluding final-round and trash-bin.
set -euo pipefail

source_dir="/home/huanghui/WorkSpace/ckpts"
remote_host="" # Set to your SSH alias or username@hostname.
remote_dir="" # Set to the absolute destination path, e.g. /home/user/other-repo/ckpts.
ssh_port=22

: "${remote_host:?Set remote_host before running this script}"
: "${remote_dir:?Set remote_dir before running this script}"

# 1. Select all top-level entries except the two archived directories.
cd -- "$source_dir"
mapfile -d '' -t checkpoints < <(
    find "$PWD" -mindepth 1 -maxdepth 1 \
        ! -name final-round ! -name trash-bin -print0
)
: "${checkpoints[0]:?No checkpoint entries to copy}"

# 2. Create the remote destination. Quote the path for the remote shell.
printf -v remote_dir_quoted '%q' "$remote_dir"
ssh -p "$ssh_port" "$remote_host" "mkdir -p -- $remote_dir_quoted"

# 3. Copy the selected entries directly into the destination (no extra ckpts/ layer).
# Includes top-level metadata files; preserves file timestamps and permissions.
printf 'Copying %s entries to %s:%s\n' "${#checkpoints[@]}" "$remote_host" "$remote_dir"
scp -r -p -P "$ssh_port" "${checkpoints[@]}" "${remote_host}:${remote_dir}/"
