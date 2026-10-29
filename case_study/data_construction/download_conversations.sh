#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export AWS_VAULT_BACKEND="${AWS_VAULT_BACKEND:-file}"
export AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
export AWS_DEFAULT_REGION="$AWS_REGION"
if [[ "${1:-}" == --help ]]; then
  exec python3 "$script_dir/download_conversations.py" --help
fi
if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
  exec python3 "$script_dir/download_conversations.py" --region "$AWS_REGION" "$@"
fi
exec aws-vault exec "${AWS_VAULT_PROFILE:-myprofile}" -- \
  python3 "$script_dir/download_conversations.py" --region "$AWS_REGION" "$@"
