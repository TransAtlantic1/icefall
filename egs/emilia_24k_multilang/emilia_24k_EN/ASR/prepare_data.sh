#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "prepare_data.sh is a compatibility wrapper for the emilia_24k main pipeline."
echo "Forwarding all arguments to ${SCRIPT_DIR}/prepare.sh"

exec bash "${SCRIPT_DIR}/prepare.sh" "$@"
