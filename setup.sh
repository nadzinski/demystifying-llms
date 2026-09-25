#!/bin/sh
# EDUCATIONAL USE ONLY: not for company work or company data. See NOTICE.md.
#
# One-time setup: installs uv if you don't have it, then downloads the model and runs it once.
# Safe to re-run: if the download fails halfway, just run ./setup.sh again.
set -e
cd "$(dirname "$0")"

installed_uv=no
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (a Python package manager)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    installed_uv=yes
fi

uv run check.py

if [ "$installed_uv" = yes ]; then
    echo
    echo "uv was just installed: open a new terminal window before running 'uv run chat.py'."
fi
