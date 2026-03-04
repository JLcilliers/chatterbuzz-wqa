#!/bin/bash
set -e

echo "=== Chatterbuzz SEO Platform Setup ==="

# Check prerequisites
command -v pnpm >/dev/null 2>&1 || { echo "pnpm is required. Install: npm i -g pnpm"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required"; exit 1; }

# Install Node dependencies
echo "Installing Node.js dependencies..."
pnpm install

# Set up Python virtual environments for each Python package
for pkg in packages/wqa-engine packages/data-pipeline packages/content-engine packages/schema-generator packages/api; do
  if [ -f "$pkg/requirements.txt" ]; then
    echo "Setting up Python venv for $pkg..."
    python3 -m venv "$pkg/.venv"
    source "$pkg/.venv/bin/activate"
    pip install -r "$pkg/requirements.txt"
    deactivate
  fi
done

# Copy env if not exists
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill in your values"
fi

echo "=== Setup complete ==="
