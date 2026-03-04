#!/bin/bash
set -e

echo "Generating Supabase TypeScript types..."

# Requires: npx supabase gen types typescript
if [ -z "$SUPABASE_DB_URL" ]; then
  echo "SUPABASE_DB_URL not set. Source .env first."
  exit 1
fi

npx supabase gen types typescript \
  --db-url "$SUPABASE_DB_URL" \
  > packages/shared/src/types/database.ts

echo "Types written to packages/shared/src/types/database.ts"
