#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get absolute path to this script's directory (pgvector root)
PGVECTOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PostgreSQL build directory and pg_config
PG_CONFIG="/home/$USER/postgresql/pg_build_tmp_2/bin/pg_config"

# Make sure pg_config exists
if [ ! -x "$PG_CONFIG" ]; then
  echo "Error: pg_config not found at $PG_CONFIG"
  exit 1
fi

echo "Building pgvector..."
cd "$PGVECTOR_DIR"
# make clean
make

echo "Installing pgvector extension..."

# Use pg_config to get installation paths
PKGLIBDIR=$($PG_CONFIG --pkglibdir)
SHAREDIR=$($PG_CONFIG --sharedir)

# Copy files
cp "$PGVECTOR_DIR/vector.so" "$PKGLIBDIR"
cp "$PGVECTOR_DIR/vector.control" "$SHAREDIR/extension/"
cp "$PGVECTOR_DIR"/sql/vector--*.sql "$SHAREDIR/extension/"

echo "✅ pgvector installed successfully."
echo "To use it in PostgreSQL, run:"
echo "  CREATE EXTENSION vector;"

# Optional: uncomment the next line to create the extension automatically
# psql -d your_database_name -c "CREATE EXTENSION IF NOT EXISTS vector;"
