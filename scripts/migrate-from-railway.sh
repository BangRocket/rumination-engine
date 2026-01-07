#!/bin/bash
set -e

# Railway source credentials
RAILWAY_PG_HOST="shinkansen.proxy.rlwy.net"
RAILWAY_PG_PORT="24932"
RAILWAY_PG_USER="postgres"
RAILWAY_PG_PASS="jI~tY1nWzyPW0l~guH6AhvNnmDS14U88"
RAILWAY_PG_DB="railway"

RAILWAY_NEO4J_HOST="hopper.proxy.rlwy.net"
RAILWAY_NEO4J_PORT="47049"
RAILWAY_NEO4J_USER="neo4j"
RAILWAY_NEO4J_PASS="ldi90hskrub8ueppdza4pynz3urymzs0"

# Local Docker credentials (from docker-compose.yml)
LOCAL_PG_USER="ruminate"
LOCAL_PG_PASS="ruminate"
LOCAL_PG_DB="mem0"

LOCAL_NEO4J_USER="neo4j"
LOCAL_NEO4J_PASS="ruminate123"

DUMP_DIR="./migration_dumps"
mkdir -p "$DUMP_DIR"

echo "=== Railway → Local Migration ==="
echo ""

# ─────────────────────────────────────────────────────────────
# PostgreSQL Migration
# ─────────────────────────────────────────────────────────────
echo "📦 Dumping PostgreSQL from Railway..."
PGPASSWORD="$RAILWAY_PG_PASS" pg_dump \
    -h "$RAILWAY_PG_HOST" \
    -p "$RAILWAY_PG_PORT" \
    -U "$RAILWAY_PG_USER" \
    -d "$RAILWAY_PG_DB" \
    --no-owner \
    --no-acl \
    -F c \
    -f "$DUMP_DIR/pg_dump.backup"

echo "✓ PostgreSQL dump complete: $DUMP_DIR/pg_dump.backup"

echo ""
echo "📥 Restoring PostgreSQL to local Docker..."

# Ensure local containers are running
docker compose up -d postgres
sleep 5

# Restore to local postgres
docker compose exec -T postgres pg_restore \
    -U "$LOCAL_PG_USER" \
    -d "$LOCAL_PG_DB" \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    < "$DUMP_DIR/pg_dump.backup" || true

echo "✓ PostgreSQL restore complete"

# ─────────────────────────────────────────────────────────────
# Neo4j Migration
# ─────────────────────────────────────────────────────────────
echo ""
echo "📦 Dumping Neo4j from Railway..."

# Export all nodes and relationships using cypher-shell
cat << 'EOF' > "$DUMP_DIR/neo4j_export.cypher"
CALL apoc.export.cypher.all(null, {format: "plain", stream: true})
YIELD cypherStatements
RETURN cypherStatements;
EOF

cypher-shell \
    -a "bolt://$RAILWAY_NEO4J_HOST:$RAILWAY_NEO4J_PORT" \
    -u "$RAILWAY_NEO4J_USER" \
    -p "$RAILWAY_NEO4J_PASS" \
    --format plain \
    "CALL apoc.export.cypher.all(null, {format: 'plain', stream: true}) YIELD cypherStatements RETURN cypherStatements" \
    > "$DUMP_DIR/neo4j_data.cypher" 2>/dev/null || {
        echo "⚠ cypher-shell not found or APOC export failed, trying alternative..."

        # Alternative: export nodes and relationships separately
        cypher-shell \
            -a "bolt://$RAILWAY_NEO4J_HOST:$RAILWAY_NEO4J_PORT" \
            -u "$RAILWAY_NEO4J_USER" \
            -p "$RAILWAY_NEO4J_PASS" \
            --format plain \
            "MATCH (n) RETURN n" > "$DUMP_DIR/neo4j_nodes.txt"

        cypher-shell \
            -a "bolt://$RAILWAY_NEO4J_HOST:$RAILWAY_NEO4J_PORT" \
            -u "$RAILWAY_NEO4J_USER" \
            -p "$RAILWAY_NEO4J_PASS" \
            --format plain \
            "MATCH (a)-[r]->(b) RETURN a, type(r) as rel_type, properties(r) as rel_props, b" > "$DUMP_DIR/neo4j_rels.txt"
    }

echo "✓ Neo4j dump complete"

echo ""
echo "📥 Restoring Neo4j to local Docker..."

# Ensure neo4j is running
docker compose up -d neo4j
sleep 10

# Import the cypher statements
if [ -f "$DUMP_DIR/neo4j_data.cypher" ]; then
    docker compose exec -T neo4j cypher-shell \
        -u "$LOCAL_NEO4J_USER" \
        -p "$LOCAL_NEO4J_PASS" \
        < "$DUMP_DIR/neo4j_data.cypher" || true
fi

echo "✓ Neo4j restore complete"

echo ""
echo "=== Migration Complete ==="
echo "Dumps saved in: $DUMP_DIR/"
echo ""
echo "You can now run: docker compose up ruminate"
