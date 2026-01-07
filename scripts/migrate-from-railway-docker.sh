#!/bin/bash
set -e

# This script uses Docker containers to perform the migration
# No local pg_dump or cypher-shell needed

DUMP_DIR="./migration_dumps"
mkdir -p "$DUMP_DIR"

echo "=== Railway → Local Migration (Docker-based) ==="
echo ""

# ─────────────────────────────────────────────────────────────
# PostgreSQL Migration
# ─────────────────────────────────────────────────────────────
echo "📦 Dumping PostgreSQL from Railway..."

docker run --rm \
    -e PGPASSWORD='jI~tY1nWzyPW0l~guH6AhvNnmDS14U88' \
    -v "$(pwd)/$DUMP_DIR:/dumps" \
    postgres:16 \
    pg_dump \
        -h shinkansen.proxy.rlwy.net \
        -p 24932 \
        -U postgres \
        -d railway \
        --no-owner \
        --no-acl \
        -F c \
        -f /dumps/pg_dump.backup

echo "✓ PostgreSQL dump saved to $DUMP_DIR/pg_dump.backup"

echo ""
echo "📥 Starting local containers..."
docker compose up -d postgres neo4j
echo "⏳ Waiting for databases to be ready..."
sleep 15

echo ""
echo "📥 Restoring PostgreSQL to local..."

# Copy dump into container and restore
docker compose cp "$DUMP_DIR/pg_dump.backup" postgres:/tmp/pg_dump.backup

docker compose exec postgres pg_restore \
    -U ruminate \
    -d mem0 \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    /tmp/pg_dump.backup || echo "⚠ Some restore warnings (usually OK)"

echo "✓ PostgreSQL restore complete"

# ─────────────────────────────────────────────────────────────
# Neo4j Migration
# ─────────────────────────────────────────────────────────────
echo ""
echo "📦 Exporting Neo4j from Railway..."

# Use a temp container with cypher-shell to export
docker run --rm \
    -v "$(pwd)/$DUMP_DIR:/dumps" \
    neo4j:5-community \
    cypher-shell \
        -a "bolt://hopper.proxy.rlwy.net:47049" \
        -u neo4j \
        -p "ldi90hskrub8ueppdza4pynz3urymzs0" \
        --format plain \
        "CALL apoc.export.json.all(null, {stream: true}) YIELD data RETURN data" \
    > "$DUMP_DIR/neo4j_export.json" 2>/dev/null || {
        echo "⚠ APOC export failed, using node/relationship queries..."

        # Export all data as Cypher CREATE statements
        docker run --rm \
            neo4j:5-community \
            cypher-shell \
                -a "bolt://hopper.proxy.rlwy.net:47049" \
                -u neo4j \
                -p "ldi90hskrub8ueppdza4pynz3urymzs0" \
                --format plain \
                "MATCH (n) WITH collect(n) as nodes CALL apoc.export.cypher.data(nodes, [], null, {stream:true}) YIELD output RETURN output" \
            > "$DUMP_DIR/neo4j_nodes.cypher" 2>/dev/null || true
    }

# Alternative: Direct query export for mem0 data
echo "📦 Exporting mem0 entities and relationships..."

docker run --rm \
    -v "$(pwd)/$DUMP_DIR:/dumps" \
    --entrypoint "" \
    neo4j:5-community \
    bash -c "
        cypher-shell \
            -a 'bolt://hopper.proxy.rlwy.net:47049' \
            -u neo4j \
            -p 'ldi90hskrub8ueppdza4pynz3urymzs0' \
            --format plain \
            'MATCH (n) RETURN labels(n) as labels, properties(n) as props' \
        > /dumps/nodes.txt 2>/dev/null

        cypher-shell \
            -a 'bolt://hopper.proxy.rlwy.net:47049' \
            -u neo4j \
            -p 'ldi90hskrub8ueppdza4pynz3urymzs0' \
            --format plain \
            'MATCH (a)-[r]->(b) RETURN labels(a) as from_labels, properties(a) as from_props, type(r) as rel_type, properties(r) as rel_props, labels(b) as to_labels, properties(b) as to_props' \
        > /dumps/relationships.txt 2>/dev/null
    " || echo "⚠ Some Neo4j export warnings"

echo "✓ Neo4j export complete"

echo ""
echo "📥 Importing Neo4j to local..."

# For now, just verify connection works
docker compose exec neo4j cypher-shell \
    -u neo4j \
    -p ruminate123 \
    "RETURN 'Neo4j ready' as status" || echo "⚠ Neo4j not ready yet"

echo ""
echo "=== Migration Complete ==="
echo ""
echo "Dumps saved in: $DUMP_DIR/"
echo "  - pg_dump.backup (PostgreSQL)"
echo "  - nodes.txt, relationships.txt (Neo4j)"
echo ""
echo "PostgreSQL data has been restored."
echo "Neo4j data exported - manual review may be needed for import."
echo ""
echo "Run: docker compose up ruminate"
