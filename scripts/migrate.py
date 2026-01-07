#!/usr/bin/env python3
"""
Migrate mem0 data from Railway to local Docker.

Usage:
    python scripts/migrate.py

Requires: pip install psycopg2-binary neo4j
"""

import json
from pathlib import Path

import psycopg2
from psycopg2.extras import Json
from neo4j import GraphDatabase

# ─────────────────────────────────────────────────────────────
# Railway (source) credentials
# ─────────────────────────────────────────────────────────────
RAILWAY_PG = {
    "host": "shinkansen.proxy.rlwy.net",
    "port": 24932,
    "user": "postgres",
    "password": "jI~tY1nWzyPW0l~guH6AhvNnmDS14U88",
    "dbname": "railway",
}

RAILWAY_NEO4J = {
    "uri": "bolt://hopper.proxy.rlwy.net:47049",
    "user": "neo4j",
    "password": "ldi90hskrub8ueppdza4pynz3urymzs0",
}

# ─────────────────────────────────────────────────────────────
# Local Docker (destination) credentials
# ─────────────────────────────────────────────────────────────
LOCAL_PG = {
    "host": "localhost",
    "port": 5432,
    "user": "ruminate",
    "password": "ruminate",
    "dbname": "mem0",
}

LOCAL_NEO4J = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "ruminate123",
}

DUMP_DIR = Path("./migration_dumps")
DUMP_DIR.mkdir(exist_ok=True)


def migrate_postgres():
    """Migrate PostgreSQL/pgvector data."""
    print("\n📦 Connecting to Railway PostgreSQL...")

    src = psycopg2.connect(**RAILWAY_PG)
    src_cur = src.cursor()

    # Get all tables
    src_cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    tables = [row[0] for row in src_cur.fetchall()]
    print(f"   Found tables: {tables}")

    print("\n📥 Connecting to local PostgreSQL...")
    dst = psycopg2.connect(**LOCAL_PG)
    dst_cur = dst.cursor()

    # Ensure vector extension exists
    dst_cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    dst.commit()

    for table in tables:
        print(f"\n   Migrating table: {table}")

        # Get table schema
        src_cur.execute(f"""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns = src_cur.fetchall()

        # Create table if not exists (simplified - may need adjustment)
        col_defs = []
        col_names = []
        for col_name, data_type, udt_name in columns:
            col_names.append(col_name)
            if udt_name == "vector":
                # Get vector dimensions
                src_cur.execute(f"""
                    SELECT atttypmod FROM pg_attribute
                    WHERE attrelid = '{table}'::regclass AND attname = '{col_name}'
                """)
                dim = src_cur.fetchone()[0]
                col_defs.append(f'"{col_name}" vector({dim})')
            elif data_type == "ARRAY":
                col_defs.append(f'"{col_name}" text[]')
            elif data_type in ("character varying", "text"):
                col_defs.append(f'"{col_name}" text')
            elif data_type == "integer":
                col_defs.append(f'"{col_name}" integer')
            elif data_type == "bigint":
                col_defs.append(f'"{col_name}" bigint')
            elif data_type == "boolean":
                col_defs.append(f'"{col_name}" boolean')
            elif data_type == "jsonb":
                col_defs.append(f'"{col_name}" jsonb')
            elif data_type == "json":
                col_defs.append(f'"{col_name}" json')
            elif "timestamp" in data_type:
                col_defs.append(f'"{col_name}" timestamp')
            elif data_type == "uuid":
                col_defs.append(f'"{col_name}" uuid')
            else:
                col_defs.append(f'"{col_name}" text')

        # Drop and recreate table
        dst_cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        create_sql = f'CREATE TABLE "{table}" ({", ".join(col_defs)})'
        try:
            dst_cur.execute(create_sql)
            dst.commit()
        except Exception as e:
            print(f"   ⚠ Error creating table: {e}")
            dst.rollback()
            continue

        # Copy data
        src_cur.execute(f'SELECT * FROM "{table}"')
        rows = src_cur.fetchall()
        print(f"   Found {len(rows)} rows")

        if rows:
            placeholders = ", ".join(["%s"] * len(col_names))
            quoted_cols = ", ".join([f'"{c}"' for c in col_names])
            insert_sql = f'INSERT INTO "{table}" ({quoted_cols}) VALUES ({placeholders})'

            inserted = 0
            for row in rows:
                # Convert dicts/lists to Json for jsonb columns
                adapted_row = []
                for val in row:
                    if isinstance(val, (dict, list)):
                        adapted_row.append(Json(val))
                    else:
                        adapted_row.append(val)

                try:
                    dst_cur.execute(insert_sql, adapted_row)
                    inserted += 1
                except Exception as e:
                    print(f"   ⚠ Row insert error: {e}")
                    dst.rollback()
                    continue

            dst.commit()
            print(f"   ✓ Inserted {inserted}/{len(rows)} rows")

    src.close()
    dst.close()
    print("\n✓ PostgreSQL migration complete")


def migrate_neo4j():
    """Migrate Neo4j graph data."""
    print("\n📦 Connecting to Railway Neo4j...")

    src_driver = GraphDatabase.driver(
        RAILWAY_NEO4J["uri"],
        auth=(RAILWAY_NEO4J["user"], RAILWAY_NEO4J["password"])
    )

    # Export nodes
    print("   Exporting nodes...")
    nodes = []
    with src_driver.session() as session:
        result = session.run("MATCH (n) RETURN labels(n) as labels, properties(n) as props, elementId(n) as id")
        for record in result:
            nodes.append({
                "id": record["id"],
                "labels": record["labels"],
                "props": dict(record["props"]) if record["props"] else {}
            })
    print(f"   Found {len(nodes)} nodes")

    # Export relationships
    print("   Exporting relationships...")
    rels = []
    with src_driver.session() as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN elementId(a) as from_id, type(r) as rel_type,
                   properties(r) as rel_props, elementId(b) as to_id
        """)
        for record in result:
            rels.append({
                "from_id": record["from_id"],
                "type": record["rel_type"],
                "props": dict(record["rel_props"]) if record["rel_props"] else {},
                "to_id": record["to_id"]
            })
    print(f"   Found {len(rels)} relationships")

    src_driver.close()

    # Save to JSON for backup
    with open(DUMP_DIR / "neo4j_nodes.json", "w") as f:
        json.dump(nodes, f, indent=2, default=str)
    with open(DUMP_DIR / "neo4j_rels.json", "w") as f:
        json.dump(rels, f, indent=2, default=str)
    print(f"   Saved backups to {DUMP_DIR}/")

    # Import to local
    print("\n📥 Connecting to local Neo4j...")
    dst_driver = GraphDatabase.driver(
        LOCAL_NEO4J["uri"],
        auth=(LOCAL_NEO4J["user"], LOCAL_NEO4J["password"])
    )

    # Clear existing data
    print("   Clearing existing data...")
    with dst_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    # Create nodes with mapping from old IDs to new
    print("   Creating nodes...")
    id_map = {}
    with dst_driver.session() as session:
        for node in nodes:
            labels_str = ":".join(node["labels"]) if node["labels"] else "Node"
            # Create node and get new ID
            result = session.run(
                f"CREATE (n:{labels_str} $props) RETURN elementId(n) as new_id",
                props=node["props"]
            )
            new_id = result.single()["new_id"]
            id_map[node["id"]] = new_id

    print(f"   ✓ Created {len(nodes)} nodes")

    # Create relationships
    print("   Creating relationships...")
    rel_count = 0
    with dst_driver.session() as session:
        for rel in rels:
            if rel["from_id"] in id_map and rel["to_id"] in id_map:
                # Sanitize relationship type
                rel_type = rel["type"].replace("-", "_").replace(" ", "_")
                try:
                    session.run(
                        f"""
                        MATCH (a), (b)
                        WHERE elementId(a) = $from_id AND elementId(b) = $to_id
                        CREATE (a)-[r:{rel_type} $props]->(b)
                        """,
                        from_id=id_map[rel["from_id"]],
                        to_id=id_map[rel["to_id"]],
                        props=rel["props"]
                    )
                    rel_count += 1
                except Exception as e:
                    print(f"   ⚠ Relationship error: {e}")

    print(f"   ✓ Created {rel_count} relationships")

    dst_driver.close()
    print("\n✓ Neo4j migration complete")


def main():
    print("=" * 60)
    print("Railway → Local Migration")
    print("=" * 60)

    print("\n⚠ Make sure local containers are running:")
    print("  docker compose up -d postgres neo4j")
    print("  (wait ~30s for neo4j to initialize)\n")

    migrate_postgres()
    migrate_neo4j()

    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print("\nYou can now run: docker compose up ruminate")


if __name__ == "__main__":
    main()
