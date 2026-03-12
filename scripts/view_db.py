"""
Simple untility to inspect the experiments SQLite database.

1. This script prints tables, their schemes, row counts, and the first few rows. 
2. By default, it looks for `Output/experiemts.sqlite` created during training, but 
   can point it to any SQLite file with ``--db``.

"""

import argparse 
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple



def _print_header(title: str):
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")

def _iter_tables(conn: sqlite3.Connection):
    cursor = conn.execute(
        "SELECT  name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    return (row[0] for row in cursor.fetchall())

def _fetch_schema(conn: sqlite3.Connection, table):
    cursor = conn.execute(f"PRAGMA table_info{table};")
    return tuple(row[1] for row in cursor.fetchall())

def _print_table_preview(conn: sqlite3.Connection, table, limit):
    columns = _fetch_schema(conn, table)
    cursor = conn.execute(f"SELECT COUNT (*) FROM {table};")
    count = cursor.fetchone()[0]
    print(f"Table: {table} (rows: {count})")
    print("Columns: ", ", ".join(columns))

    if count == 0:
        print("(no rows)\n")
        return 
    
    cursor = conn.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?;", (limit,))
    rows = cursor.fetchall()
    for idx, row in enumerate(rows, start=1):
        print(f"ROW {idx}: {row}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspet experiments SQLite database")
    parser.add_argument(
        "--db",
        default=Path("Output/experiment.sqlite"),
        type=Path,
        help="Path to SQLite database (default to Output/experiment.sqlite)",
    )
    parser.add_argument(
        "--limit",
        default=5,
        type=int,
        help="Number of rows to preview from each table (default: 5)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")
    
    conn = sqlite3.connect(args.db)
    try:
        tables = list(_iter_tables(conn))
        if not tables:
            raise SystemExit(f"No user table found")
        
        _print_header(f"Database: {args.db}")
        for table in tables:
            _print_table_preview(conn, table, args.limit)
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()