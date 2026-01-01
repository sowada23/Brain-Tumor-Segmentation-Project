"""
Utilities for logging trainning metadata to a SQLite database.

- 'ExperimentDB' 
   Wraps a SQLite connection and exposes small helper methods that the runner.py can call while tranining. 
   1. The tranning script imports this module
   2. Instantiates 'ExperimentDB' with the shared 'experiments.sqlite' path
   3. Invokes the logging methods after each major event (traninng start, every epoch, best checkpoint and test run)
   4. Save the database next to 'Output/' artifacts so it is easy to query.

"""

import datetime
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping



class ExperimentDB:
    """Lightweight wrapper around a SQLite database for experiment tracking."""

    def __init__(self, db_path):
        """Open or create a new database file and ensure tables exist."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def _init_schema(self):
        """Create tables the first time the DB is opened."""

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                config_json TEXT,
                best_val_dice REAL,
                best_checkpoint_path TEXT,
                best_epoch INTEGER
            );

            CREATE TABLE IF NOT EXISTS epochs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                train_accuracy REAL,
                train_dice_no_bg REAL,
                train_mean_iou REAL,
                val_loss REAL,
                val_accuracy REAL,
                val_dice_no_bg REAL,
                val_mean_iou REAL,
                lr REAL,
                logged_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS test_results(
                run_id TEXT PRIMARY KEY,
                loss REAL,
                accuracy REAL,
                dice_no_bg REAL,
                mean_iou REAL,
                logged_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """

        )
        self.conn.commit()

    def log_run_start(self, run_id, output_dir, config: Mapping[str, Any]):
        """Insert a run row if it doesn't exit."""

        config_json = json.dumps(config, default=str) # ?
        self.conn.execute(
            """
            INSERT OR IGNORE INTO runs (run_id, started_at, output_dir, config_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, datetime.datetime.utcnow().isoformat(), str(output_dir), config_json)
        )
        self.conn.commit()

    def log_epoch(self, run_id: str, epoch: int, train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any], lr: float):
        """Record metrics for a single trainning epoch."""

        self.conn.execute(
            """
            INSERT INTO epochs (
                run_id, epoch,
                train_loss, train_accuracy, train_dice_no_bg, train_mean_iou,
                val_loss, val_accuracy, val_dice_no_bg, val_mean_iou,
                lr, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(epoch),
                _safe_float(train_metrics.get("loss")),
                _safe_float(train_metrics.get("accuracy")),
                _safe_float(train_metrics.get("dice_no_bg")),
                _safe_float(train_metrics.get("mean_iou")),
                _safe_float(val_metrics.get("loss")),
                _safe_float(val_metrics.get("accuracy")),
                _safe_float(val_metrics.get("dice_no_bg")),
                _safe_float(val_metrics.get("mean_iou")),
                _safe_float(lr),
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def log_best_checkpoint(self, run_id, epoch, checkpoint_path, val_dice):
        """Update the 'runs' table with the best validation Dice and checkpoint. """

        self.conn.execute(
            """
            UPDATE runs
            SET best_val_dice = ?, best_checkpoint_path = ?, best_epoch = ?
            WHERE run_id = ? 
            """,
            (_safe_float(val_dice), str(checkpoint_path), int(epoch), run_id),
        )
        self.conn.commit()

    def log_test_results(self, run_id, metrics: Mapping[str, Any]):
        """Persist final test metrics after training finishes."""

        self.conn.execute(
            """
            INSERT OR REPLACE INTO test_results (
            run_id, loss, accuracy, dice_no_bg, mean_iou, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _safe_float(metrics.get("loss")),
                _safe_float(metrics.get("accuracy")),
                _safe_float(metrics.get("dice_no_bg")),
                _safe_float(metrics.get("mean_iou")),
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number