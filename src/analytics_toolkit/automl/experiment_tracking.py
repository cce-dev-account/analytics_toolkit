"""
Experiment tracking and model registry for AutoML workflows.
"""

import json
import pickle
import sqlite3
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class RunMetrics:
    """Container for experiment run metrics."""

    run_id: str
    experiment_name: str
    model_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, str] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: str = "running"  # 'running', 'finished', 'failed'
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetrics":
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


class ExperimentTracker:
    """
    MLflow-style experiment tracking for AutoML workflows.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "automl_experiment",
        auto_log: bool = True,
    ):
        """
        Initialize ExperimentTracker.

        Parameters
        ----------
        tracking_uri : str, optional
            Path to tracking directory (defaults to ./mlruns)
        experiment_name : str, default="automl_experiment"
            Name of the experiment
        auto_log : bool, default=True
            Enable automatic logging of model parameters and metrics
        """
        self.tracking_uri = tracking_uri or "./mlruns"
        self.experiment_name = experiment_name
        self.auto_log = auto_log

        # Create tracking directory
        self.tracking_path = Path(self.tracking_uri)
        self.tracking_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.tracking_path / "experiments.db"
        self._init_database()

        self.current_run = None
        self._run_stack = []

    def _init_database(self):
        """Initialize SQLite database for tracking."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")  # Better for concurrent access
        cursor = conn.cursor()

        # Create experiments table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_time TEXT NOT NULL,
                tags TEXT DEFAULT '{}'
            )
        """
        )

        # Create runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                run_name TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """
        )

        # Create metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                step INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """
        )

        # Create parameters table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS parameters (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """
        )

        # Create tags table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """
        )

        conn.commit()
        conn.close()

    def create_experiment(
        self, name: str, tags: Optional[dict[str, str]] = None
    ) -> str:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())
        tags = tags or {}

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments (experiment_id, name, created_time, tags)
                VALUES (?, ?, ?, ?)
            """,
                (experiment_id, name, datetime.now().isoformat(), json.dumps(tags)),
            )
            conn.commit()

        self.experiment_name = name
        return experiment_id

    def get_experiment_id(self, name: str) -> Optional[str]:
        """Get experiment ID by name."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?", (name,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Start a new run.

        Parameters
        ----------
        run_name : str, optional
            Name for the run
        nested : bool, default=False
            Whether this is a nested run
        tags : dict, optional
            Tags to associate with the run

        Returns
        -------
        str
            Run ID
        """
        run_id = str(uuid.uuid4())

        # Get or create experiment
        experiment_id = self.get_experiment_id(self.experiment_name)
        if experiment_id is None:
            experiment_id = self.create_experiment(self.experiment_name)

        # Create run record
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (run_id, experiment_id, run_name, start_time)
                VALUES (?, ?, ?, ?)
            """,
                (run_id, experiment_id, run_name, datetime.now().isoformat()),
            )

            # Add tags
            if tags:
                for key, value in tags.items():
                    cursor.execute(
                        """
                        INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)
                    """,
                        (run_id, key, str(value)),
                    )

            conn.commit()

        # Create run directory
        run_dir = self.tracking_path / run_id
        run_dir.mkdir(exist_ok=True)

        # Create RunMetrics object
        run_metrics = RunMetrics(
            run_id=run_id,
            experiment_name=self.experiment_name,
            model_name="",
            parameters={},
            metrics={},
            tags=tags or {},
            notes="",
        )

        # Handle nested runs
        if nested and self.current_run:
            self._run_stack.append(self.current_run)

        self.current_run = run_metrics
        return run_id

    def end_run(self, status: str = "finished"):
        """End the current run."""
        if self.current_run is None:
            return

        self.current_run.end_time = datetime.now()
        self.current_run.duration_seconds = (
            self.current_run.end_time - self.current_run.start_time
        ).total_seconds()
        self.current_run.status = status

        # Update database
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs SET end_time = ?, status = ? WHERE run_id = ?
            """,
                (
                    self.current_run.end_time.isoformat(),
                    status,
                    self.current_run.run_id,
                ),
            )
            conn.commit()

        # Save run metadata
        run_dir = self.tracking_path / self.current_run.run_id
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2)

        # Handle nested runs
        if self._run_stack:
            self.current_run = self._run_stack.pop()
        else:
            self.current_run = None

    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self.current_run is None:
            warnings.warn("No active run. Start a run first.")
            return

        self.current_run.parameters[key] = value

        # Update database
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO parameters (run_id, key, value)
                VALUES (?, ?, ?)
            """,
                (self.current_run.run_id, key, str(value)),
            )
            conn.commit()

    def log_params(self, params: dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int = 0):
        """Log a metric."""
        if self.current_run is None:
            warnings.warn("No active run. Start a run first.")
            return

        self.current_run.metrics[key] = value

        # Update database
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics (run_id, key, value, timestamp, step)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    self.current_run.run_id,
                    key,
                    float(value),
                    datetime.now().isoformat(),
                    step,
                ),
            )
            conn.commit()

    def log_metrics(self, metrics: dict[str, float], step: int = 0):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_model(
        self,
        model: BaseEstimator,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """Log a trained model."""
        if self.current_run is None:
            warnings.warn("No active run. Start a run first.")
            return

        run_dir = self.tracking_path / self.current_run.run_id
        model_dir = run_dir / artifact_path
        model_dir.mkdir(exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save model info
        model_info = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "artifact_path": artifact_path,
            "serialization_format": "pickle",
        }

        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        # Update current run
        self.current_run.artifacts[artifact_path] = str(model_path.relative_to(run_dir))
        self.current_run.model_name = model.__class__.__name__

        # Log to model registry if specified
        if registered_model_name:
            self._register_model(model, registered_model_name, str(model_path))

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.current_run is None:
            warnings.warn("No active run. Start a run first.")
            return

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")

        run_dir = self.tracking_path / self.current_run.run_id
        if artifact_path is None:
            artifact_path = local_path.name

        # Copy artifact to run directory
        import shutil

        artifact_dest = run_dir / artifact_path
        artifact_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, artifact_dest)

        # Update current run
        self.current_run.artifacts[artifact_path] = str(
            artifact_dest.relative_to(run_dir)
        )

    def _register_model(self, model: BaseEstimator, name: str, model_path: str):
        """Register model in model registry."""
        # This would integrate with a model registry service
        # For now, just log the information
        registry_info = {
            "model_name": name,
            "version": 1,  # Would increment in real implementation
            "run_id": self.current_run.run_id,
            "model_path": model_path,
            "registered_time": datetime.now().isoformat(),
        }

        registry_path = self.tracking_path / "model_registry.json"

        # Load existing registry
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {}

        # Add model
        if name not in registry:
            registry[name] = []
        registry[name].append(registry_info)

        # Save updated registry
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        """Get run by ID."""
        run_dir = self.tracking_path / run_id
        metadata_path = run_dir / "run_metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        return RunMetrics.from_dict(data)

    def search_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> list[RunMetrics]:
        """Search runs with filtering."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            query = """
                SELECT r.run_id FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                WHERE 1=1
            """
            params = []

            if experiment_name:
                query += " AND e.name = ?"
                params.append(experiment_name)

            if order_by:
                if order_by.startswith("-"):
                    query += f" ORDER BY {order_by[1:]} DESC"
                else:
                    query += f" ORDER BY {order_by} ASC"

            cursor.execute(query, params)
            run_ids = [row[0] for row in cursor.fetchall()]

        # Load run metadata
        runs = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs.append(run)

        return runs

    def compare_runs(self, run_ids: list[str]) -> pd.DataFrame:
        """Compare multiple runs."""
        runs = [self.get_run(run_id) for run_id in run_ids]
        runs = [run for run in runs if run is not None]

        if not runs:
            return pd.DataFrame()

        # Create comparison DataFrame
        data = []
        for run in runs:
            row = {
                "run_id": run.run_id,
                "model_name": run.model_name,
                "start_time": run.start_time,
                "duration_seconds": run.duration_seconds,
                "status": run.status,
            }

            # Add parameters
            for key, value in run.parameters.items():
                row[f"param_{key}"] = value

            # Add metrics
            for key, value in run.metrics.items():
                row[f"metric_{key}"] = value

            data.append(row)

        return pd.DataFrame(data)

    def get_best_run(
        self,
        metric_name: str,
        experiment_name: Optional[str] = None,
        maximize: bool = True,
    ) -> Optional[RunMetrics]:
        """Get the best run based on a metric."""
        runs = self.search_runs(experiment_name=experiment_name)

        if not runs:
            return None

        # Filter runs that have the metric
        filtered_runs = [run for run in runs if metric_name in run.metrics]

        if not filtered_runs:
            return None

        # Find best run
        best_run = (
            max(filtered_runs, key=lambda x: x.metrics[metric_name])
            if maximize
            else min(filtered_runs, key=lambda x: x.metrics[metric_name])
        )

        return best_run


class ModelRegistry:
    """
    Simple model registry for versioning and managing trained models.
    """

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.registry_path / "registry.db"
        self._init_database()

    def _init_database(self):
        """Initialize model registry database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS registered_models (
                    name TEXT PRIMARY KEY,
                    created_time TEXT NOT NULL,
                    updated_time TEXT NOT NULL,
                    description TEXT DEFAULT ''
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    model_path TEXT NOT NULL,
                    created_time TEXT NOT NULL,
                    stage TEXT DEFAULT 'None',
                    run_id TEXT,
                    metrics TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '{}',
                    PRIMARY KEY (name, version),
                    FOREIGN KEY (name) REFERENCES registered_models (name)
                )
            """
            )

            conn.commit()

    def register_model(
        self,
        model: BaseEstimator,
        name: str,
        description: str = "",
        tags: Optional[dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """Register a new model or version."""
        tags = tags or {}
        current_time = datetime.now().isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Create or update registered model
            cursor.execute(
                """
                INSERT OR REPLACE INTO registered_models (name, created_time, updated_time, description)
                VALUES (?,
                        COALESCE((SELECT created_time FROM registered_models WHERE name = ?), ?),
                        ?, ?)
            """,
                (name, name, current_time, current_time, description),
            )

            # Get next version number
            cursor.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM model_versions WHERE name = ?",
                (name,),
            )
            version = cursor.fetchone()[0]

            # Save model
            model_dir = self.registry_path / name / str(version)
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save model metadata
            metadata = {
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
                "version": version,
                "registered_time": current_time,
                "tags": tags,
            }

            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Register version in database
            cursor.execute(
                """
                INSERT INTO model_versions (name, version, model_path, created_time, run_id, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    version,
                    str(model_path),
                    current_time,
                    run_id,
                    json.dumps(tags),
                ),
            )

            conn.commit()

        return version

    def get_model(
        self, name: str, version: Optional[int] = None
    ) -> Optional[BaseEstimator]:
        """Load a registered model."""
        if version is None:
            # Get latest version
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT MAX(version) FROM model_versions WHERE name = ?", (name,)
                )
                result = cursor.fetchone()
                if result[0] is None:
                    return None
                version = result[0]

        model_path = self.registry_path / name / str(version) / "model.pkl"

        if not model_path.exists():
            return None

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM registered_models ORDER BY name")

            models = []
            for row in cursor.fetchall():
                models.append(
                    {
                        "name": row[0],
                        "created_time": row[1],
                        "updated_time": row[2],
                        "description": row[3],
                    }
                )

        return models

    def get_model_versions(self, name: str) -> list[dict[str, Any]]:
        """Get all versions of a model."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_versions WHERE name = ? ORDER BY version DESC",
                (name,),
            )

            versions = []
            for row in cursor.fetchall():
                versions.append(
                    {
                        "name": row[0],
                        "version": row[1],
                        "model_path": row[2],
                        "created_time": row[3],
                        "stage": row[4],
                        "run_id": row[5],
                        "metrics": json.loads(row[6]) if row[6] else {},
                        "tags": json.loads(row[7]) if row[7] else {},
                    }
                )

        return versions
