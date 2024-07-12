from pathlib import Path

from mlflow import (
    log_params, search_runs, set_experiment, set_tracking_uri, start_run, ActiveRun
)
from torch import profiler

from config import GlobalConfig


class MlTrackContext:
    """Wrapper for mlflow tracking context manager."""

    def __init__(self, config: GlobalConfig, activate: bool = True):
        self.config = config
        self.activate = activate

    def __enter__(self) -> ActiveRun | None:
        if self.activate:
            set_tracking_uri(self.config.mlflow_dir)
            set_experiment(self.config.experiment_name)

            if search_runs(
                filter_string=f"run_name='{self.config.run_name}'"
            ).empty:
                run_id = None
            else:
                run_id = search_runs(
                    filter_string=f"run_name='{self.config.run_name}'"
                ).iloc[0]["run_id"]

            self.run = start_run(
                run_name=self.config.run_name, run_id=run_id, log_system_metrics=True
            )
            self.run.__enter__()
            log_params(self.config.export())
            return self.run

        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activate:
            self.run.__exit__(exc_type, exc_val, exc_tb)
        else:
            return exc_type is None


class TorchProfilerContext:
    """Wrapper for PyTorch profiler context manager."""

    def __init__(
        self, profiler_dir: Path = Path().cwd() / "profiler", profile: bool = False
    ):
        self.profiler_dir = profiler_dir
        self.profile = profile

    def __enter__(self) -> profiler.profile | None:
        if self.profile:
            self.profiler = profiler.profile(
                schedule=profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
                on_trace_ready=profiler.tensorboard_trace_handler(
                    str(self.profiler_dir)
                ),
                profile_memory=True,
                with_stack=False,
                record_shapes=False,
            )
            self.profiler.__enter__()
            return self.profiler
        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)


def mltrack_context(config: GlobalConfig, activate: bool = True) -> MlTrackContext:
    """Custom context manager for mlflow tracking."""
    return MlTrackContext(config, activate)


def torch_profiler_context(
    profiler_dir: Path = Path().cwd() / "profiler", profile: bool = False
) -> TorchProfilerContext:
    """Custom context manager for PyTorch profiler."""
    return TorchProfilerContext(profiler_dir, profile)
