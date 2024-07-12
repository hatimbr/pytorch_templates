from mlflow import log_params, search_runs, set_experiment, set_tracking_uri, start_run
from config import GlobalConfig


class MlTrackContext:
    """Wrapper for mlflow tracking context manager."""

    def __init__(self, config: GlobalConfig, activate: bool = True):
        self.config = config
        self.activate = activate

    def __enter__(self):
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


def mltrack_context(config: GlobalConfig, activate: bool = True):
    """Custom context manager for mlflow tracking."""
    return MlTrackContext(config, activate)
