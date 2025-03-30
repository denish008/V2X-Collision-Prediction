import inspect
import os


class Experiment:
    RUN_FILENAME = "run.py"
    RUN_NUM_PREFIX = "run"
    EXPERIMENT_NUM_PREFIX = "exp"
    EXPERIMENTS_DIRNAME = "experiments"
    MODEL_CHECKPOINTS_DIRNAME = "mcheckpoints"
    CSV_LOGGER_DIRNAME = "logs"
    PROFILE_DIRNAME = "profiles"

    def __init__(
        self, project: str, model_name: str, run_num: str, notes: str = ""
    ) -> None:
        self.project = project
        self.model_name = model_name
        self.notes = notes

        if not run_num.startswith(self.RUN_NUM_PREFIX):
            raise RuntimeError(
                f"Run number should start with prefix: {self.RUN_NUM_PREFIX}, got {run_num}"
            )

        self.run_num = run_num
        self.dirpath = self._get_dirpath()
        self.num = self._get_num()
        self.name = self._get_name()

        self.model_checkpoints_dirpath = os.path.join(
            self.dirpath, self.MODEL_CHECKPOINTS_DIRNAME, self.name
        )
        self.csv_logger_dirpath = os.path.join(
            self.dirpath,
            self.CSV_LOGGER_DIRNAME,
        )
        self.gcs_dirpath = os.path.join(
            self.project,
            self.EXPERIMENTS_DIRNAME,
            self.num,
            self.MODEL_CHECKPOINTS_DIRNAME,
            self.name,
        )
        self.profile_dirpath = os.path.join(self.dirpath, self.PROFILE_DIRNAME)

    def _get_dirpath(self) -> str:
        exp_filepath = inspect.stack()[-1].filename
        if not exp_filepath.endswith(self.RUN_FILENAME):
            raise RuntimeError(
                f"experiment file should be {self.RUN_FILENAME}, not {exp_filepath}"
            )

        exp_dirpath = os.path.dirname(exp_filepath)
        exp_num = os.path.basename(exp_dirpath)
        if not exp_num.startswith(self.EXPERIMENT_NUM_PREFIX):
            raise RuntimeError(
                f"experiment name should start with prefix: {self.EXPERIMENT_NUM_PREFIX}, got {exp_num}"
            )

        exp_dirname = os.path.dirname(exp_dirpath)
        if not exp_dirname.endswith(self.EXPERIMENTS_DIRNAME):
            raise RuntimeError(
                f"experiments dirname should be: {self.EXPERIMENTS_DIRNAME}, got {exp_dirname}"
            )
        return exp_dirpath

    def _get_num(self) -> str:
        exp_num = os.path.basename(self._get_dirpath())
        return exp_num

    def _get_name(self) -> str:
        return f"{self._get_num()}_{self.run_num}__{self.model_name}"
