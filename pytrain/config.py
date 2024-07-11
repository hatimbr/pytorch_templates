from argparse import ArgumentParser
from configparser import ConfigParser
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(kw_only=True)
class Config:
    config_name: str = "DEFAULT"
    config_file: Path = field(
        default=Path.cwd() / "config.ini", metadata={"converter": Path, "export": False}
    )

    def __post_init__(self) -> None:
        self.fields_names = [field.name for field in fields(self)]

        args = self.get_args()
        if args.config_file is not None:
            self.config_file = args.config_file

        self.from_file(Path(self.config_file))
        self.from_args(args)
        self.correct_type()

    def from_file(self, config_path) -> None:
        confparser = ConfigParser()
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            print("Using default values and command line arguments only.")
            return

        confparser.read(config_path)
        for key, val in confparser[self.config_name].items():
            if key in self.fields_names:
                setattr(self, key, val)
            else:
                print(f"Unknown key from config file, ignoring: {key}")

    def get_args(self) -> ArgumentParser:
        argparser = ArgumentParser()
        for dataclass_field in fields(self):
            argparser.add_argument(
                f"--{dataclass_field.name}",
                action="store_true" if dataclass_field.type == bool else "store",
                default=None,
            )
        args, _ = argparser.parse_known_args()
        return args

    def from_args(self, args) -> None:
        for key, val in vars(args).items():
            if val is not None and key in self.fields_names:
                setattr(self, key, val)

    def correct_type(self) -> None:
        # Convert the values of the config attributes
        for dataclass_field in fields(self):
            converter = dataclass_field.metadata.get("converter", None)
            if converter is not None:
                value = getattr(self, dataclass_field.name)
                if value is not None:
                    self.__setattr__(dataclass_field.name, converter(value))

    # def export(self) -> dict:
    #     config_dict = {}
    #     for dataclass_field in fields(self):
    #         if dataclass_field.metadata.get("export"):
    #             if dataclass_field.type != Config:
    #                 config_dict[dataclass_field.name] = getattr(
    #                     self, dataclass_field.name
    #                 )
    #             else:
    #     return config_dict


OPTIM_PARAMS = {
    "adamw": ["lr", "beta1", "beta2", "eps", "weight_decay"],
    "sgd": ["lr", "momentum", "weight_decay"],
}


@dataclass(kw_only=True)
class OptimizerConfig(Config):
    config_name: str = "OPTIMIZER"

    optimizer_name: str = field(
        default="adamw",
        metadata={"converter": str, "export": True, "optim_params": False}
    )
    lr: float | None = field(
        default=1e-05,
        metadata={"converter": float, "export": True, "optim_params": True}
    )
    beta1: float | None = field(
        default=0.9,
        metadata={"converter": float, "export": True, "optim_params": True}
    )
    beta2: float | None = field(
        default=0.999,
        metadata={"converter": float, "export": True, "optim_params": True})
    eps: float | None = field(
        default=1e-08,
        metadata={"converter": float, "export": True, "optim_params": True}
    )
    weight_decay: float | None = field(
        default=0.01,
        metadata={"converter": float, "export": True, "optim_params": True}
    )
    momentum: float | None = field(
        default=0.0,
        metadata={"converter": float, "export": True, "optim_params": True}
    )
    lr_scheduler_name: str | None = field(
        default=None, metadata={"converter": str, "export": True, "optim_params": False}
    )
    num_warmup_steps: int | None = field(
        default=100, metadata={"converter": int, "export": True, "optim_params": False}
    )

    def default_to_none(self) -> None:
        """Set unused optimizer parameters to None."""
        if self.optimizer_name not in OPTIM_PARAMS:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        for dataclass_field in fields(self):
            if (
                dataclass_field.metadata.get("optim_params")
                and dataclass_field.name not in OPTIM_PARAMS[self.optimizer_name]
            ):
                self.__setattr__(dataclass_field.name, None)

        if self.lr_scheduler_name is None:
            self.num_warmup_steps = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.default_to_none()


@dataclass(kw_only=True)
class GlobalConfig(Config):
    config_name: str = "GLOBAL"

    model_dir: Path = field(
        default=Path.cwd() / "model", metadata={"converter": Path, "export": False}
    )
    data_dir: Path = field(
        default=Path.cwd() / "data", metadata={"converter": Path, "export": False}
    )

    epochs: int = field(default=1, metadata={"converter": int, "export": True})
    model_name: str = field(
        default="model", metadata={"converter": str, "export": True}
    )

    track: bool = field(default=False, metadata={"converter": bool, "export": False})
    dev_test: bool = field(default=False, metadata={"converter": bool, "export": False})

    optimizer_config: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(), metadata={"export": True}
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name
