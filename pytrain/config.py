from argparse import ArgumentParser
from configparser import ConfigParser
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(kw_only=True)
class Config:
    config_name: str = "DEFAULT"
    config_file: Path = field(
        default=Path.cwd() / "config.ini", metadata={"converter": Path, "track": False}
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


@dataclass(kw_only=True)
class OptimizerConfig(Config):
    config_name: str = "OPTIMIZER"

    optimizer_name: str = field(
        default="adamw", metadata={"converter": str, "track": True}
    )
    lr: float = field(default=1e-05, metadata={"converter": float, "track": True})
    beta1: float = field(default=0.9, metadata={"converter": float, "track": True})
    beta2: float = field(default=0.999, metadata={"converter": float, "track": True})
    eps: float = field(default=1e-08, metadata={"converter": float, "track": True})
    weight_decay: float = field(
        default=0.01, metadata={"converter": float, "track": True}
    )
    momentum: float = field(default=0.0, metadata={"converter": float, "track": True})
    lr_scheduler: str | None = field(
        default=None, metadata={"converter": str, "track": True}
    )


@dataclass(kw_only=True)
class GlobalConfig(Config):
    config_name: str = "GLOBAL"

    model_dir: Path = field(
        default=Path.cwd() / "model", metadata={"converter": Path, "track": False}
    )
    data_dir: Path = field(
        default=Path.cwd() / "data", metadata={"converter": Path, "track": False}
    )

    epochs: int = field(default=1, metadata={"converter": int, "track": True})
    model_name: str = field(default="model", metadata={"converter": str, "track": True})

    track: bool = field(default=False, metadata={"converter": bool, "track": False})
    dev_test: bool = field(default=False, metadata={"converter": bool, "track": False})

    optimizer_config: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(), metadata={"track": True}
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name
