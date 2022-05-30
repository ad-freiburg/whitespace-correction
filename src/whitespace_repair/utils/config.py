import os
import re
from typing import Any, Dict, List, Optional, Set, Union, cast

import yaml


class BaseConfig:
    required_arguments: Set[str] = set()

    @classmethod
    def _check_required(cls, d: Dict[str, Any]) -> None:
        if (d is None or len(d) == 0) and len(cls.required_arguments) > 0:
            raise ValueError(f"Got empty dictionary or None but found the required arguments. "
                             f"Please specify all the required arguments {cls.required_arguments}.")
        for arg in cls.required_arguments:
            if arg not in d:
                raise ValueError(f"Could not find required argument {arg} in {d.keys()}. "
                                 f"Please specify all the required arguments {cls.required_arguments}.")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseConfig":
        raise NotImplementedError()

    @classmethod
    def _rep_env_variables(cls, s: str) -> str:
        orig_s = s
        env_var_regex = re.compile(r"\${([A-Z_]+):?(.*?)}")

        length_change = 0
        for match in re.finditer(env_var_regex, s):
            env_var, env_default = match.groups()
            if env_var not in os.environ:
                if env_default == "":
                    raise ValueError(f"Environment variable {env_var} not found and no default was given")
                else:
                    env_var = env_default
            else:
                env_var = os.environ[env_var]
            lower_idx = match.start() + length_change
            upper_idx = match.end() + length_change
            s = s[:lower_idx] + env_var + s[upper_idx:]
            length_change = len(s) - len(orig_s)
        return s

    @classmethod
    def from_yaml(cls, filepath: str) -> "BaseConfig":
        with open(filepath, "r", encoding="utf8") as f:
            raw_yaml = f.read()
        raw_yaml_with_env_vars = cls._rep_env_variables(raw_yaml)
        parsed_yaml = yaml.load(raw_yaml_with_env_vars, Loader=yaml.FullLoader)
        config = cls.from_dict(parsed_yaml)
        return config

    def get_parsed_vars(self) -> Dict[str, Any]:
        var = vars(self)
        parsed_var: Dict[str, Any] = {}
        for k, v in sorted(var.items()):
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], BaseConfig):
                parsed_var[k] = [v_i.get_parsed_vars() for v_i in v]
            elif isinstance(v, BaseConfig):
                parsed_var[k] = v.get_parsed_vars()
            else:
                parsed_var[k] = v
        return parsed_var

    def __repr__(self) -> str:
        parsed_var = self.get_parsed_vars()
        return yaml.dump(parsed_var, default_flow_style=False)


class OptimizerConfig(BaseConfig):
    required_arguments = {"type", "learning_rate"}

    def __init__(self,
                 type: str,
                 learning_rate: float,
                 weight_decay: float):
        self.type: str = type
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        cls._check_required(d)
        config = OptimizerConfig(
            type=d["type"],
            learning_rate=d["learning_rate"],
            weight_decay=d.get("weight_decay", 0.0))
        return config


class LossConfig(BaseConfig):
    required_arguments = {"type"}

    def __init__(self,
                 type: str,
                 arguments: Dict[str, Any]):
        self.type: str = type
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LossConfig":
        cls._check_required(d)
        config = LossConfig(
            type=d["type"],
            arguments=d.get("arguments", {}))
        return config


class LRSchedulerConfig(BaseConfig):
    required_arguments = {"type", "arguments"}

    def __init__(self,
                 type: str,
                 arguments: Dict[str, Any]):
        self.type: str = type
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LRSchedulerConfig":
        cls._check_required(d)
        config = LRSchedulerConfig(
            type=d["type"],
            arguments=d["arguments"])
        return config


class TrainConfig(BaseConfig):
    required_arguments = {"train_data", "optimizer", "loss"}

    def __init__(self,
                 num_epochs: int,
                 train_data: str,
                 in_memory: bool,
                 num_workers: int,
                 batch_size: int,
                 batch_max_tokens: int,
                 min_seq_length: int,
                 max_seq_length: int,
                 optimizer: OptimizerConfig,
                 lr_scheduler: Optional[LRSchedulerConfig],
                 loss: LossConfig,
                 swap_inputs_and_targets: bool,
                 mixed_precision: bool,
                 eval_interval: int,
                 log_interval: int,
                 keep_last_n_checkpoints: int):
        self.num_epochs: int = num_epochs
        self.swap_inputs_and_targets: bool = swap_inputs_and_targets
        self.loss: LossConfig = loss
        self.optimizer: OptimizerConfig = optimizer
        self.lr_scheduler: Optional[LRSchedulerConfig] = lr_scheduler
        self.mixed_precision: bool = mixed_precision
        self.min_seq_length: int = min_seq_length
        self.max_seq_length: int = max_seq_length
        self.eval_interval: int = eval_interval
        self.log_interval: int = log_interval
        self.keep_last_n_checkpoints: int = keep_last_n_checkpoints
        self.in_memory: bool = in_memory
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size
        self.batch_max_tokens: int = batch_max_tokens
        self.train_data: str = train_data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        cls._check_required(d)

        optimizer_config = OptimizerConfig.from_dict(d["optimizer"])
        loss_config = LossConfig.from_dict(d["loss"])
        lr_scheduler_config = LRSchedulerConfig.from_dict(d["lr_scheduler"]) \
            if d.get("lr_scheduler", None) is not None else None

        config = TrainConfig(
            num_epochs=d.get("num_epochs", 5),
            train_data=d["train_data"],
            in_memory=d.get("in_memory", False),
            num_workers=d.get("num_workers", 4),
            batch_size=d.get("batch_size", None),
            batch_max_tokens=d.get("batch_max_tokens", None),
            min_seq_length=d.get("min_seq_length", 0),
            max_seq_length=d.get("max_seq_length", 512),
            optimizer=optimizer_config,
            lr_scheduler=lr_scheduler_config,
            loss=loss_config,
            swap_inputs_and_targets=d.get("swap_inputs_and_targets", False),
            mixed_precision=d.get("mixed_precision", False),
            eval_interval=d.get("eval_interval", 10000),
            log_interval=d.get("log_interval", 500),
            keep_last_n_checkpoints=d.get("keep_last_n_checkpoints", -1)
        )
        return config


class MetricConfig(BaseConfig):
    required_arguments = {"name"}

    def __init__(self,
                 name: str,
                 arguments: Dict[str, Any]):
        self.name: str = name
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricConfig":
        cls._check_required(d)
        config = MetricConfig(name=d["name"],
                              arguments=d.get("arguments", {}))
        return config


class ValConfig(BaseConfig):
    required_arguments = {"val_data"}

    def __init__(self,
                 val_data: Union[float, str, int],
                 text_metrics: List[MetricConfig]):
        self.val_data: Union[float, int, str] = val_data
        self.text_metrics: List[MetricConfig] = text_metrics

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValConfig":
        cls._check_required(d)
        if d.get("text_metrics", None) is not None:
            text_metric_configs = [MetricConfig.from_dict(cfg) for cfg in d["text_metrics"]]
        else:
            text_metric_configs = []

        config = ValConfig(val_data=d["val_data"],
                           text_metrics=text_metric_configs)
        return config


class EncoderDecoderConfig(BaseConfig):
    required_arguments = {"tokenizer"}

    def __init__(self,
                 type: str,
                 arguments: Dict[str, Any],
                 fixed: bool,
                 pretrained: str,
                 tokenizer: str,
                 max_num_embeddings: int,
                 embedding_dim: int,
                 learned_positional_embeddings: bool,
                 norm_embeddings: bool,
                 model_dim: int,
                 feedforward_dim: int,
                 attention_heads: int,
                 dropout: float,
                 num_layers: int,
                 share_parameters: bool,
                 activation: str):
        self.type: str = type
        self.arguments: Dict[str, Any] = arguments
        self.fixed: bool = fixed
        self.pretrained: str = pretrained
        self.tokenizer: str = tokenizer
        self.max_num_embeddings: int = max_num_embeddings
        self.embedding_dim: int = embedding_dim
        self.learned_positional_embeddings: bool = learned_positional_embeddings
        self.norm_embeddings: bool = norm_embeddings
        self.model_dim: int = model_dim
        self.feedforward_dim: int = feedforward_dim
        self.attention_heads: int = attention_heads
        self.dropout: float = dropout
        self.num_layers: int = num_layers
        self.share_parameters: bool = share_parameters
        self.activation: str = activation

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EncoderDecoderConfig":
        cls._check_required(d)
        config = EncoderDecoderConfig(type=d.get("type", "default"),
                                      arguments=d.get("arguments", {}),
                                      fixed=d.get("fixed", False),
                                      pretrained=d.get("pretrained", None),
                                      tokenizer=d["tokenizer"],
                                      max_num_embeddings=d.get("max_num_embeddings", 1024),
                                      embedding_dim=d.get("embedding_dim", 768),
                                      learned_positional_embeddings=d.get("learned_positional_embeddings", False),
                                      norm_embeddings=d.get("norm_embeddings", False),
                                      model_dim=d.get("model_dim", 768),
                                      feedforward_dim=d.get("feedforward_dim", 3072),
                                      attention_heads=d.get("attention_heads", 12),
                                      dropout=d.get("dropout", 0.1),
                                      num_layers=d.get("num_layers", 12),
                                      share_parameters=d.get("share_parameters", False),
                                      activation=d.get("activation", "gelu"))
        return config


class HeadConfig(BaseConfig):
    required_arguments = {"type"}

    def __init__(self,
                 type: str,
                 arguments: Dict[str, Any]):
        self.type: str = type
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeadConfig":
        cls._check_required(d)
        return HeadConfig(type=d["type"],
                          arguments=d.get("arguments", {}))


class ModelConfig(BaseConfig):
    required_arguments = {"name", "type"}

    def __init__(self,
                 name: str,
                 type: str,
                 pretrained: str,
                 share_encoder_decoder_embeddings: bool,
                 share_decoder_input_output_embeddings: bool,
                 encoder: Optional[EncoderDecoderConfig],
                 decoder: Optional[EncoderDecoderConfig],
                 head: Optional[HeadConfig]):
        self.decoder: Optional[EncoderDecoderConfig] = decoder
        self.encoder: Optional[EncoderDecoderConfig] = encoder
        self.head: Optional[HeadConfig] = head
        self.share_decoder_input_output_embeddings: bool = share_decoder_input_output_embeddings
        self.share_encoder_decoder_embeddings: bool = share_encoder_decoder_embeddings
        self.pretrained: str = pretrained
        self.name: str = name
        self.type: str = type

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        cls._check_required(d)

        encoder_config = EncoderDecoderConfig.from_dict(d["encoder"]) if d.get("encoder", None) is not None else None
        decoder_config = EncoderDecoderConfig.from_dict(d["decoder"]) if d.get("decoder", None) is not None else None
        head_config = HeadConfig.from_dict(d["head"]) if d.get("head", None) is not None else None

        config = ModelConfig(name=d["name"],
                             type=d["type"],
                             pretrained=d.get("pretrained", None),
                             share_encoder_decoder_embeddings=d.get("share_encoder_decoder_embeddings", False),
                             share_decoder_input_output_embeddings=d.get("share_decoder_input_output_embeddings",
                                                                         False),
                             encoder=encoder_config,
                             decoder=decoder_config,
                             head=head_config)
        return config


class Config(BaseConfig):
    required_arguments = {"experiment", "train", "val", "model"}

    def __init__(self,
                 experiment: str,
                 experiment_dir: str,
                 seed: int,
                 train: TrainConfig,
                 val: ValConfig,
                 model: ModelConfig):
        self.model: ModelConfig = model
        self.val: ValConfig = val
        self.train: TrainConfig = train
        self.seed: int = seed
        self.experiment: str = experiment
        self.experiment_dir: str = experiment_dir

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        # for mypy type checking
        return cast(Config, super().from_yaml(filepath))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cls._check_required(d)
        train_config = TrainConfig.from_dict(d["train"])
        val_config = ValConfig.from_dict(d["val"])
        model_config = ModelConfig.from_dict(d["model"])
        config = Config(experiment=d["experiment"],
                        experiment_dir=d.get("experiment_dir", "experiments"),
                        seed=d.get("seed", None),
                        train=train_config,
                        val=val_config,
                        model=model_config)
        return config


class PreprocessingConfig(BaseConfig):
    def __init__(self,
                 type: str,
                 arguments: Dict[str, Any]):
        self.type: str = type
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessingConfig":
        cls._check_required(d)
        if d is None:
            return None

        config = PreprocessingConfig(type=d.get("type", None),
                                     arguments=d.get("arguments", {}))
        return config


class DataPreprocessingConfig(BaseConfig):
    required_arguments = {"data", "output_dir", "tokenizer"}

    def __init__(self,
                 data: List[str],
                 seed: int,
                 output_dir: str,
                 tokenizer: str,
                 target_tokenizer: str,
                 pretokenize: bool,
                 ensure_equal_length: bool,
                 preprocessing: List[PreprocessingConfig],
                 lmdb_name: str,
                 max_sequences: int,
                 max_sequence_length: int,
                 cut_overflowing: bool):
        self.data: List[str] = data
        self.seed: int = seed
        self.output_dir: str = output_dir
        self.tokenizer: str = tokenizer
        self.target_tokenizer: str = target_tokenizer
        self.pretokenize: bool = pretokenize
        self.ensure_equal_length: bool = ensure_equal_length
        self.preprocessing: List[PreprocessingConfig] = preprocessing
        self.lmdb_name: str = lmdb_name
        self.max_sequences: int = max_sequences
        self.max_sequence_length: int = max_sequence_length
        self.cut_overflowing: bool = cut_overflowing

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataPreprocessingConfig":
        cls._check_required(d)
        if "preprocessing" in d:
            preprocessing_config = [PreprocessingConfig.from_dict(cfg) for cfg in d["preprocessing"]]
        else:
            preprocessing_config = []
        config = DataPreprocessingConfig(data=d["data"],
                                         seed=d.get("seed", None),
                                         output_dir=d["output_dir"],
                                         tokenizer=d["tokenizer"],
                                         target_tokenizer=d.get("target_tokenizer", None),
                                         pretokenize=d.get("pretokenize", False),
                                         ensure_equal_length=d.get("ensure_equal_length", False),
                                         preprocessing=preprocessing_config,
                                         lmdb_name=d.get("lmdb_name", "lmdb"),
                                         max_sequences=d.get("max_sequences", None),
                                         max_sequence_length=d.get("max_sequence_length", None),
                                         cut_overflowing=d.get("cut_overflowing", False))
        return config
