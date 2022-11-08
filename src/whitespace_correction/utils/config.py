import os
import re
from typing import Any, Dict, List, Optional, Set, Union, cast, Tuple

import yaml

from whitespace_correction.utils import constants


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


class TokenizerConfig(BaseConfig):
    required_arguments = "name"

    def __init__(
            self,
            name: str,
            default_prefix_tokens: Tuple[str, ...],
            default_suffix_tokens: Tuple[str, ...],
            additional_tokens: Optional[List[str]]
    ) -> None:
        self.name = name
        self.default_prefix_tokens = default_prefix_tokens
        self.default_suffix_tokens = default_suffix_tokens
        self.additional_tokens = additional_tokens

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizerConfig":
        cls._check_required(d)
        config = TokenizerConfig(
            name=d["name"],
            default_prefix_tokens=d.get("default_prefix_tokens", (constants.BOS,)),
            default_suffix_tokens=d.get("default_suffix_tokens", (constants.EOS,)),
            additional_tokens=d.get("additional_tokens")
        )
        return config


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
            weight_decay=d.get("weight_decay", 0.0)
        )
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
                 batch_size: Optional[int],
                 batch_max_tokens: Optional[int],
                 min_seq_length: int,
                 max_seq_length: int,
                 optimizer: OptimizerConfig,
                 lr_scheduler: Optional[LRSchedulerConfig],
                 loss: LossConfig,
                 swap_inputs_and_targets: bool,
                 mixed_precision: bool,
                 eval_interval: Union[int, float],
                 log_interval: Union[int, float],
                 keep_last_n_checkpoints: int):
        self.num_epochs: int = num_epochs
        self.swap_inputs_and_targets: bool = swap_inputs_and_targets
        self.loss: LossConfig = loss
        self.optimizer: OptimizerConfig = optimizer
        self.lr_scheduler: Optional[LRSchedulerConfig] = lr_scheduler
        self.mixed_precision: bool = mixed_precision
        self.min_seq_length: int = min_seq_length
        self.max_seq_length: int = max_seq_length
        self.eval_interval: Union[int, float] = eval_interval
        self.log_interval: Union[int, float] = log_interval
        self.keep_last_n_checkpoints: int = keep_last_n_checkpoints
        self.in_memory: bool = in_memory
        self.num_workers: int = num_workers
        self.batch_size = batch_size
        self.batch_max_tokens = batch_max_tokens
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
            eval_interval=d.get("eval_interval", 1.0),
            log_interval=d.get("log_interval", 0.01),
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
        config = MetricConfig(name=d["name"], arguments=d.get("arguments", {}))
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
        return HeadConfig(type=d["type"], arguments=d.get("arguments", {}))


class TransformerEncoderDecoderConfig(BaseConfig):
    required_arguments = {"tokenizer"}

    def __init__(self,
                 fixed: bool,
                 pretrained: str,
                 tokenizer: TokenizerConfig,
                 max_num_embeddings: int,
                 embedding_dim: int,
                 positional_embeddings: Optional[str],
                 group_name: Optional[str],
                 group_at: str,
                 group_aggregation: str,
                 norm_embeddings: bool,
                 model_dim: int,
                 feedforward_dim: int,
                 attention_heads: int,
                 dropout: float,
                 num_layers: int,
                 share_parameters: bool,
                 activation: str):
        self.fixed: bool = fixed
        self.pretrained: str = pretrained
        self.tokenizer = tokenizer
        self.max_num_embeddings: int = max_num_embeddings
        self.embedding_dim: int = embedding_dim
        self.positional_embeddings: Optional[str] = positional_embeddings
        self.group_name: Optional[str] = group_name
        self.group_at: str = group_at
        self.group_aggregation: str = group_aggregation
        self.norm_embeddings: bool = norm_embeddings
        self.model_dim: int = model_dim
        self.feedforward_dim: int = feedforward_dim
        self.attention_heads: int = attention_heads
        self.dropout: float = dropout
        self.num_layers: int = num_layers
        self.share_parameters: bool = share_parameters
        self.activation: str = activation

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformerEncoderDecoderConfig":
        cls._check_required(d)
        config = TransformerEncoderDecoderConfig(
            fixed=d.get("fixed", False),
            pretrained=d.get("pretrained", None),
            tokenizer=TokenizerConfig.from_dict(d["tokenizer"]),
            max_num_embeddings=d.get("max_num_embeddings", 1024),
            embedding_dim=d.get("embedding_dim", 768),
            positional_embeddings=d.get("positional_embeddings"),
            group_name=d.get("group_name"),
            group_aggregation=d.get("group_aggregation", "mean"),
            group_at=d.get("group_at", "after"),
            norm_embeddings=d.get("norm_embeddings", False),
            model_dim=d.get("model_dim", 768),
            feedforward_dim=d.get("feedforward_dim", 3072),
            attention_heads=d.get("attention_heads", 12),
            dropout=d.get("dropout", 0.1),
            num_layers=d.get("num_layers", 12),
            share_parameters=d.get("share_parameters", False),
            activation=d.get("activation", "gelu")
        )
        return config


class TransformerModelConfig(BaseConfig):
    required_arguments = {"name", "type"}

    def __init__(self,
                 name: str,
                 type: str,
                 pretrained: str,
                 share_encoder_decoder_embeddings: bool,
                 share_decoder_input_output_embeddings: bool,
                 encoder: Optional[TransformerEncoderDecoderConfig],
                 decoder: Optional[TransformerEncoderDecoderConfig],
                 head: Optional[HeadConfig]):
        self.decoder: Optional[TransformerEncoderDecoderConfig] = decoder
        self.encoder: Optional[TransformerEncoderDecoderConfig] = encoder
        self.head: Optional[HeadConfig] = head
        self.share_decoder_input_output_embeddings: bool = share_decoder_input_output_embeddings
        self.share_encoder_decoder_embeddings: bool = share_encoder_decoder_embeddings
        self.pretrained: str = pretrained
        self.name: str = name
        self.type: str = type

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        cls._check_required(d)

        encoder_config = TransformerEncoderDecoderConfig.from_dict(
            d["encoder"]
        ) if d.get("encoder", None) is not None else None
        decoder_config = TransformerEncoderDecoderConfig.from_dict(
            d["decoder"]
        ) if d.get("decoder", None) is not None else None
        head_config = HeadConfig.from_dict(d["head"]) if d.get("head", None) is not None else None

        config = TransformerModelConfig(
            name=d["name"],
            type=d["type"],
            pretrained=d.get("pretrained", None),
            share_encoder_decoder_embeddings=d.get("share_encoder_decoder_embeddings", False),
            share_decoder_input_output_embeddings=d.get("share_decoder_input_output_embeddings", False),
            encoder=encoder_config,
            decoder=decoder_config,
            head=head_config
        )
        return config


class RNNEncoderDecoderConfig(BaseConfig):
    required_arguments = {"tokenizer"}

    def __init__(self,
                 type: str,
                 bidirectional: bool,
                 fixed: bool,
                 pretrained: str,
                 tokenizer: TokenizerConfig,
                 embedding_dim: int,
                 group_name: Optional[str],
                 group_at: str,
                 group_aggregation: str,
                 norm_embeddings: bool,
                 model_dim: int,
                 dropout: float,
                 num_layers: int):
        self.type: str = type
        self.bidirectional: bool = bidirectional
        self.fixed: bool = fixed
        self.pretrained: str = pretrained
        self.tokenizer = tokenizer
        self.embedding_dim: int = embedding_dim
        self.norm_embeddings: bool = norm_embeddings
        self.group_name: Optional[str] = group_name
        self.group_at: str = group_at
        self.group_aggregation: str = group_aggregation
        self.model_dim: int = model_dim
        self.dropout: float = dropout
        self.num_layers: int = num_layers

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RNNEncoderDecoderConfig":
        cls._check_required(d)
        config = RNNEncoderDecoderConfig(
            type=d.get("type", "lstm"),
            bidirectional=d.get("bidirectional", False),
            fixed=d.get("fixed", False),
            pretrained=d.get("pretrained", None),
            tokenizer=TokenizerConfig.from_dict(d["tokenizer"]),
            embedding_dim=d.get("embedding_dim", 768),
            group_name=d.get("group_name"),
            group_aggregation=d.get("group_aggregation", "mean"),
            group_at=d.get("group_at", "after"),
            norm_embeddings=d.get("norm_embeddings", False),
            model_dim=d.get("model_dim", 768),
            dropout=d.get("dropout", 0.1),
            num_layers=d.get("num_layers", 12)
        )
        return config


class RNNModelConfig(BaseConfig):
    required_arguments = {"name", "type"}

    def __init__(self,
                 name: str,
                 type: str,
                 pretrained: str,
                 share_encoder_decoder_embeddings: bool,
                 share_decoder_input_output_embeddings: bool,
                 encoder: Optional[RNNEncoderDecoderConfig],
                 decoder: Optional[RNNEncoderDecoderConfig],
                 head: Optional[HeadConfig]):
        self.decoder: Optional[RNNEncoderDecoderConfig] = decoder
        self.encoder: Optional[RNNEncoderDecoderConfig] = encoder
        self.head: Optional[HeadConfig] = head
        self.share_decoder_input_output_embeddings: bool = share_decoder_input_output_embeddings
        self.share_encoder_decoder_embeddings: bool = share_encoder_decoder_embeddings
        self.pretrained: str = pretrained
        self.name: str = name
        self.type: str = type

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        cls._check_required(d)

        encoder_config = RNNEncoderDecoderConfig.from_dict(
            d["encoder"]
        ) if d.get("encoder", None) is not None else None
        decoder_config = RNNEncoderDecoderConfig.from_dict(
            d["decoder"]
        ) if d.get("decoder", None) is not None else None
        head_config = HeadConfig.from_dict(d["head"]) if d.get("head", None) is not None else None

        config = RNNModelConfig(
            name=d["name"],
            type=d["type"],
            pretrained=d.get("pretrained", None),
            share_encoder_decoder_embeddings=d.get("share_encoder_decoder_embeddings", False),
            share_decoder_input_output_embeddings=d.get("share_decoder_input_output_embeddings", False),
            encoder=encoder_config,
            decoder=decoder_config,
            head=head_config
        )
        return config


EncoderDecoderConfig = Union[TransformerEncoderDecoderConfig, RNNEncoderDecoderConfig]
ModelConfig = Union[TransformerModelConfig, RNNModelConfig]


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
        assert "type" in d["model"], "expected model config to have a field named type"
        if d["model"]["type"].startswith("transformer"):
            model_config = TransformerModelConfig.from_dict(d["model"])
        else:
            model_config = RNNModelConfig.from_dict(d["model"])
        config = Config(
            experiment=d["experiment"],
            experiment_dir=d.get("experiment_dir", "experiments"),
            seed=d.get("seed", 22),
            train=train_config,
            val=val_config,
            model=model_config
        )
        return config


class PreprocessingConfig(BaseConfig):
    required_arguments = {"type"}

    def __init__(self,
                 type: str,
                 arguments: Dict[str, Any]):
        self.type: str = type
        self.arguments: Dict[str, Any] = arguments

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessingConfig":
        cls._check_required(d)
        config = PreprocessingConfig(
            type=d["type"],
            arguments=d.get("arguments", {})
        )
        return config


class DataPreprocessingConfig(BaseConfig):
    required_arguments = {"data", "output_dir", "tokenizer"}

    def __init__(self,
                 data: List[str],
                 language_codes: Optional[List[str]],
                 seed: int,
                 output_dir: str,
                 tokenizer: TokenizerConfig,
                 target_tokenizer: Optional[TokenizerConfig],
                 preprocessing: List[PreprocessingConfig],
                 lmdb_name: str,
                 max_sequences: Optional[int],
                 max_sequence_length: Optional[int]):
        self.data: List[str] = data
        self.language_codes = language_codes
        self.seed: int = seed
        self.output_dir: str = output_dir
        self.tokenizer = tokenizer
        self.target_tokenizer = target_tokenizer
        self.preprocessing: List[PreprocessingConfig] = preprocessing
        self.lmdb_name: str = lmdb_name
        self.max_sequences = max_sequences
        self.max_sequence_length = max_sequence_length

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataPreprocessingConfig":
        cls._check_required(d)
        if "preprocessing" in d:
            preprocessing_config = [PreprocessingConfig.from_dict(cfg) for cfg in d["preprocessing"]]
        else:
            preprocessing_config = []
        config = DataPreprocessingConfig(
            data=d["data"],
            seed=d.get("seed", 22),
            output_dir=d["output_dir"],
            tokenizer=TokenizerConfig.from_dict(d["tokenizer"]),
            target_tokenizer=TokenizerConfig.from_dict(d["target_tokenizer"]) if "target_tokenizer" in d else None,
            preprocessing=preprocessing_config,
            lmdb_name=d.get("lmdb_name", "lmdb"),
            max_sequences=d.get("max_sequences", None),
            max_sequence_length=d.get("max_sequence_length", None),
            language_codes=d.get("language_codes", None)
        )
        return config
