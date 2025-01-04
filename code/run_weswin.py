import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import json

import pandas as pd
import datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version

from data_util.dataset import load_custom_dataset, concatenate_datasets
from data_util.tokenization import preprocess_token_classification
from data_util.collation import DataCollatorForTokenClassification

from eval_util.evaluation import compute_metrics_token_classification
from eval_util.prediction import PredictionFactory, get_doc_grouped_df

from run_util.util import get_sent_sep_token, add_new_special_token, freeze_parameters, count_parameters, CustomEarlyStoppingCallback, get_resume_from_checkpoint, load_best_model, get_preds_dir_name, get_cache_file_name


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.35.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    add_prefix_space: bool = field(
        default=None, metadata={"help": "Whether to enable add_prefix_space (if suitable for the model) to the tokenizer."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    use_legacy_module_names: bool = field(default=False, metadata={"help": "Whether to use legacy module names."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_log_name: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    predict_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input predict data file to predict on (a csv or JSON file)."},
    )
    doc_id_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of `doc_id` to input in the file (a csv or JSON file)."}
    )
    id_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of `id` to input in the file (a csv or JSON file)."}
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of `text` to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of `label` to input in the file (a csv or JSON file)."}
    )
    drop_titles_probability: Optional[float] = field(
        default=None, metadata={"help": "Probability of keeping a title among sentences (between 0 and 1)"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "The number of evaluations to wait before early stopping."}
    )
    early_stopping_threshold: Optional[float] = field(
        default=None, metadata={"help": "The threshold to trigger early stopping."}
    )
    
    partition_strategy: Optional[str] = field(default=None, metadata={"help": "The partition strategy to use for the dataset."})
    partition_value: Optional[float] = field(default=None, metadata={"help": "The partition value to use alongside partition_strategy for the dataset."})
    partition_disable_strategy: Optional[str] = field(default=None, metadata={"help": "The strategy to disable sequence boundries."})
    partition_disable_value: Optional[int] = field(default=None, metadata={"help": "Number of sentences to disable from the end of sequences in partitioning."})

    other_dataset_log_name: Optional[str] = field(default=None)
    other_dataset_name: Optional[str] = field(default=None)
    other_train_file: Optional[str] = field(default=None)
    other_eval_file: Optional[str] = field(default=None)
    other_predict_file: Optional[str] = field(default=None)
    other_max_train_samples: Optional[int] = field(default=None)
    other_max_eval_samples: Optional[int] = field(default=None)
    other_max_predict_samples: Optional[int] = field(default=None)
    
    other_drop_titles_probability: Optional[float] = field(default=None)
    other_overwrite_cache: bool = field(default=False)
    other_preprocessing_num_workers: Optional[int] = field(default=None)
    
    # Result collection
    out_results_path: Optional[str] = field(default=None, metadata={"help": "The path to save the inference results as a row in a csv file."})
    expr_name: Optional[str] = field(default=None, metadata={"help": "The name of the experiment."})
    expr_threshold: Optional[str] = field(default=None, metadata={"help": "The threshold to use for the experiment."})
    expr_hpm_path: Optional[str] = field(default=None, metadata={"help": "The path to save the best hyperparameters of the experiment."})
    expr_metric: Optional[str] = field(default=None, metadata={"help": "The metric to use for the experiment."})
    
    other_out_results_path: Optional[str] = field(default=None, metadata={"help": "The path to save the inference results as a row in a csv file."})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.eval_file is None and self.predict_file is None:
            raise ValueError("Need either a dataset name or a training/eval/predict file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
            if self.eval_file is not None:
                extension = self.eval_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`eval_file` should be a csv or a json file."
            if self.predict_file is not None:
                extension = self.predict_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`predict_file` should be a csv or a json file."

        self.partition_strategy = self.partition_strategy or 'ex'
        if self.partition_strategy in ['ex', 'ss', 'si']:
            self.partition_value = int(self.partition_value or 0)

        self.partition_disable_strategy, self.partition_disable_value = self.partition_disable_strategy or 'na', self.partition_disable_value or 0
        # assert self.partition_disable_value >= 0
        
        if self.expr_name is not None:
            assert set(self.expr_name).issubset(set('trw'))
            if self.partition_strategy == 'ex':
                assert 'r' not in self.expr_name and 'w' not in self.expr_name
        if self.expr_threshold is not None:
            try:
                self.expr_threshold = float(self.expr_threshold)
                assert 0 < self.expr_threshold < 1
            except:
                et = self.expr_threshold.split(':')
                if len(et) == 2:
                    t1, t2 = et
                    step = 0.01
                elif len(et) == 3:
                    t1, t2, step = et
                else:
                    raise ValueError("Invalid threshold value.")
                self.expr_threshold = (float(t1), float(t2), float(step))
                assert 0 < self.expr_threshold[0] < self.expr_threshold[1] < 1
                assert 0 < self.expr_threshold[2] < self.expr_threshold[1] - self.expr_threshold[0] < 1
    
        if self.expr_hpm_path is not None:
            assert os.path.isfile(self.expr_hpm_path) and self.expr_hpm_path.endswith('.pkl'), "Invalid path to the hyperparameter file."
        
        assert self.drop_titles_probability is None or 0 <= self.drop_titles_probability <= 1
        
        assert self.out_results_path is None or self.out_results_path.endswith('.csv')
        assert self.other_out_results_path is None or self.other_out_results_path.endswith('.csv')


@dataclass
class TaskArguments:
    task_name: Optional[str] = field(default=None, metadata={"help": "The name of the task."})
    frozens: Optional[str] = field(default="", metadata={"help": "Comma-separated names of the modules to freeze."})

    num_logits: Optional[int] = field(default=None, metadata={"help": "The number of logits."})
    
    attention_window: Optional[int] = field(default=None, metadata={"help": "The attention window size specific to LongFormer model."})
    max_tokens_per_seq: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total input sequence length after tokenization. If set, sequences longer than this will be truncated."},
    )
        
    sent_sep: Optional[str] = field(default="sep", metadata={"help": "The sentence separator token."})

    def __post_init__(self):
        assert self.task_name is None or self.task_name in ["multi_seg"]

        assert self.num_logits is None or self.num_logits > 0
        assert self.max_tokens_per_seq is None or self.max_tokens_per_seq >= 0
        assert self.sent_sep in ["sep", "snt"]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TaskArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
        model_args, data_args, training_args, task_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()
    
    training_args.include_inputs_for_metrics = True

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}, "
        + f"Seed: {training_args.seed}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model_name_or_path = model_args.model_name_or_path

    # Load config
    config_name_or_path = model_args.config_name if model_args.config_name else model_name_or_path
    config = AutoConfig.from_pretrained(
        config_name_or_path,
        # finetuning_task=task_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load tokenizer
    add_prefix_space = (model_args.add_prefix_space and config.model_type in {"roberta", "longformer"}) if training_args.do_train else getattr(config, 'add_prefix_space', None)
    tokenizer_kwargs = {'add_prefix_space': True} if add_prefix_space else {} 
    
    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        **tokenizer_kwargs,
    )
    
    # Load datasets
    raw_datasets = load_custom_dataset(data_args.dataset_log_name, 
                                       data_args.dataset_name, data_args.train_file, data_args.eval_file, data_args.predict_file)
    column_names = raw_datasets["train"].column_names if training_args.do_train else raw_datasets["predict"].column_names
    features = raw_datasets["train"].features if training_args.do_train else raw_datasets["predict"].features

    doc_id_column_name = data_args.doc_id_column_name or 'id'
    id_column_name = data_args.id_column_name or 'ids'
    text_column_name = data_args.text_column_name or 'sentences'
    label_column_name = data_args.label_column_name or 'labels'

    # Set task-specific config arguments for all tasks if training
    if training_args.do_train:
        task_name = task_args.task_name  # Task_name should be chosen carefully: config.task_name may not equal task_args.task_name
        
        label_list = list(range(features[label_column_name].feature.num_classes))
        label_names = features[label_column_name].feature.names
        num_labels = len(label_list)
        
        config.num_labels = num_labels  # This is used by the model to set the size of classification head
        config.num_logits = task_args.num_logits or config.num_labels
        assert 0 < config.num_logits <= config.num_labels
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = dict(enumerate(label_list))

        config.task_name = task_name

        config.add_prefix_space = add_prefix_space

        if task_name in ["multi_seg"]:
            assert config.num_logits == config.num_labels, "Different number of logits (!= 2) is not supported"
            config.max_tokens_per_seq = task_args.max_tokens_per_seq or config.max_position_embeddings
            config.sent_sep_token = get_sent_sep_token(task_args.sent_sep, tokenizer)

        # Model specific configs
        if config.model_type == 'longformer' and task_args.attention_window:
            config.attention_window = task_args.attention_window

    else:  # predict_only
        task_name = config.task_name
        
        if not getattr(config, 'num_logits', None):
            config.num_logits = getattr(config, 'num_labels', None) or 2
                
    # Gather model kwargs
    model_kwargs = dict(
        pretrained_model_name_or_path=model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        local_files_only=not training_args.do_train,
    )

    compute_metrics = partial(compute_metrics_token_classification, n_logits=config.num_logits, threshold=None, ignore_last_sentence=False, segeval=False)

    if task_name == "multi_seg":
        model = AutoModelForTokenClassification.from_pretrained(**model_kwargs)

        if training_args.do_train:
            add_new_special_token(config.sent_sep_token, tokenizer, model)

        # Preprocess function
        preprocess_fn = {
            'function': preprocess_token_classification,
            'fn_kwargs': dict(
                tokenizer=tokenizer, sent_sep_token=config.sent_sep_token, 
                doc_id_column_name=doc_id_column_name, id_column_name=id_column_name, text_column_name=text_column_name, label_column_name=label_column_name, 
                layout='seqs_2d', model_type=config.model_type,
                drop_titles_probability = data_args.drop_titles_probability, 
                max_tokens_per_seq = config.max_tokens_per_seq,
                partition_strategy = data_args.partition_strategy, partition_value = data_args.partition_value, 
                partition_disable_strategy = data_args.partition_disable_strategy, partition_disable_value = data_args.partition_disable_value
            )
        }

        # Data collator
        pad_to_multiple_of = None
        if config.model_type == 'longformer':
            pad_to_multiple_of = config.attention_window if isinstance(config.attention_window, int) else config.attention_window[0]
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    else:
        raise ValueError(f'Task name {task_name} is not supported')

    any_other_dataset = data_args.other_dataset_log_name
    if any_other_dataset:
        other_raw_datasets = load_custom_dataset(data_args.other_dataset_log_name, data_args.other_dataset_name, 
                                                 data_args.other_train_file, data_args.other_eval_file, data_args.other_predict_file)
        other_column_names = other_raw_datasets["train"].column_names if training_args.do_train else other_raw_datasets["predict"].column_names

        other_preprocess_fn = {
            'function': preprocess_token_classification,
            'fn_kwargs': dict(
                tokenizer=tokenizer, sent_sep_token=config.sent_sep_token, 
                doc_id_column_name=doc_id_column_name, id_column_name=id_column_name, text_column_name=text_column_name, label_column_name=label_column_name, 
                layout='seqs_2d', model_type=config.model_type,
                drop_titles_probability = data_args.other_drop_titles_probability, 
                max_tokens_per_seq = config.max_tokens_per_seq,
                partition_strategy = data_args.partition_strategy, partition_value = data_args.partition_value,
                partition_disable_strategy = data_args.partition_disable_strategy, partition_disable_value = data_args.partition_disable_value                
            )
        }

    # Freeze if required
    if training_args.do_train and task_args.frozens:
        if task_args.frozens.startswith('~'):
            freeze_parameters(model, whitelist=task_args.frozens[1:].split(','))
        else:
            freeze_parameters(model, blacklist=task_args.frozens.split(','))

    print('model_params_info:', count_parameters(model))

    print('output_dir:', training_args.output_dir)
      
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        max_train_samples = min(len(train_dataset), data_args.max_train_samples) if data_args.max_train_samples is not None else len(train_dataset)
        if max_train_samples < len(train_dataset):
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))
            raw_datasets["train"] = train_dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                **preprocess_fn,
                cache_file_name = get_cache_file_name(
                    preprocess_fn['function'], 
                    **(preprocess_fn['fn_kwargs'] | {'dataset_name': data_args.dataset_log_name, 'max_train_samples': max_train_samples, 'seed': training_args.seed})
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )
            
        if any_other_dataset:
            other_train_dataset = other_raw_datasets["train"]
            other_max_train_samples = min(len(other_train_dataset), data_args.other_max_train_samples) if data_args.other_max_train_samples is not None else len(other_train_dataset)
            if other_max_train_samples < len(other_train_dataset):
                other_train_dataset = other_train_dataset.shuffle(seed=training_args.seed).select(range(other_max_train_samples))
                other_raw_datasets["train"] = other_train_dataset
            with training_args.main_process_first(desc="other_train dataset map pre-processing"):
                other_train_dataset = other_train_dataset.map(
                    **other_preprocess_fn,
                    cache_file_name = get_cache_file_name(
                        other_preprocess_fn['function'], 
                        **(other_preprocess_fn['fn_kwargs'] | {'dataset_name': data_args.other_dataset_log_name, 'max_train_samples': other_max_train_samples, 'seed': training_args.seed})
                    ),
                    batched=True,
                    num_proc=data_args.other_preprocessing_num_workers,
                    load_from_cache_file=not data_args.other_overwrite_cache,
                    remove_columns=other_column_names,
                    desc="Running tokenizer on other train dataset",
                )
            print(f"Merging '{data_args.dataset_log_name}' ({len(train_dataset)}) and '{data_args.other_dataset_log_name}' ({len(other_train_dataset)}) datasets")
            train_dataset = concatenate_datasets(train_dataset, other_train_dataset)

    if training_args.do_eval:
        if "eval" not in raw_datasets:
            raise ValueError("--do_eval requires a eval dataset")
        eval_dataset = raw_datasets["eval"]
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples) if data_args.max_eval_samples is not None else len(eval_dataset)
        if max_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(max_eval_samples))
            raw_datasets["eval"] = eval_dataset
        with training_args.main_process_first(desc="eval dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                **preprocess_fn,
                cache_file_name = get_cache_file_name(
                    preprocess_fn['function'], 
                    **(preprocess_fn['fn_kwargs'] | {'dataset_name': data_args.dataset_log_name, 'max_eval_samples': max_eval_samples, 'seed': training_args.seed})
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=column_names,
                desc="Running tokenizer on eval dataset",
            )

        if any_other_dataset:
            other_eval_dataset = other_raw_datasets["eval"]
            other_max_eval_samples = min(len(other_eval_dataset), data_args.other_max_eval_samples) if data_args.other_max_eval_samples is not None else len(other_eval_dataset)
            if other_max_eval_samples < len(other_eval_dataset):
                other_eval_dataset = other_eval_dataset.shuffle(seed=training_args.seed).select(range(other_max_eval_samples))
                other_raw_datasets["eval"] = other_eval_dataset
            with training_args.main_process_first(desc="other_eval dataset map pre-processing"):
                other_eval_dataset = other_eval_dataset.map(
                    **other_preprocess_fn,
                    cache_file_name = get_cache_file_name(
                        other_preprocess_fn['function'], 
                        **(other_preprocess_fn['fn_kwargs'] | {'dataset_name': data_args.other_dataset_log_name, 'max_eval_samples': other_max_eval_samples, 'seed': training_args.seed})
                    ),
                    batched=True,
                    num_proc=data_args.other_preprocessing_num_workers,
                    load_from_cache_file=not data_args.other_overwrite_cache,
                    remove_columns=other_column_names,
                    desc="Running tokenizer on other eval dataset",
                )
            print(f"Merging '{data_args.dataset_log_name}' ({len(eval_dataset)}) and '{data_args.other_dataset_log_name}' ({len(other_eval_dataset)}) datasets")
            eval_dataset = concatenate_datasets(eval_dataset, other_eval_dataset)

    if training_args.do_predict:
        if "predict" not in raw_datasets:
            raise ValueError("--do_predict requires a predict dataset")
        predict_dataset = raw_datasets["predict"]
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples) if data_args.max_predict_samples is not None else len(predict_dataset)
        if max_predict_samples < len(predict_dataset):
            predict_dataset = predict_dataset.shuffle(seed=training_args.seed).select(range(max_predict_samples))
            raw_datasets["predict"] = predict_dataset
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                **preprocess_fn,
                cache_file_name = get_cache_file_name(
                    preprocess_fn['function'], 
                    **(preprocess_fn['fn_kwargs'] | {'dataset_name': data_args.dataset_log_name, 'max_predict_samples': max_predict_samples, 'seed': training_args.seed})
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=column_names,
                desc="Running tokenizer on prediction dataset",
            )

        if any_other_dataset:
            other_predict_dataset = other_raw_datasets["predict"]
            other_max_predict_samples = min(len(other_predict_dataset), data_args.other_max_predict_samples) if data_args.other_max_predict_samples is not None else len(other_predict_dataset)
            if other_max_predict_samples < len(other_predict_dataset):
                other_predict_dataset = other_predict_dataset.shuffle(seed=training_args.seed).select(range(other_max_predict_samples))
                other_raw_datasets["predict"] = other_predict_dataset
            with training_args.main_process_first(desc="other_prediction dataset map pre-processing"):
                other_predict_dataset = other_predict_dataset.map(
                    **other_preprocess_fn,
                    cache_file_name = get_cache_file_name(
                        other_preprocess_fn['function'], 
                        **(other_preprocess_fn['fn_kwargs'] | {'dataset_name': data_args.other_dataset_log_name, 'max_predict_samples': other_max_predict_samples, 'seed': training_args.seed})
                    ),
                    batched=True,
                    num_proc=data_args.other_preprocessing_num_workers,
                    load_from_cache_file=not data_args.other_overwrite_cache,
                    remove_columns=other_column_names,
                    desc="Running tokenizer on other prediction dataset",
                )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomEarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience, early_stopping_threshold=data_args.early_stopping_threshold)
            # EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience, early_stopping_threshold=data_args.early_stopping_threshold)
        ]
    )

    # Training
    best_model_loaded = False
    if training_args.do_train:
        assert training_args.load_best_model_at_end, "Training without loading the best model at the end?"
        try:
            train_result = trainer.train(
                resume_from_checkpoint = get_resume_from_checkpoint(training_args)
            )
        except KeyboardInterrupt:
            logger.warning(f"Training interrupted. Saving the current chechpoint in output_dir={training_args.output_dir}")
            # if load_best_model(trainer, training_args):  # NOTE: If best is loaded, samples from best upto latest checkpoint will be skipped in a new resumption
            trainer.save_model()  # training_args.output_dir
            trainer.save_state()
            print('output_dir:', training_args.output_dir)
            raise
        
        if trainer.state.best_model_checkpoint:
            best_model_loaded = True
            logger.warning(f">>> Best checkpoint ({trainer.state.best_model_checkpoint.split('/')[-1]}) loaded at the end of training.")

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

        metrics = train_result.metrics
        metrics["train_n_docs"] = max_train_samples
        if any_other_dataset:
            metrics["other_train_n_docs"] = other_max_train_samples
        metrics["train_n_seqs"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    print(f">>> trainer.state.best_model_checkpoint: {trainer.state.best_model_checkpoint}")
    if training_args.do_eval or training_args.do_predict:
        if not best_model_loaded:
            load_best_model(trainer, training_args)
            best_model_loaded = True

        _max_samples = f"{max_eval_samples}-{max_predict_samples}" if training_args.do_eval else f"{max_predict_samples}"
        _preds_dir_name = get_preds_dir_name(dataset_name=data_args.dataset_log_name, max_samples=_max_samples, 
                                                 seed=training_args.seed, drop_titles_probability=data_args.drop_titles_probability, 
                                                 partition_strategy=data_args.partition_strategy, partition_value=data_args.partition_value,
                                                 partition_disable_strategy=data_args.partition_disable_strategy, partition_disable_value=data_args.partition_disable_value)
        preds_save_dir = os.path.join(training_args.output_dir, _preds_dir_name)

        os.makedirs(preds_save_dir, exist_ok=True)

        def get_df_metrics(dataset, split, max_split_samples, overwrite_cache):
            df_path = os.path.join(preds_save_dir, f"{split}_df.csv")
            metrics_path = os.path.join(preds_save_dir, f"{split}_metrics.json")
            
            df, metrics = None, None
            if not training_args.do_train and (os.path.exists(df_path) and os.path.exists(metrics_path)) and not overwrite_cache:
                print(f'--- Reusing cached predictions of {split} split ---')
                if trainer.is_world_process_zero():
                    df = pd.read_csv(df_path, dtype={'doc_id': str})
                    for col in df.columns:
                        if col.endswith('s'):
                            df[col] = df[col].apply(lambda x: eval(x))
                    metrics = json.load(open(metrics_path, 'r'))
            else:
                print(f'--- Predicting {split} split ---')
                pout = trainer.predict(dataset, metric_key_prefix=split)
                logits, labels, metrics = pout.predictions, pout.label_ids, pout.metrics or {}

                metrics[f"{split}_n_docs"] = max_split_samples
                metrics[f"{split}_n_seqs"] = len(dataset)
                trainer.log_metrics(split, metrics)
                # trainer.save_metrics(split, metrics)
                
                if trainer.is_world_process_zero():
                    df = get_doc_grouped_df(dataset, logits, labels, tokenizer=tokenizer, n_logits=config.num_logits)
                    
                    df.to_csv(df_path, index=False)
                    json.dump(metrics, open(metrics_path, 'w'), indent=4)                
                    
            return df, metrics

    # Evaluation
    eval_df = None
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_df, eval_metrics = get_df_metrics(eval_dataset, 'eval', 
                                               max_split_samples = str((max_eval_samples, other_max_eval_samples)) if any_other_dataset else max_eval_samples, 
                                               overwrite_cache=data_args.overwrite_cache)

    # Predict
    if training_args.do_predict:
        pfactory = PredictionFactory(trainer.log_metrics, trainer.save_metrics, expr_metric=data_args.expr_metric)
        if len(predict_dataset) > 0:
            logger.info("*** Predict ***")
            
            predict_df, predict_metrics = get_df_metrics(predict_dataset, 'predict', max_predict_samples, data_args.overwrite_cache)
            
            if trainer.is_world_process_zero():
                out_resutls_kwargs = dict(
                    output_dir=training_args.output_dir, seed=training_args.seed, batch_size=training_args.per_device_eval_batch_size,
                    expr_name=data_args.expr_name,
                    partition_strategy=data_args.partition_strategy, partition_value=data_args.partition_value, 
                    partition_disable_strategy=data_args.partition_disable_strategy, partition_disable_value=data_args.partition_disable_value, 
                    split='predict', metrics=predict_metrics,
                    save_path=data_args.out_results_path,
                )
                raw_predict_df = raw_datasets["predict"].to_pandas()
                pfactory.predict_and_save(raw_predict_df, predict_df, 
                                          expr_name=data_args.expr_name, expr_threshold=data_args.expr_threshold, expr_hpm_path=data_args.expr_hpm_path,
                                          eval_df=eval_df, 
                                          docs_save_dir = preds_save_dir if not training_args.do_train else None, 
                                          chunks_save_dir = preds_save_dir if not training_args.do_train else None, 
                                          hpms_save_dir = preds_save_dir if not training_args.do_train else None,
                                          out_results_kwargs = out_resutls_kwargs)
                                   
        else:
            logger.warning("Predict dataset is empty")
        
        # Other prediction #
        if any_other_dataset:
            if len(other_predict_dataset) > 0:
                logger.info("*** Other Predict ***")
                
                other_predict_df, other_predict_metrics = get_df_metrics(other_predict_dataset, 'other_predict', other_max_predict_samples, data_args.other_overwrite_cache)
                
                if trainer.is_world_process_zero():
                    other_out_results_kwargs = dict(
                        output_dir=training_args.output_dir, seed=training_args.seed, batch_size=training_args.per_device_eval_batch_size,
                        expr_name=data_args.expr_name,
                        partition_strategy=data_args.partition_strategy, partition_value=data_args.partition_value, 
                        partition_disable_strategy=data_args.partition_disable_strategy, partition_disable_value=data_args.partition_disable_value, 
                        split='other_predict', metrics=other_predict_metrics,
                        save_path=data_args.other_out_results_path,
                    )

                    other_raw_predict_df = other_raw_datasets["predict"].to_pandas()
                    pfactory.predict_and_save(other_raw_predict_df, other_predict_df, 
                                              expr_name=data_args.expr_name, expr_threshold=data_args.expr_threshold, expr_hpm_path=data_args.expr_hpm_path,
                                              eval_df=eval_df, 
                                              docs_save_dir=None, chunks_save_dir=None, hpms_save_dir=None, out_results_kwargs=other_out_results_kwargs)
                            
            else:
                logger.warning("Other predict dataset is empty")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()