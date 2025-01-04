import logging
from typing import Dict, List, Optional, Union
import os
import inspect
import json
import hashlib

from torch.utils.data import SequentialSampler, RandomSampler
from transformers import Trainer, EarlyStoppingCallback, TrainerCallback, IntervalStrategy, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def get_cache_file_name(function, __suffix='.arrow', **fn_kwargs):
    fn_kwargs = {k: v for k, v in fn_kwargs.items() if k not in ['tokenizer', 'verbose']}
    
    fn_source = inspect.getsource(function)
    args_serialized = json.dumps(fn_kwargs, sort_keys=True)
    serialized = fn_source + args_serialized
    
    path = os.environ['HF_HOME']
    uid = hashlib.md5(serialized.encode('utf-8')).hexdigest()
    path = os.path.join(path, 'custom_datasets', fn_kwargs['dataset_name'], uid + __suffix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print('\tTokenization cache path:', path)
    return path


def get_preds_dir_name(dataset_name, max_samples, seed, drop_titles_probability, partition_strategy, partition_value, partition_disable_strategy, partition_disable_value):
    return f"predictions_{dataset_name}_{max_samples}" \
            + f"_seed-{seed}" \
            + f"_dtp-{drop_titles_probability or 0}" \
            + f"_{partition_disable_strategy}-{partition_disable_value}" \
            + f"_{partition_strategy}-{partition_value}"


def get_sent_sep_token(sent_sep, tokenizer):
    if sent_sep == 'cls':
        return tokenizer.cls_token
    elif sent_sep == 'sep':
        return tokenizer.sep_token
    elif sent_sep == 'snt':
        sep_token = tokenizer.sep_token
        s, e = sep_token[0], sep_token[-1]
        sep_name = sent_sep.upper() if sep_token.isupper() else sent_sep.lower()
        return f'{s}{sep_name}{e}'
    else:
        raise ValueError(f'Unknown sentence separator token: {sent_sep}')


def add_new_special_token(token, tokenizer, model):
    if token not in tokenizer.all_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [token]})
        model.resize_token_embeddings(len(tokenizer))


def count_parameters(model):
    from collections import defaultdict
    out = defaultdict(int)
    # Iterating through the model's layers
    for name, p in model.named_parameters():
        pname = name.split('.')[0]
        out[pname] += p.numel() if p.requires_grad else 0

    out['total'] = sum(out.values())
    return dict(out)


def freeze_parameters(model, whitelist=None, blacklist=None):
    assert not (whitelist and blacklist), "Only one of whitelist and blacklist can be specified"

    if whitelist:
        print('Freezing parameters other than:', whitelist)
        for pname, p in model.named_parameters():
            if not any(pname.startswith(n) for n in whitelist):
                p.requires_grad = False

    elif blacklist:
        print('Freezing parameters:', blacklist)
        for pname, p in model.named_parameters():
            if any(pname.startswith(n) for n in blacklist):
                p.requires_grad = False


def get_resume_from_checkpoint(training_args: TrainingArguments):
    if not training_args.do_train:
        return None
    
    if training_args.resume_from_checkpoint is not None:
        return training_args.resume_from_checkpoint
    
    # Look for a last checkpoint in output_dir to resume training if desired
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


### Inspired from transformers.Trainer._inner_train_loop
def load_best_model(trainer: Trainer, args):
    import torch.distributed as dist
    from transformers.training_args import ParallelMode
    
    if trainer.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        # if is_torch_xla_available():
        #     xm.rendezvous("load_best_model_at_end")
        if args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        # elif is_sagemaker_mp_enabled():
        #     smp.barrier()

        trainer._load_best_model()
        
        print(f"Best checkpoint loaded from {trainer.state.best_model_checkpoint.split('/')[-1]}")
        return True
    return False

### Copied and modified from transformers.EarlyStoppingCallback ###
class CustomEarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model

        def improved():
            """Supports negative self.early_stopping_threshold"""
            if state.best_metric is None:
                return True
            delta_imp = metric_value - state.best_metric if args.greater_is_better else state.best_metric - metric_value
            return delta_imp > self.early_stopping_threshold
            
        if improved():
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


class NoShuffleTrainer(Trainer):
    def _get_train_sampler(self) -> SequentialSampler:
        return SequentialSampler(self.train_dataset)

    # def get_train_dataloader(self) -> DataLoader:
    #     return super().get_train_dataloader()
