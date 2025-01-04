from collections import defaultdict
from typing import Any
from math import ceil

import torch


def _to_list(tensor_or_iterable):
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.tolist()
    return list(tensor_or_iterable)


def _pad1d(tokens, padding_token: int, max_length: int, padding_side: str = 'right'):
    assert padding_side in ["right", "left"]
    tokens = _to_list(tokens)
    padding_tokens = [padding_token] * (max_length - len(tokens))
    if padding_side == "right": 
        return tokens + padding_tokens  
    else: 
        return padding_tokens + tokens


def _tokenizer_pad1d(feature: dict, max_length: int, tokenizer, 
                     INPUT_IDS='input_ids', ATTENTION_MASK='attention_mask', TOKEN_TYPE_IDS='token_type_ids', GLOBAL_ATTENTION_MASK='global_attention_mask'):
    out = {
        INPUT_IDS: _pad1d(feature[INPUT_IDS], tokenizer.pad_token_id, max_length, tokenizer.padding_side),
        ATTENTION_MASK: _pad1d(feature[ATTENTION_MASK], 0, max_length, tokenizer.padding_side),
    }
    if TOKEN_TYPE_IDS in feature:
        out[TOKEN_TYPE_IDS] = _pad1d(feature[TOKEN_TYPE_IDS], 0, max_length, tokenizer.padding_side)
    if GLOBAL_ATTENTION_MASK in feature:
        out[GLOBAL_ATTENTION_MASK] = _pad1d(feature[GLOBAL_ATTENTION_MASK], 0, max_length, tokenizer.padding_side)
    return out


class DataCollatorForTokenClassification():
    def __init__(self, tokenizer, label_pad_token_id=-100, pad_to_multiple_of=None) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.debug = True  # TODO: False
    
    def __call__(self, features, return_tensors=None) -> Any:
        """
        features[0] = {
            'input_ids': (token_len, ),
            ...,
            'labels': (token_len),
        }
        batch = {
            'input_ids': (batch_size, max_token_len),
            ...,
            'labels': (batch_size, max_token_len),
        }
        """
        max_length = max(len(feature['input_ids']) for feature in features)
        if self.pad_to_multiple_of is not None:
            max_length = int(ceil(max_length / self.pad_to_multiple_of)) * self.pad_to_multiple_of

        batch = defaultdict(list)
        for feature in features:
            seq_batch = _tokenizer_pad1d(feature, max_length, self.tokenizer)
            if 'labels' in feature:
                seq_batch['labels'] = _pad1d(feature['labels'], self.label_pad_token_id, max_length, self.tokenizer.padding_side) 
            for key in seq_batch:
                batch[key].append(seq_batch[key])

        for key in batch:
            batch[key] = torch.tensor(batch[key], dtype=torch.long)
            # print(key, batch[key].shape)

        if self.debug:
            _log_batch1d(batch, self.tokenizer)
            self.debug = False

        return dict(batch)


### Debugging ###

from eval_util.util import decode_seq_to_sents


def _log_batch1d(batch, tokenizer):
    print('\n### BATCH:')
    for key in batch:
        print('\t', key, batch[key].shape) #, batch[key][0])

    try: 
        for i in range(len(batch['input_ids'])):
            print(f'\n### SEQ {i}:')
            
            seq_text, seq_indices, _ = decode_seq_to_sents(tokenizer, batch['input_ids'][i], 
                                                        sep_token_mask = batch['sep_token_mask'][i] if 'sep_token_mask' in batch else None,
                                                        labels = batch['labels'][i] if 'labels' in batch else None, 
                                                        return_indices=True)
            assert len(seq_indices) == len(seq_text)
                    
            for j, (idx, text) in enumerate(zip(seq_indices, seq_text)):
                assert tokenizer.decode(batch['input_ids'][i][idx].item()) in tokenizer.all_special_tokens
                assert batch['attention_mask'][i][idx].item() == 1
                if 'token_type_ids' in batch: 
                    assert batch['token_type_ids'][i][idx].item() == 0
                if 'global_attention_mask' in batch: 
                    assert batch['global_attention_mask'][i][idx].item() == 1
                
                sent_label = batch['labels'][i][idx].item()
                assert sent_label in [0, 1, -100]

                print(f'\t SENT {j}: {sent_label} | {text}')
                if sent_label == 1:
                    print(20 * '-')

            print(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False), end='\n\n')

            if i >= 2:
                break
    except Exception as e:
        # print(e)
        pass
