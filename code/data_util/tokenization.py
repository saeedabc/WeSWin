from collections import defaultdict
import random
from typing import Any
import numpy as np

from transformers import PreTrainedTokenizerBase


def drop_titles(sent_titles_mask, *sent_others_list, drop_titles_probability=None):
    keep_titles = not drop_titles_probability or (1 > drop_titles_probability <= random.random())
    if keep_titles:
        return sent_titles_mask, *sent_others_list
    
    new_titles_mask = [tm for tm in sent_titles_mask if tm == 0]
    out = (new_titles_mask, )
    for sent_others in sent_others_list:
        assert len(sent_others) == len(sent_titles_mask)
        new_sent_others = [so for so, tm in zip(sent_others, sent_titles_mask) if tm == 0]
        out = out + (new_sent_others, )
    return out


def preprocess_token_classification(examples, 
                                    tokenizer: PreTrainedTokenizerBase, sent_sep_token: str, 
                                    doc_id_column_name: str = 'id', id_column_name: str = 'ids', text_column_name: str = 'sentences', label_column_name: str = 'labels', title_mask_column_name: str = 'titles_mask', 
                                    layout: str = 'seqs_2d', model_type: str = None,
                                    max_tokens_per_seq: int = 512, max_sents_per_doc: int = None, max_seqs_per_doc: int = None,
                                    drop_titles_probability: float = None,
                                    partition_strategy: str = 'ex', partition_value: Any = 0, 
                                    partition_disable_strategy: str = 'na', partition_disable_value: int = 0,
                                    verbose: bool = False):
    
    assert layout in ['seqs_2d', 'docs_2d', 'docs_3d']
    
    assert drop_titles_probability is None or 0 <= drop_titles_probability <= 1

    # partition_disable
    assert partition_disable_strategy in ['na', 'cl', 'cr', 'clr']
    if partition_disable_strategy == 'na':
        assert partition_disable_value == 0
    else:
        assert isinstance(partition_disable_value, int) and partition_disable_value > 0
    
    # partition
    if partition_strategy == 'ex':
        assert partition_value == 0
    else:
        if partition_strategy in ['si', 'ss']:
            assert isinstance(partition_value, int) and partition_value > 0
        elif partition_strategy in ['sip', 'ssp']:
            assert isinstance(partition_value, float) and 0 < partition_value <= 1
            
        assert layout == 'seqs_2d', 'Only layout=seqs_2d is supported with overlapping partitioning'
        assert max_seqs_per_doc is None and max_sents_per_doc is None
    
    assert sent_sep_token in tokenizer.all_special_tokens
    sent_sep_token_id = tokenizer.convert_tokens_to_ids(sent_sep_token)
    pad_token_id = tokenizer.pad_token_id; assert pad_token_id is not None
    cls_token_id = tokenizer.cls_token_id; assert cls_token_id is not None
    sep_token_id = tokenizer.sep_token_id; assert sep_token_id is not None
    eos_token_id = tokenizer.eos_token_id or sep_token_id

    max_tokens_per_sent = min(128, max_tokens_per_seq) - 3  # cls, sent_sep, eos/sep

    if verbose: 
        print(
            'max_tokens_per_seq', max_tokens_per_seq, 'max_seqs_per_doc', max_seqs_per_doc, 'max_sents_per_doc', max_sents_per_doc, 
            'drop_titles_probability', drop_titles_probability, 
            'partition_strategy', partition_strategy, 'partition_value', partition_value, 
            'partition_disable_value', partition_disable_value, 'partition_disable_strategy', partition_disable_strategy,
        )

        
    class Document:
        def __init__(self, id, sents, sent_ids, sent_labels, pad_to_max=False):
            self.id = id
            
            self.sents = sents
            self.sent_ids = sent_ids
            self.sent_labels = sent_labels

            assert len(sents) > 0
            self.tokenized_sents = tokenizer(sents, add_special_tokens=False, padding=False, truncation=True, max_length=max_tokens_per_sent)
                    
            assert len(sents) == len(sent_ids) == len(sent_labels) == len(self.tokenized_sents['input_ids'])
            self.active_counts = [0] * len(sents)
            
            self.pad_to_max = pad_to_max
                
        def _sent_len(self, sent_idx, snt=True):
            return len(self.tokenized_sents['input_ids'][sent_idx]) + (1 if snt else 0)
        
        def _next_tokenized_seq_span(self, sent_idx, max_tokens, max_sents=None):
            """
            Returns a sized span of sent indices of form (sent_idx, disable) where disable is True if the sentence is disabled.
            The first active sentence is always sent_idx.
            """
            
            assert max_sents is None, 'max_sents is not supported'
            
            def _prepend(sidx):
                nonlocal span, active, rem_len, lctx
                
                if rem_len > 0 and lctx > 0:
                    if sidx >= 0 and (sent_len := self._sent_len(sidx)) <= rem_len:
                        span.insert(0, sidx)
                        active.insert(0, False)
                        rem_len -= sent_len
                        lctx -= 1
                    else:
                        lctx = 0
            
            def _append(sidx):
                nonlocal span, active, rem_len, rctx
                
                if rem_len > 0 and rctx > 0:
                    if sidx < len(self.sents) and (sent_len := self._sent_len(sidx)) <= rem_len:
                        span.append(sidx)
                        active.append(True)
                        rem_len -= sent_len
                        rctx -= 1
                    else:
                        rctx = 0
            
            def _prepend_while():
                i = 1
                while rem_len > 0 and lctx > 0:
                    _prepend(sent_idx - i)
                    i += 1
            
            def _append_while():
                i = 1
                while rem_len > 0 and rctx > 0:
                    _append(sent_idx + i)
                    i += 1
            
            def _prepend_append_while():
                i = 1
                while rem_len > 0 and (lctx > 0 or rctx > 0):
                    _prepend(sent_idx - i)
                    _append(sent_idx + i)
                    i += 1
            
            def _disable_rtx():
                nonlocal active
                
                rctx = partition_disable_value if partition_disable_strategy in ['cr', 'clr'] and span[-1] < len(self.sents) - 1 else 0
                if rctx > 0:
                    for i in range(len(span) - 1, -1, -1):
                        if rctx > 0 and span[i] > sent_idx:
                            active[i] = False
                            rctx -= 1
                        else:
                            break
            
            def _append_partial_rtx():
                nonlocal span, active, rem_len
                
                if rem_len > 0 and (tail_idx := span[-1] + 1) < len(self.sents):
                    span.append(tail_idx)
                    active.append(False)
                    rem_len -= self._sent_len(tail_idx)
                    assert rem_len < 0
            
            span = [sent_idx]
            active = [True]
            rem_len = max_tokens - self._sent_len(sent_idx)
            
            lctx = partition_disable_value if partition_disable_strategy in ['cl', 'clr'] and sent_idx > 0 else 0
            rctx = (len(self.sents) - 1) - (sent_idx + 1) + 1
              
            if partition_disable_strategy == 'na':
                _append_while()
            elif partition_disable_strategy == 'cl':
                _prepend_while()
                _append_while()
            elif partition_disable_strategy == 'cr':
                _append_while()
                _disable_rtx()
            elif partition_disable_strategy == 'clr':        
                _prepend_append_while()
                _disable_rtx()
            else:
                raise ValueError(f'Invalid partition_disable_strategy: {partition_disable_strategy}')
            
            _append_partial_rtx()
            
            assert len(span) == len(active) and active[span.index(sent_idx)] == True
            assert 0 <= span[0] <= span[-1] <= len(self.sents) - 1
            assert all(span[i+1] - span[i] == 1 for i in range(len(span) - 1)), f'{span}'
            # assert len(span) == len(dis)
            return list(zip(span, active))

        def next_tokenized_seq(self, sent_idx, max_tokens, max_sents=None):

            def add_tokens(out, kwargs):
                key = next(iter(kwargs))      
                if isinstance(kwargs[key], list):
                    for key in kwargs:
                        out[key].extend(kwargs[key])
                else:
                    for key in kwargs:
                        out[key].append(kwargs[key])
            
            def get_cls_out(cls_token_id):
                return dict(input_ids=cls_token_id, attention_mask=1, token_type_ids=0, global_attention_mask=0, sep_token_mask=0, labels=-100)  # global_attention_mask = 1?
            
            def get_sent_out(input_ids, attention_mask):
                slen = len(input_ids)
                return dict(input_ids=input_ids, attention_mask=attention_mask, 
                            token_type_ids=[0] * slen, global_attention_mask=[0] * slen, sep_token_mask=[0] * slen, labels=[-100] * slen)
            
            def get_snt_out(snt_token_id, labels, sep_token_mask=1, **kwargs):
                return dict(input_ids=snt_token_id, attention_mask=1, token_type_ids=0, global_attention_mask=1, sep_token_mask=sep_token_mask, labels=labels, **kwargs)

            def get_eos_out(eos_token_id):
                return dict(input_ids=eos_token_id, attention_mask=1, token_type_ids=0, global_attention_mask=0, sep_token_mask=0, labels=-100)
            
            def get_pad_out(pad_token_id, pad_len):
                return dict(input_ids=[pad_token_id] * pad_len, attention_mask=[0] * pad_len, 
                                token_type_ids=[0] * pad_len, global_attention_mask=[0] * pad_len, sep_token_mask=[0] * pad_len, labels=[-100] * pad_len)
            
            def sanity_check(out, max_tokens):
                expected_keys = [
                    'doc_id', 
                    'input_ids', 'attention_mask', 'token_type_ids', 'global_attention_mask', 'sep_token_mask', 'labels', 
                    'sent_ids', 'sent_indices', 
                ]
                assert set(out.keys()) == set(expected_keys), f'{set(out.keys())} != {set(expected_keys)}'
                
                for key in out:
                    if key in ['doc_id']:
                        continue
                    elif key in ['sent_ids', 'sent_indices']:
                        assert len(out[key]) == out['sep_token_mask'].count(1)  # == (len(window)-1 if ignore_last_sentence_label else len(window))
                    else:
                        assert len(out[key]) == len(out['input_ids']) <= max_tokens, f'{key}: {len(out[key])}, input_ids: {len(out["input_ids"])}, max_length={max_tokens}'
    
            assert max_sents is None, 'max_sents is not supported for now'
            
            rem_len = max_tokens - 2  # [CLS], [EOS]

            out = defaultdict(list)

            # [CLS]
            cls_out = get_cls_out(cls_token_id)
            add_tokens(out, cls_out)
                        
            # sent tokens of a window
            seq_span = self._next_tokenized_seq_span(sent_idx, max_tokens=rem_len, max_sents=max_sents)
            for (sidx, active) in seq_span:
                assert rem_len > 0
                
                sent_rem_len = min(rem_len, self._sent_len(sidx, snt=False))
                sent_out = get_sent_out(input_ids = self.tokenized_sents['input_ids'][sidx][:sent_rem_len], 
                                        attention_mask = self.tokenized_sents['attention_mask'][sidx][:sent_rem_len])
                add_tokens(out, sent_out)
                rem_len -= sent_rem_len
                
                if rem_len > 0:
                    if active:
                        snt_out = get_snt_out(sent_sep_token_id, labels=sent_labels[sidx], sep_token_mask=1, sent_ids=sent_ids[sidx], sent_indices=sidx)
                        self.active_counts[sidx] += 1
                    else:
                        snt_out = get_snt_out(sent_sep_token_id, labels=-100, sep_token_mask=0)
                    add_tokens(out, snt_out)
                    rem_len -= 1
                               
            # [EOS]
            eos_out = get_eos_out(eos_token_id)
            add_tokens(out, eos_out)
            
            # [PAD]
            if self.pad_to_max:
                pad_out = get_pad_out(pad_token_id, pad_len=rem_len)
                add_tokens(out, pad_out)
            
            out |= {'doc_id': self.id}                 
            sanity_check(out, max_tokens)
            
            return out, self._next_sent_idx(out)
        
        def _active_bounds(self, seq_out=None, seq_span=None):
            if seq_out is not None:
                seq_start_idx = seq_out['sent_indices'][0]
                seq_end_idx = seq_out['sent_indices'][-1]
            if seq_span is not None:
                seq_start_idx = next(sidx for sidx, active in seq_span if active)
                seq_end_idx = next(sidx for sidx, active in reversed(seq_span) if active)
            
            assert 0 <= seq_start_idx <= seq_end_idx <= len(self.sents) - 1
            return seq_start_idx, seq_end_idx
        
        def _next_sent_idx(self, seq_out):
            seq_start_idx, seq_end_idx = self._active_bounds(seq_out)

            def ex_update():
                return seq_end_idx + 1
            
            def si_update(k: int):
                assert k > 0
                return max(seq_end_idx + 1 - k, seq_start_idx + 1)
            
            def sip_update(p: float):
                assert 0 < p <= 1
                for sidx in range(seq_end_idx, seq_start_idx, -1):
                    if random.random() < p:
                        return sidx
                return seq_end_idx + 1
            
            def ss_update(k: int):
                assert k > 0
                return min(seq_start_idx + k, seq_end_idx)
                # return min(seq_start_idx + k, seq_end_idx + 1)
            
            def ssp_update(p: float):
                assert 0 < p <= 1
                for sidx in range(seq_start_idx + 1, seq_end_idx + 1):
                    if random.random() < p:
                        return sidx
                return seq_end_idx
            
            def next_sent_idx():
                if seq_end_idx == len(self.sents) - 1:
                    return ex_update()
                
                if partition_strategy == 'ex':
                    return ex_update()
                if partition_strategy == 'ss':
                    return ss_update(partition_value)
                if partition_strategy == 'si':
                    return si_update(partition_value)
                if partition_strategy == 'ssp':
                    return ssp_update(partition_value)
                if partition_strategy == 'sip':
                    return sip_update(partition_value)   
                raise ValueError(f'Invalid partition_strategy: {partition_strategy}')
                
            return next_sent_idx()
        
        def generate_subdoc_seqs(self):
            def sanity_check(subdoc):
                assert not max_seqs_per_doc or 0 < len(subdoc) <= max_seqs_per_doc
                assert not max_sents_per_doc or 0 < sum(seq_out['sent_indices'] for seq_out in subdoc) <= max_sents_per_doc
                for seq_out in subdoc:
                    assert len(seq_out['input_ids']) <= max_tokens_per_seq

            def get_subdoc_out(subdoc):
                sanity_check(subdoc)
                
                subdoc_out = defaultdict(list)
                for seq_out in subdoc:
                    for key in seq_out:
                        subdoc_out[key].append(seq_out[key])
                return subdoc_out

            subdoc = []
            subdoc_sent_len = 0

            sent_idx = 0
            while sent_idx < len(self.sents):
                # Generate the next sequence
                seq_out, sent_idx = self.next_tokenized_seq(sent_idx, 
                                                            max_tokens = max_tokens_per_seq, 
                                                            max_sents = max_sents_per_doc - subdoc_sent_len if max_sents_per_doc else None)
            
                # Add the sequence to the current doc
                subdoc.append(seq_out)
                subdoc_sent_len += len(seq_out['sent_indices'])

                # Yield the current doc if it's full
                if (max_seqs_per_doc and len(subdoc) == max_seqs_per_doc) or (max_sents_per_doc and subdoc_sent_len == max_sents_per_doc):
                    yield get_subdoc_out(subdoc)
                    subdoc = []
                    subdoc_sent_len = 0

            # Yield the current doc if not yielded yet
            if subdoc:
                yield get_subdoc_out(subdoc)
    

    # all_sents: (batch/n_docs, n_sents) - str
    all_sents = examples[text_column_name]
    all_doc_ids = examples[doc_id_column_name]
    all_sent_ids = examples[id_column_name]
    # all_labels: (batch/n_docs, n_sents) - int
    all_sent_labels = examples.get(label_column_name, [[None] * len(sents) for sents in all_sents])
    all_sent_titles_mask = examples.get(title_mask_column_name)

    out = defaultdict(list)

    # For each document
    for doc_idx, sents in enumerate(all_sents):
        doc_id = all_doc_ids[doc_idx]
        sent_ids = all_sent_ids[doc_idx]
        sent_labels = all_sent_labels[doc_idx]
        assert len(sents) == len(sent_labels) == len(sent_ids)

        sent_titles_mask = all_sent_titles_mask[doc_idx] if all_sent_titles_mask is not None else None
        if sent_titles_mask is not None:
            assert len(sents) == len(sent_titles_mask)
            sent_titles_mask, sents, sent_labels, sent_ids = drop_titles(sent_titles_mask, sents, sent_labels, sent_ids, drop_titles_probability=drop_titles_probability)
        
        document = Document(doc_id, sents, sent_ids, sent_labels,
                            pad_to_max=(layout == 'docs_3d' or layout == 'docs_2d'))
        
        for subdoc_out in document.generate_subdoc_seqs():
            for key in subdoc_out:
                out[key].append(subdoc_out[key])
        
        assert all(a >= 1 for a in document.active_counts), f'{doc_id}: {document.active_counts}'
        
    # Remove labels if not present in examples
    if label_column_name not in examples:
        del out['labels']

    if model_type != 'longformer':
        del out['global_attention_mask']
    if model_type in ['roberta', 'longformer']:
        del out['token_type_ids']

    # Log stats
    if verbose:
        def get_stats(li):
            return round(np.mean(li), 2), round(np.std(li), 2)

        stats = {
            'n_docs': len(out['input_ids']), 
            'n_seqs_per_doc': get_stats([len(doc) for doc in out['input_ids']]), 
            'n_sents_per_doc': get_stats([sum(len(seq) for seq in doc) for doc in out['sent_ids']]), 
            'n_sents_per_seq': get_stats([len(seq) for doc in out['sent_ids'] for seq in doc])
        }
        print(stats)

    # Reshape out based on layout
    if layout == 'docs_3d':
        return out

    elif layout == 'docs_2d':
        out2d = defaultdict(list)
        for key in out:
            out2d[key] = [[token for seq in doc for token in seq] for doc in out[key]]
        return out2d
    
    elif layout == 'seqs_2d':
        out2d = defaultdict(list)
        for key in out:
            out2d[key] = [seq for doc in out[key] for seq in doc]
        return out2d

    else:
        raise ValueError(f'layout={layout} not supported')
