from pathlib import Path
from tqdm.auto import tqdm
import json
from functools import partial
import itertools
import dill

import numpy as np
import pandas as pd

from eval_util.util import logits_to_probs, probs_to_preds, clean_indices, decode_seq_to_sents
from eval_util.evaluation import compute_metrics


def extract_segments(docs, pred_col, prob_col, save_dir):
    
    def iter_segments(doc):
        segment = []
        for sent_id, sent, pred, prob in zip(doc['ids'], doc['sentences'], doc[pred_col], doc[prob_col]):
            segment.append({
                'id': sent_id, 
                'text': sent, 
                'prob': prob
            })
            if pred == 1:
                yield segment
                segment = []
        if len(segment) > 0:
            yield segment
            segment = []
    
    out = {
        'metadata': {},  # | {'n_docs': len(df), 'n_chunks': n_segs, 'n_sents': n_sents},
        'chunks': [],
    }
    n_segs = 0
    n_sents = 0
    for doc in docs:
        for _, segment in enumerate(iter_segments(doc)):
            out['chunks'].append({
                'id': n_segs + 1, 
                'chunk': segment,
                'doc_id': doc['id'],
            })
            n_segs += 1
            n_sents += len(segment)
        
    out['metadata']['n_docs'] = len(docs)
    out['metadata']['n_chunks'] = n_segs
    out['metadata']['n_sents'] = n_sents
        
    save_path = os.path.join(save_dir, f'segments_{pred_col}.json')  # {df_path.stem}
    print(f"\n{save_path}")
    
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=4)
    
    return out, save_path


def postprocess_predictions(raw_df, df, pred_col, prob_col, is_seg, save_dir):
    save_path = os.path.join(save_dir, f'docs_{pred_col}.jsonl')
    print(f"\n{save_path}")
    
    assert len(raw_df) == len(df)
    raw_doc_ids = raw_df['id'].tolist()
    raw_df = raw_df.set_index('id')
    df = df.set_index('doc_id')
 
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    docs = []
    for doc_id in raw_doc_ids:
        raw_row = raw_df.loc[doc_id].apply(_convert)
        row = df.loc[doc_id].apply(_convert)
        
        assert list(raw_row['ids']) == list(row['sent_ids'])
        assert len(raw_row['ids']) == len(row[pred_col]) == len(row[prob_col])
        
        doc = raw_row.to_dict()
        doc['id'] = doc_id
        doc[prob_col] = [round(prob, 4) for prob in row[prob_col]]
        doc[pred_col] = row[pred_col]
        docs.append(doc)
        
    with open(save_path, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
    
    return docs, save_path


def get_doc_grouped_df(dataset, out_logits, out_labels, 
                       tokenizer, n_logits):
    ### Initializing df ###
    labels_provided = out_labels is not None

    df = dataset.to_pandas()
    
    def add_sents_col(row):
        return decode_seq_to_sents(tokenizer=tokenizer, input_ids=row['input_ids'], sep_token_mask=row['sep_token_mask'])
    df['sents'] = df.apply(add_sents_col, axis=1)
    
    df['sent_ids'] = df['sent_ids'].apply(lambda sids: sids.tolist())

    n_sents = {
        'sent_ids': df['sent_ids'].apply(len).sum(),
        'sep_token_mask': df['sep_token_mask'].apply(sum).sum(),
        **({'out_labels': sum(sum(l != -100 for l in label) for label in out_labels)} if labels_provided else {}),
        **({'labels': df['labels'].apply(lambda label: sum(l != -100 for l in label)).sum()} if labels_provided else {}),
    }
    assert len(df) == len(out_logits)
    if labels_provided:
        assert len(df) == len(out_labels)
        for i in range(len(df)):
            assert all(l1 == l2 for l1, l2 in zip(df['labels'][i], out_labels[i])), f"labels: {df['labels'][i]}\nout_labels: {out_labels[i]}"

        df['logits'] = clean_indices(inputs=out_logits, sep_token_mask=df['sep_token_mask'], nd=2)
        df['labels'] = clean_indices(inputs=out_labels, sep_token_mask=df['sep_token_mask'], nd=2)
        n_sents['clean_labels'] = df['labels'].apply(lambda label: sum(l != -100 for l in label)).sum()
    else:
        df['logits'] = clean_indices(inputs=out_logits, sep_token_mask=df['sep_token_mask'], nd=2)

    def add_pos_probs_col(logits):
        probs = logits_to_probs(logits, n_logits=n_logits)
        if n_logits == 2:
            probs = probs[..., 1]
        return probs.tolist()
    df['probs'] = df['logits'].apply(add_pos_probs_col)
    
    def add_positions_col(sents):
        n = len(sents)
        return [(i, n) for i in range(n)]
    df['positions'] = df['sents'].apply(add_positions_col)
    
    keep_columns = ['doc_id', 'sent_ids', 'sents', 'logits', 'probs', 'positions', 'labels']
    df = df[[col for col in df.columns if col in keep_columns]]
    
    ### Grouping multiple sentence predictions in document level ###
    
    print('Exploding dataframe to sent rows...')
    df = df.apply(lambda col: col.explode() if col.name.endswith('s') else col).reset_index(drop=True)

    print('Grouping by doc_id and sent_ids...')
    # Group by 'doc_id' and 'sent_ids', aggregate 'logits' and 'labels'
    cols = [col for col in df.columns if col not in ['doc_id', 'sent_ids']]
    list_cols = ['logits', 'probs', 'positions']
    first_cols = [col for col in cols if col not in list_cols]
    df = df.groupby(['doc_id', 'sent_ids'], sort=False).agg({
        **({col: list for col in list_cols}),
        **({col: 'first' for col in first_cols}),
    }).reset_index()

    print('Preparing list of docs for inference...')
    # Convert back to df
    cols = [col for col in df.columns if col not in ['doc_id']]
    list_cols = [col for col in cols if col.endswith('s')]
    first_cols = [col for col in cols if col not in list_cols]
    df = df.groupby('doc_id', sort=False).agg({
        **({col: list for col in list_cols}),
        **({col: 'first' for col in first_cols}),
    }).reset_index()
    
    n_sents_grouped = {
        'sent_ids': df['sent_ids'].apply(len).sum(),
        'sents': df['sents'].apply(len).sum(),
        'probs': df['probs'].apply(len).sum(),
    }

    assert len(set(n_sents.values())) == 1, n_sents.values()
    assert len(set(n_sents_grouped.values())) == 1, n_sents_grouped.values()
    
    return df


### Loss-based Weighting ###

def _rel_pos(i, n, method=None):
    method = method or 'asym'
    if method == 'asym':
        return i if i <= (n - 1) / 2 else i - n
    elif method == 'sym':
        return min(i, n - 1 - i)
    else:
        raise ValueError(f"Invalid method: {method}")


def _sorted_pkeys(pdict, k=None):
    k = k or float('inf')
    pos_pkeys = sorted(pkey for pkey in pdict if 0 <= pkey < k)
    neg_pkeys = sorted(pkey for pkey in pdict if -k <= pkey < 0)
    return pos_pkeys, neg_pkeys


def _smoothed_pweights(pweights, k=None):
    # k = k or len(pweights)
    pwk = {}
    
    pos_pkeys, neg_pkeys = _sorted_pkeys(pweights, k=k)
    
    maxl = 0
    for pkey in pos_pkeys:
        maxl = max(pweights[pkey], maxl)
        pwk[pkey] = maxl

    maxr = 0
    for pkey in reversed(neg_pkeys):
        maxr = max(pweights[pkey], maxr)
        pwk[pkey] = maxr
        
    pwk = {pkey: pwk[pkey] for pkey in pos_pkeys + neg_pkeys}
    max_ = max(maxl, maxr)
    return pwk, max_


def get_lob_weights(df, method=None, count_normalized=False):
    from collections import defaultdict
    import torch
    import numpy as np
    from model.util import cross_entropy_loss

    def _mean_bce_error(logits, labels, nlogits):
        assert len(logits) == len(labels) > 0
        return cross_entropy_loss(torch.tensor(logits), torch.tensor(labels), num_logits=nlogits).item()
    
    d = defaultdict(list)
    for _, row in df.iterrows():
        for pos_li, logit_li, label in zip(row['positions'], row['logits'], row['labels']):
            for (i, n), logit in zip(pos_li, logit_li):
                pkey = _rel_pos(i, n, method=method)
                d[pkey].append((logit, label))

    pcounts = {}
    perrors = {}
    for pkey in d:
        plogits, plabels = zip(*d[pkey])
        pcounts[pkey] = len(plabels) 
        perrors[pkey] = _mean_bce_error(plogits, plabels, nlogits=2)

    pos_pkeys, neg_pkeys = _sorted_pkeys(perrors)
    pkeys = pos_pkeys + neg_pkeys
    pcounts = {pkey: pcounts[pkey] for pkey in pkeys}
    perrors = {pkey: perrors[pkey] for pkey in pkeys}
    
    def _get_weights():
        assert len(pkeys) > 0
                
        perror_values = list(perrors.values())
        pcount_values = list(pcounts.values())
        
        ## Normalize errors ##
        mean = np.sum([s * c for s, c in zip(perror_values, pcount_values)]) / sum(pcount_values)
        std = np.sqrt(np.sum([c * (s - mean)**2 for s, c in zip(perror_values, pcount_values)]) / sum(pcount_values))
        
        max_pcount = max(pcount_values)
        lower = max(mean - 3 * std, 0)
        upper = min(mean + 3 * std, 1)
        for i in range(len(perror_values)):
            if count_normalized:    
                pcount_norm = pcount_values[i] / max_pcount
                perror_values[i] = pcount_norm * perror_values[i] + (1 - pcount_norm) * mean 
            
            if perror_values[i] > upper:
                perror_values[i] = upper
            elif perror_values[i] < lower:
                perror_values[i] = lower

        ## Error-complemented normalized weights ##
        pweights = {pkey: 1 - perr for pkey, perr in zip(pkeys, perror_values)}
        pweights, _ = _smoothed_pweights(pweights)
        
        min_, max_ = min(pweights.values()), max(pweights.values())
        if min_ == max_:
            return {pkey: 1.0 for pkey in pweights}
        
        pweights = {pkey: (pweights[pkey] - min_) / (max_ - min_) for pkey in pweights}
        return pweights

    pweights = _get_weights()
    assert pweights.keys() == perrors.keys() == pcounts.keys()

    def _get_lob_pwfn(pweights, k=None):
        if k is not None:
            pweights, max_ = _smoothed_pweights(pweights, k=k)
        else:
            max_ = max(pweights.values())
        # pweights = _smoothed_pweights(pweights, k=k)
        
        def lob_pwfn(i, n, e):
            # e = e or 0
            pkey = _rel_pos(i, n, method=method)
            return e + (1 - e) * pweights.get(pkey, max_)
            # return e + (1 - e) * pweights.get(pkey, 1.0)
        
        return lob_pwfn
     
    _get_lob_pwfn = partial(_get_lob_pwfn, pweights=pweights)
     
    return _get_lob_pwfn, pweights, perrors, pcounts
     

def poly_pwfn(i, n, p, e, k=None):
    def _poly(i, n, k, p):
        assert k > 0
        ri = _rel_pos(i, n, method='sym')
        ri = min(ri, k)
        return 1 - (1 - ri / k) ** p
    k = k if k is not None else max(n // 2, 1)
    return e + (1 - e) * _poly(i, n, k, p)

    
def lin_pwfn(i, n, e, k=None):
    def _lin(i, n, k):
        assert k > 0
        ri = _rel_pos(i, n, method='sym')
        ri = min(ri, k)
        return ri / k
        # return min(ri / k, 1.0)
    k = k if k is not None else max(n // 2, 1)
    return e + (1 - e) * _lin(i, n, k)


class PredictionFactory:
    def __init__(self, log_metrics, save_metrics, expr_metric='f1'):
        self.log_metrics = log_metrics
        self.save_metrics = save_metrics
        self.expr_metric = expr_metric
            
    @property
    def _default_t(self):
        return 0.5
    
    @property
    def _default_r(self):
        return 'mean'
    
    @property
    def _default_w(self):
        return ('uniform', lambda i, n: 1)
    
    @property
    def _default_hp(self):
        return self._hp(self._default_t, self._default_r, self._default_w)
    
    def _hp(self, t, r, w):
        return {'threshold': t, 'reduction': r, 'weight_fn': w}
    
    def _hps(self, ts, rs, ws):
        hps = [{'threshold': t, 'reduction': r, 'weight_fn': w} for t, r, w in itertools.product(ts, rs, ws)]
        return hps
    
    def _explore_ws(self, fn=None, eval_df=None):
        # default
        weight_fns = []
        
        if fn == 'uniform':
            weight_fns.append(self._default_w)
                    
        # linear & polynomial
        elif fn in ['lin', 'poly']:
            es = [0.1]
            ks = [5, 8, 10, 12, None]
            ps = [2]
            for k in ks:
                for e in es:
                    if fn == 'lin':
                        wkey = f'{fn}_k{k}_e{e}' if k is not None else f'{fn}_e{e}'
                        witem = (wkey, partial(lin_pwfn, k=k, e=e))
                        weight_fns.append(witem)
                    else:
                        for p in ps:
                            wkey = f'{fn}_k{k}_p{p}_e{e}' if k is not None else f'{fn}_p{p}_e{e}'
                            witem = (wkey, partial(poly_pwfn, k=k, p=p, e=e))
                            weight_fns.append(witem)
                            if k == 1:
                                break
        
        # loss-based
        elif fn in ['lob', 'lobcn']:
            assert eval_df is not None
            
            es = [0.1]
            
            ks = [5, 8, 10, 12]
            cn = False
            if fn == 'lobcn':
                ks = [5, 10, None]
                cn = True
            
            _get_lob_pwfn, *_ = get_lob_weights(eval_df, method='asym', count_normalized=cn)
            
            for k in ks:
                lob_pwfn_k = _get_lob_pwfn(k=k)
                for e in es:
                    wkey = f'{fn}' + (f'_k{k}' if k is not None else '') + (f'_e{e}' if e is not None else '')
                    witem = (wkey, partial(lob_pwfn_k, e=e))
                    weight_fns.append(witem)
        
        else:
            raise ValueError(f"Invalid weight function: {fn}")
            
        return weight_fns
    
    def _explore_rs(self):
        return ['mean', 'mode']
    
    def _explore_ts(self, t1=0.3, t2=0.7, step=0.01):
        return [t / 1000 for t in range(int(t1 * 1000), int(t2 * 1000) + 1, int(step * 1000))]
    
    def ensemble_predict(self, probs, positions, hps: list, 
                            labels = None, compute_metrics_fn = None, 
                            return_preds: bool = True, log_metrics: bool = True):

        def pos_probs_to_preds(probs, threshold: float):
            """Reduce document sentence probs to sentence preds"""
            return probs_to_preds(probs, n_logits=1, threshold=threshold).tolist()

        def mean_reduce(doc_row, wkey=None):
            """[[0.1, 0.9, 0.5], [0.2, 0.8], [0.3]] -> [0.75, 0.5, 0.3]"""
            doc_probs = doc_row['probs']
            if wkey is None:
                return [np.mean(sent_probs).item() for sent_probs in doc_probs]
            doc_weights = doc_row[wkey]
            return [np.average(sent_probs, weights=sent_weights).item() for sent_probs, sent_weights in zip(doc_probs, doc_weights)]
        
        def mode_reduce(doc_row, predict_fn, wkey=None):
            """[[0.1, 0.9, 0.5], [0.2, 0.8], [0.3]] -> [[0, 1, 1], [0, 1], [0]] -> [0.66, 0.5, 0]"""
            doc_probs = doc_row['probs']
            out = []
            for i, sent_probs in enumerate(doc_probs):
                sent_preds = predict_fn(sent_probs)
                if wkey is None:
                    mode_prob = np.mean(sent_preds).item()
                else:
                    sent_weights = doc_row[wkey][i]
                    mode_prob = np.average(sent_preds, weights=sent_weights).item()
                out.append(mode_prob)
            return out
        
        def positions_to_weights(doc_positions, wfn):
            return [
                [wfn(i=i, n=n) for (i, n) in sent_positions]
                for sent_positions in doc_positions
            ]
        
        def out_util(probs, preds, preds_key, hp, pbar, out, best_hpm):
            # nonlocal out, best_hpm
            def _desc(hp, metrics=None):
                score = (metrics or {}).get(self.expr_metric, 0)
                return f"{hp['weight_fn'][0]}_{hp['reduction']}_{hp['threshold']} [{self.expr_metric}={score:.2f}]"
            
            def better_found(metrics, hp):
                score = metrics.get(self.expr_metric, None)
                if not score:
                    return False
                
                if not best_hpm or (score > best_hpm['metrics'][self.expr_metric]):
                    return True
                
                if score == best_hpm['metrics'][self.expr_metric]:
                    # if metrics['wd_docw'] > best_hpm['metrics']['wd_docw'] or metrics['pk_docw'] > best_hpm['metrics']['pk_docw']:
                    #     return True
                    # if np.abs(hp['threshold'] - 0.5) < np.abs(best_hpm['hp']['threshold'] - 0.5):
                    #     return True
                    pm = 'precision_docw' if self.expr_metric.endswith('docw') else 'precision'
                    rm = 'recall_docw' if self.expr_metric.endswith('docw') else 'recall'
                    if np.abs(metrics.get(pm) - metrics.get(rm)) < np.abs(best_hpm['metrics'][pm] - best_hpm['metrics'][rm]):
                        return True
                return False

            if labels is not None:
                metrics = compute_metrics_fn(labels=labels, predictions=preds)
                if log_metrics:
                    self.log_metrics(preds_key, metrics)
                    
                if better_found(metrics, hp):
                    best_hpm = {
                        'metrics': metrics,
                        'preds_key': preds_key,
                        'hp': hp,
                    }
            else:
                metrics = None
                best_hpm = {
                    'metrics': None,
                    'preds_key': preds_key,
                    'hp': hp,
                }
                
            pbar.set_description(f"{_desc(hp, metrics)} | {_desc(best_hpm['hp'], best_hpm['metrics'])}")
            
            if return_preds:
                assert preds_key.count('|') == 1, preds_key
                probs_key = preds_key.split('|')[1]
                out[probs_key] = probs
                out[preds_key] = preds
            
            return out, best_hpm, metrics

        out = {}
        best_hpm = None
        
        w2r2t = {}
        for hp in hps:
            w = hp['weight_fn']
            w2r2t[w] = w2r2t.get(w, {})
            
            r = hp['reduction']
            w2r2t[w][r] = w2r2t[w].get(r, {})
            
            t = hp['threshold']
            w2r2t[w][r][t] = hp
        
        df = pd.DataFrame({'probs': probs, 'positions': positions})
        pbar = tqdm(total=len(w2r2t), desc='Ensemble Predictions')
        
        for w in sorted(w2r2t, key=lambda x: x[0]):
            # Calculate weights from positions (i, n)
            wkey, wfn = w
            df[wkey] = df['positions'].apply(positions_to_weights, wfn=wfn)
            for r in sorted(w2r2t[w]):
                es_count = 0
                es_patience = 10
                best_t_hp = None
                for t in sorted(w2r2t[w][r]):
                    # hp = self._hp(t, r, (wkey, wfn))
                    hp = w2r2t[w][r][t]
                    
                    # Mean
                    if r == 'mean':
                        probs_mean = df.apply(mean_reduce, axis=1, wkey=wkey)
                        preds_mean = probs_mean.apply(pos_probs_to_preds, threshold=t)
                        preds_key = f'preds_{t}|probs_{r}_{wkey}'
                        out, best_hpm, metrics = out_util(probs_mean, preds_mean, preds_key, hp, pbar, out, best_hpm)
                        
                    # Mode
                    elif r == 'mode':  
                        predict_fn_t = partial(pos_probs_to_preds, threshold=t)
                        probs_mode = df.apply(mode_reduce, axis=1, predict_fn=predict_fn_t, wkey=wkey)
                        preds_mode = probs_mode.apply(pos_probs_to_preds, threshold=0.5)
                        preds_key = f'preds|probs_{r}_{t}_{wkey}'
                        out, best_hpm, metrics = out_util(probs_mode, preds_mode, preds_key, hp, pbar, out, best_hpm)
                    
                    else:
                        raise ValueError(f"Invalid reduction: {r}")
                                                                  
                    # Early stopping
                    if not best_t_hp or ((score := metrics.get(self.expr_metric, None)) and (score > best_t_hp['metrics'][self.expr_metric])):
                        es_count = 0
                        best_t_hp = {
                            'hp': hp,
                            'preds_key': preds_key,
                            'metrics': metrics,
                        }
                    else:
                        es_count += 1
                        if es_count >= es_patience:
                            break
            
            df.drop(columns=[wkey], inplace=True)
            pbar.update()
        
        pbar.close()
        
        return out, best_hpm

    def find_best_hps(self, eval_df, is_seg, expr_name=None, expr_threshold=None):
        _ensemble_predict = partial(self.ensemble_predict,
            compute_metrics_fn = partial(compute_metrics, ignore_last_sentence=is_seg, segeval=False, f1_docw=True),
            return_preds = False, log_metrics = False
        )
        
        def _log_search(ts, rs, ws):
            ts = f"[{ts[0]}, ..., {ts[-1]}]" if len(ts) > 3 else ts
            ws = [w[0] for w in ws]
            ws = f"[{ws[0]}, ..., {ws[-1]}]" if len(ws) > 3 else ws
            print(f"Exploring t={ts}, r={rs}, w={ws} on eval...")
        
        best_hps = []
        if expr_name and 't' in expr_name:
            if expr_threshold is not None:
                t1, t2, step = expr_threshold
                ts = self._explore_ts(t1=t1, t2=t2, step=step)
            else:
                ts = self._explore_ts()
                t1, t2, step = ts[0], ts[-1], ts[1] - ts[0]
            rs = [self._default_r]
            ws = [self._default_w]
            hps = self._hps(ts, rs, ws)
            _log_search(ts, rs, ws)
            
            _, best_hpm = _ensemble_predict(hps=hps, probs=eval_df['probs'], positions=eval_df['positions'], labels=eval_df['labels'])
            best_hps.append(best_hpm['hp'])
            
            if 'r' in expr_name or 'w' in expr_name:
                best_t = best_hpm['hp']['threshold']
                ts = self._explore_ts(t1 = max(best_t - 0.01, t1), 
                                      t2 = min(best_t + 0.01, t2), 
                                      step = step)
                
                if 'r' in expr_name:
                    rs = self._explore_rs()
                
                if 'w' in expr_name:
                    for fn in ['lin', 'poly', 'lob', 'lobcn']:  # 'uniform', 
                        wdf, edf = eval_df, eval_df
                        # if fn == 'lob' or fn == 'lobcn':
                            # n_wdf = min(len(eval_df) // 2, 1000)
                            # wdf, edf = eval_df.iloc[:n_wdf], eval_df.iloc[n_wdf:]
                        # else: 
                            # wdf, edf = None, eval_df
                        ws = self._explore_ws(fn=fn, eval_df=wdf)
                        hps = self._hps(ts, rs, ws)
                        _log_search(ts, rs, ws)
                        
                        _, best_hpm = _ensemble_predict(hps=hps, probs=edf['probs'], positions=edf['positions'], labels=edf['labels'])
                        best_hps.append(best_hpm['hp'])

        if self._default_hp not in best_hps:
            best_hps.insert(0, self._default_hp)
        
        return best_hps
    
    def _predict_and_save(self, raw_df, df, is_seg, labels_provided, hp, docs_save_dir=None, chunks_save_dir=None): #, hpms_save_dir=None):    
        print(f'\n=== Document-grouped inference (ignore_last_sentence: {is_seg}) ===')
        
        pout, hpm = self.ensemble_predict(
            probs = df['probs'],
            positions = df['positions'],
            hps = [hp],
            labels = df['labels'] if labels_provided else None,
            compute_metrics_fn = partial(compute_metrics, ignore_last_sentence=is_seg, segeval=is_seg, f1_docw=True), 
            return_preds = True, 
            log_metrics = True
        )
        
        out_paths = tuple()
        if docs_save_dir:
            pred_col = hpm['preds_key']
            prob_col = hpm['preds_key'].split('|')[1]
                                                            
            df = pd.concat([df, pd.DataFrame(pout)], axis=1)
            docs, docs_save_path = postprocess_predictions(raw_df, df, pred_col=pred_col, prob_col=prob_col, is_seg=is_seg, save_dir=docs_save_dir)
            out_paths += (docs_save_path,)
            
            if is_seg and chunks_save_dir:
                _, chunks_save_path = extract_segments(docs, pred_col=pred_col, prob_col=prob_col, save_dir=chunks_save_dir)
                out_paths += (chunks_save_path,)
            
        return hpm, out_paths
    
    def predict_and_save(self, raw_predict_df, predict_df, 
                         expr_name=None, expr_threshold=None, expr_hpm_path=None, eval_df=None, 
                         docs_save_dir=None, chunks_save_dir=None, hpms_save_dir=None,
                         out_results_kwargs=None):
        
        is_seg = True
        labels_provided = ('labels' in predict_df.columns)
        
        _predict_and_save = partial(self._predict_and_save,
                                    raw_df = raw_predict_df,
                                    df = predict_df,
                                    is_seg = is_seg,
                                    labels_provided = labels_provided,)
        
        # Searching for best hyperparameters
        if expr_name and eval_df is not None:
            best_hps = self.find_best_hps(
                eval_df = eval_df,
                is_seg = is_seg,
                expr_name = expr_name, 
                expr_threshold = expr_threshold
            )

            expr_best_hpm = None
            for best_hp in best_hps:
                hpm, _ = _predict_and_save(hp=best_hp)  #, hpms_save_dir=hpms_save_dir)
                
                if out_results_kwargs and labels_provided:
                    save_out_results(**out_results_kwargs, ens_metrics=hpm)
                    
                if not expr_best_hpm or (hpm['metrics'][self.expr_metric] > expr_best_hpm['metrics'][self.expr_metric]):
                    expr_best_hpm = hpm
            
            if hpms_save_dir and expr_best_hpm:
                hpm_save_path = Path(hpms_save_dir) / f"{hpm['preds_key']}.pkl"
                print(f"Saving best hpm:\n{hpm_save_path}")
                with open(hpm_save_path, 'wb') as f:
                    dill.dump(hpm, f)
             
        # Using best or default hyperparameters       
        else:
            hp = self._default_hp

            if expr_hpm_path is not None:
                with open(expr_hpm_path, 'rb') as f:
                    hp = dill.load(f)['hp']
            
            if expr_threshold is not None:
                hp['threshold'] = expr_threshold

            hpm, _ = _predict_and_save(hp=hp, docs_save_dir=docs_save_dir, chunks_save_dir=chunks_save_dir)

            if out_results_kwargs and labels_provided:
                save_out_results(**out_results_kwargs, ens_metrics=hpm)


### OutResults ###

import os
from datetime import datetime
import json
import socket
import pandas as pd


def save_out_results(output_dir, seed, batch_size, 
                     expr_name,
                     partition_strategy, partition_value, 
                     partition_disable_strategy, partition_disable_value, 
                     split, metrics, ens_metrics,
                     save_path):
    out = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': f'{socket.gethostname()}:{output_dir}',
        
        'expr': expr_name,
        'partition': f'{partition_strategy}-{partition_value}',
        'partition_disable': f'{partition_disable_strategy}-{partition_disable_value}',
        
        'seed': seed,
        'bs': (len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')), batch_size),
    }
    
    # metrics
    runtime = metrics[f'{split}_runtime']
    n_docs = metrics[f'{split}_n_docs']
    n_seqs = metrics[f'{split}_n_seqs']
    n_sents = metrics[f'{split}_n_sents']
    loss = metrics[f'{split}_loss']
    out |= { 
        'samples': (n_docs, n_seqs, n_sents),
        'runtime': f'{runtime:.2f}',
        'dpm': f'{n_docs / runtime:.2f}',
    }
    
    # ens_metrics
    out |= {
        'preds_key': ens_metrics['preds_key'],
        **{k: (v[0] if isinstance(v, tuple) else v) for k, v in ens_metrics['hp'].items()},
        **{k: round(ens_metrics['metrics'][k], 2) for k in ['f1', 'precision', 'recall', 'f1_docw', 'precision_docw', 'recall_docw', 'pk_docw', 'wd_docw'] if k in ens_metrics['metrics']},
    }
    
    # save
    row_df = pd.DataFrame([out])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, row_df], ignore_index=True)
        # df = df.sort_values(by=['row_id'])
    else:
        df = row_df
    
    print(f'Saving the output results into {save_path}')
    df.to_csv(save_path, index=False)
