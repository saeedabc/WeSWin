import numpy as np
import torch
import torch.nn.functional as F 


def numpyify(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def logits_to_probs(logits: np.ndarray, n_logits: int) -> np.ndarray:
    """Normalize logits to probabilities"""
    logits = numpyify(logits)
        
    logits = torch.from_numpy(logits)
    if n_logits == 1:
        probs = F.sigmoid(logits)
    else:
        probs = F.softmax(logits, dim=-1)
    return probs.numpy()


def probs_to_preds(probs: np.ndarray, n_logits: int, threshold: float = None) -> np.ndarray:
    """Compute binary predictions from probs based on n_logits and threshold"""
    probs = numpyify(probs)
        
    if n_logits == 1:
        threshold = threshold if threshold is not None else 0.5
        preds = (probs >= threshold).astype(np.int32)
    else:
        assert threshold is None
        preds = np.argmax(probs, axis=-1)
    return preds


def logits_to_preds(
    logits: np.ndarray, 
    n_logits: int, 
    normalize: bool = True, 
    threshold: float = None, 
    return_probs: bool = False
):
    """
    Compute binary predictions from logits
    - Logits are normalized to probabilities if normalize is True
    - Use threshold with n_logits=1 (or n_logits=2 if only interested in positive probs)
    - Set return_probs to True to return probs in addition to binary preds
    """
    logits = numpyify(logits)
    
    probs = logits_to_probs(logits, n_logits=n_logits) if normalize else logits
    
    if n_logits == 2 and threshold is not None:
        probs = probs[..., 1]
        n_logits = 1

    preds = probs_to_preds(probs, n_logits=n_logits, threshold=threshold)

    return preds if not return_probs else (preds, probs)


def clean_indices(inputs=None, sep_token_mask=None, labels=None, nd=2):
    assert nd in [1, 2], f"Invalid nd: {nd}"
    assert (sep_token_mask is not None) or (labels is not None), "sep_token_mask or labels must be provided"
    
    def to_native(x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return x
    
    def clean_indices_1d(inputs=None, sep_token_mask=None, labels=None):
        if sep_token_mask is not None:
            indices = [i for i, x in enumerate(sep_token_mask) if x == 1]
        elif labels is not None:
            indices = [i for i, x in enumerate(labels) if x != -100]
        if inputs is not None:
            inputs = to_native(inputs)
            return [inputs[i] for i in indices]
        return indices
    
    if nd == 1:
        return clean_indices_1d(inputs, sep_token_mask, labels)
    elif nd == 2:
        n = len(sep_token_mask) if sep_token_mask is not None else len(labels)
        return [
            clean_indices_1d(inputs[i] if inputs is not None else None, 
                             sep_token_mask=sep_token_mask[i] if sep_token_mask is not None else None, 
                             labels=labels[i] if labels is not None else None) 
            for i in range(n)
        ]


def decode_seq_to_sents(tokenizer, input_ids, sep_token_mask=None, labels=None, return_indices=False):
    assert (sep_token_mask is not None) or (labels is not None), "sep_token_mask or labels must be provided"
    
    sent_sep_token_id = tokenizer.additional_special_tokens_ids[0]
    all_indices = [i for i, x in enumerate(input_ids) if x == sent_sep_token_id]
    
    active_indices = clean_indices(sep_token_mask=sep_token_mask, labels=labels, nd=1)
    
    assert set(active_indices).issubset(set(all_indices))
    
    sents = []
    last_idx = 0
    for idx in all_indices:
        if idx in active_indices:
            sents.append(input_ids[last_idx:idx])
        last_idx = idx
    
    decoded_sents = tokenizer.batch_decode(sents, skip_special_tokens=True)
    
    return decoded_sents if not return_indices else (decoded_sents, active_indices, all_indices)
