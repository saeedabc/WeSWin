from sklearn.metrics import confusion_matrix
from collections import defaultdict
from transformers.trainer_utils import EvalPrediction

from eval_util.util import logits_to_preds, clean_indices


##########################
### Evaluation Metrics ###
##########################


def compute_segeval_metrics(labels, predictions):
    import segeval

    def segment_sizes(boundary_list):
        """
        Convert a list of binary boundary indicators to a list of segment sizes.
        :param boundary_list: A list of binary values where 1 indicates a segment boundary
        :return: A list of segment sizes
        """
        sizes = []
        current_size = 0
        for boundary in boundary_list:
            current_size += 1
            if boundary == 1:
                sizes.append(current_size)
                current_size = 0
        # Add the last segment if the list does not end with a boundary
        if current_size > 0:
            sizes.append(current_size)
        return sizes
    
    assert len(labels) == len(predictions)
    
    metrics = {'docw': defaultdict(float)}
    for label, prediction in zip(labels, predictions):
        assert len(label) == len(prediction)
        
        # Convert boundary lists to segment sizes
        label_ss = segment_sizes(label)
        prediction_ss = segment_sizes(prediction)
        
        # Compute the p_k and windowdiff metrics
        try:
            assert sum(label_ss) == sum(prediction_ss)
            
            pk_score = float(segeval.pk(label_ss, prediction_ss))
            wd_score = float(segeval.window_diff(label_ss, prediction_ss))
        except Exception as e:
            print('Error in segeval (continue):', e)
            # pk_score, wd_score = None, None
            continue

        metrics['docw']['pk'] += pk_score
        metrics['docw']['wd'] += wd_score
        metrics['docw']['total'] += 1
    
    for key in list(metrics.keys()):
        metrics[f'pk_{key}'] = 100 * (metrics[key]['pk'] / metrics[key]['total'])
        metrics[f'wd_{key}'] = 100 * (metrics[key]['wd'] / metrics[key]['total'])
        del metrics[key]

    return metrics


LABELS = [0, 1]

def compute_f1_metrics(labels, predictions, nd=2, ignore_last_sentence=False, f1_docw=False):
    assert nd in [1, 2]
    assert len(labels) == len(predictions)

    if nd == 1:
        labels = [labels]
        predictions = [predictions]
        
    def get_metrics(tn, fp, fn, tp):
        if (tp + fp) == 0 or (tp + fn) == 0:
            return {}
        return {
            'accuracy': 100 * (tp + tn) / (tp + tn + fp + fn),
            'precision': 100 * tp / (tp + fp),
            'recall': 100 * tp / (tp + fn),
            'f1': 100 * (2 * tp) / (2 * tp + fp + fn),
        }
    
    if not f1_docw: 
        labels = [l for label in labels for i, l in enumerate(label) if (i < len(label) - 1 or not ignore_last_sentence)]
        predictions = [p for prediction in predictions for i, p in enumerate(prediction) if (i < len(prediction) - 1 or not ignore_last_sentence)]
        assert len(labels) == len(predictions)
        assert labels and predictions, 'No labels or predictions to compute metrics'
        
        tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions, labels=LABELS).ravel()
        return get_metrics(tn, fp, fn, tp)
    
    else:
        tns, fps, fns, tps = 0, 0, 0, 0
        dmetrics = defaultdict(list)
        for label, prediction in zip(labels, predictions):
            assert len(label) == len(prediction)
            label = [l for i, l in enumerate(label) if (i < len(label) - 1 or not ignore_last_sentence)]
            prediction = [p for i, p in enumerate(prediction) if (i < len(prediction) - 1 or not ignore_last_sentence)]
            
            if label and prediction:
                tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=prediction, labels=LABELS).ravel()
                tns += tn
                fps += fp
                fns += fn
                tps += tp
                
                for key, value in get_metrics(tn, fp, fn, tp).items():
                    dmetrics[key].append(value)
        
        metrics = get_metrics(tns, fps, fns, tps)
        for key, values in dmetrics.items():
            metrics[f'{key}_docw'] = sum(values) / len(values)

        return metrics


def compute_metrics(labels, predictions, ignore_last_sentence=False, segeval=False, f1_docw=False):
    """Compute all evaluation metrics from cleaned predictions and labels"""
    
    metrics = {
        # 'n_seqs': len(labels),
        'n_sents': sum(len(l) for l in labels),
    }
    
    metrics |= compute_f1_metrics(labels, predictions, nd=2, ignore_last_sentence=ignore_last_sentence, f1_docw=f1_docw)
    
    if segeval:
        metrics |= compute_segeval_metrics(labels, predictions)
    
    return metrics
        

def compute_metrics_token_classification(eval_preds: EvalPrediction, n_logits: int, threshold: float = None, ignore_last_sentence=False, segeval=False):  # TODO: could remove ignore_last_sentence and segeval
    """Compute evaluation metrics from logits and labels for token classification tasks"""
    
    logits = eval_preds.predictions
    labels = eval_preds.label_ids

    predictions = logits_to_preds(logits, n_logits=n_logits, threshold=threshold)
    
    assert labels.shape == predictions.shape and len(labels.shape) == 2, f'labels.shape: {labels.shape}, predictions.shape: {predictions.shape}'
    
    true_labels = clean_indices(inputs=labels, labels=labels, nd=2)
    true_predictions = clean_indices(inputs=predictions, labels=labels, nd=2)

    return compute_metrics(labels=true_labels, predictions=true_predictions, ignore_last_sentence=ignore_last_sentence, segeval=segeval)
