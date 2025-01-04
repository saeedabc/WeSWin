from functools import partial
import datasets

from data_util.wiki727k_dataset import load_wiki727k_dataset
from data_util.wikisection_dataset import load_wikisection_dataset


def load_custom_dataset(dataset_log_name, dataset_name=None, train_file=None, eval_file=None, predict_file=None):
    
    data_kwargs = {}
    if dataset_name is not None:  
        data_kwargs['data_dir'] = dataset_name
    else:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if eval_file is not None:
            data_files["eval"] = eval_file
        if predict_file is not None:
            data_files["predict"] = predict_file
        data_kwargs['data_files'] = data_files
    
    if dataset_log_name in ['wiki727k', 'wiki50']:
        dsets = load_wiki727k_dataset(**data_kwargs)
    
    elif dataset_log_name in ['en_city', 'en_disease']:
        dsets = load_wikisection_dataset(**data_kwargs)

    else:
        raise ValueError(f'Unknown dataset log name: {dataset_log_name}')

    return dsets


def concatenate_datasets(dset1, dset2):
    if len(dset2) == 0:
        print('Warning: dataset #1 is empty!')
        return dset1
    
    if len(dset1) == 0:
        print('Warning: dataset #2 is empty!')
        return dset2
    
    keep_columns = set(dset1.column_names) & set(dset2.column_names)

    def remove_columns(dataset):
        return dataset.remove_columns([col for col in dataset.column_names if col not in keep_columns])

    dset1 = remove_columns(dset1)
    dset2 = remove_columns(dset2)

    dset = datasets.concatenate_datasets([dset1, dset2])
    assert set(dset.column_names) == keep_columns

    return dset