import os
from pathlib import Path
import json
import argparse
from tqdm.auto import tqdm


def prepare_wiki727k_splits(data_dir, data_name='wiki727k', drop_titles=False, save_dir=None, overwrite_cache=False, 
                           max_train_samples=None, max_eval_samples=None, max_predict_samples=None):

    if data_name == 'wiki727k':
        split_raw_names = {'train': 'train', 'eval': 'dev', 'predict': 'test'}
        split_max_samples = {'train': max_train_samples, 'eval': max_eval_samples, 'predict': max_predict_samples}
        glob_pattern = '*/*/*/*'
    elif data_name == 'wiki50':
        split_raw_names = {'predict': 'test'}
        split_max_samples = {'predict': max_predict_samples}
        glob_pattern = '*'
    else:
        raise ValueError(f'Unknown dataset name: {data_name}')

    def _doc_to_sentences(doc: dict):
        def non_empty(s):
            return s and not s.isspace()

        doc_id = doc['id']
        doc_sent_ids = []
        doc_sentences = []
        doc_levels = []
        doc_titles_mask = []
        doc_labels = []

        titles = []
        for sec_idx, section in enumerate(doc['sections']):
            level = section['level']
            title = section['title']
            sentences = [sent for sent in section['sentences'] if non_empty(sent)]

            # Remove irrelevant titles history
            while titles and (last_level := titles[-1][0]) >= level:
                titles.pop()
            
            # Add current title
            titles.append((level, title))
            titles_str = ' '.join([t for l, t in titles if non_empty(t)])

            # Don't keep 'preface' in the titles history 
            if title.lower() == 'preface.' and level == 1:
                titles.pop()

            # If section is empty, continue
            if not sentences:
                continue

            # Add the titles history as a single sentence
            if not drop_titles and non_empty(titles_str):
                doc_sent_ids.append(f'{doc_id}_sec{sec_idx}_title')
                doc_sentences.append(titles_str)
                doc_levels.append(level)
                doc_titles_mask.append(1)
                doc_labels.append(0)

            # Add the sentences
            for sent_idx, sent in enumerate(sentences):
                doc_sent_ids.append(f'{doc_id}_sec{sec_idx}_sent{sent_idx}')
                doc_sentences.append(sent)
                doc_levels.append(level)
                doc_titles_mask.append(0)
                doc_labels.append(1 if sent_idx == len(sentences) - 1 else 0)

        out = {'id': doc_id, 'ids': doc_sent_ids, 'sentences': doc_sentences, 'levels': doc_levels, 'titles_mask': doc_titles_mask, 'labels': doc_labels}
        return out

    def _parse_article(text: str, id: str):
        # Split the text into sections
        sections = text.strip().split("========,")
        
        doc = {
            'id': id, 
            'sections': []
        }
        for section in sections[1:]:  # Skip the first split as it will be empty
            lines = section.split("\n")

            header = lines[0].split(',')
            level, title = header[0].strip(), header[1].strip()

            sentences = [stripped_sent for sent in lines[1:] if (stripped_sent := sent.strip())]

            doc['sections'].append({
                'level': int(level),
                'title': title,
                'sentences': sentences
            })

        return _doc_to_sentences(doc)

    def _read_split_files(data_dir, split):
        def _path_to_id(path, data_dir):
            id = str(path).split(str(data_dir))[1].strip('/')
            id = id.split('.')[0]
            id = '_'.join(id.split('/'))
            return id

        data_dir = Path(data_dir)
        assert data_dir.is_dir()

        split_path = data_dir / split
        for path in tqdm(split_path.glob(glob_pattern), desc=f'{split} - reading Wiki-727K'):
            if not path.is_file():
                continue

            with open(path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            yield _parse_article(
                text=raw_text, 
                id=_path_to_id(path, data_dir)
            )

    data_dir = Path(data_dir)
    save_dir = save_dir or data_dir
    assert Path(save_dir).exists()
    
    for split, max_samples in split_max_samples.items():   
        split_file_name = split + ('-wt' if not drop_titles else '-wot') + (f'-{max_samples}' if max_samples is not None else '') + '.jsonl'
        split_file_path = Path(save_dir) / split_file_name
        
        if not overwrite_cache and split_file_path.exists():
            print(f'Split file {str(split_file_path)} already exists. Skipping.')
            continue

        print(f'Writing {split} split into {str(split_file_path)}')
        with open(split_file_path, 'w', encoding='utf-8') as fout:
            for i, obj in enumerate(_read_split_files(data_dir, split_raw_names[split])):
                if max_samples is not None and i >= max_samples:
                    break
                # if i == 0:
                #     print(f'First {split} sample:\n{obj}\n')
                fout.write(json.dumps(obj) + '\n')


def load_wiki727k_dataset(data_dir=None, data_files=None, cache_dir=None, split=None):
    import datasets 

    if data_dir is not None:
        assert data_files is None
        data_files = {
            'train': str(Path(data_dir) / 'train-wt.jsonl'),
            'eval': str(Path(data_dir) / 'eval-wt.jsonl'),
            'predict': str(Path(data_dir) / 'predict-wt.jsonl'),
        }

    assert split is None or split in data_files

    features = datasets.Features(
        {
            "id": datasets.Value("string"),  # document id --> [doc0, doc1, ...]
            "ids": datasets.Sequence(        # document sentence ids --> [[doc0_sec0_sent0, doc0_sec0_sent1, ...], ...]
                datasets.Value("string")
            ),
            "sentences": datasets.Sequence(
                datasets.Value("string")
            ),
            "levels": datasets.Sequence(
                datasets.Value("uint8")
            ),
            "titles_mask": datasets.Sequence(
                datasets.Value("uint8")
            ),
            "labels": datasets.Sequence(
                # datasets.ClassLabel(num_classes=2, names=['Internal', 'Border'])
                datasets.ClassLabel(num_classes=2, names=['Negative', 'Positive'])
            ),
        }
    )

    return datasets.load_dataset('json', features=features, data_files=data_files, cache_dir=cache_dir, split=split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='wiki727k', help="Name of the dataset to prepare splits for")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where extracted splits lie")
    parser.add_argument('--drop_titles', action='store_true', help="Whether to include titles among sentences")
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    parser.add_argument('--max_predict_samples', type=int, default=None)

    parser.add_argument('--save_dir', type=str, default=None, help="Where to save jsonl splits; defaults to `data_dir`")
    parser.add_argument('--overwrite_cache', action='store_true')

    args = parser.parse_args()

    prepare_wiki727k_splits(data_name=args.data_name, data_dir=args.data_dir, drop_titles=args.drop_titles, save_dir=args.save_dir, overwrite_cache=args.overwrite_cache,
                           max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples, max_predict_samples=args.max_predict_samples)
