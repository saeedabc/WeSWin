import json
from pathlib import Path
import argparse
from tqdm.auto import tqdm

from data_util.util import sent_tokenize


def prepare_wikisection_splits(data_dir, sent_tokenizer='nltk'):
    def parse_split(data):
        sent_ids = set()
        for i, row in enumerate(tqdm(data)):
            id = row['id']
            title = row['title']
            # abstract = row.get('abstract')
            text = row['text']
            sections = row['annotations']
            
            doc = {
                'id': id,
                'title': title,
                'ids': [],
                'sentences': [],
                'titles_mask': [],
                'labels': [],
            }
            
            for j, sec in enumerate(sections):
                
                sec_title = sec['sectionHeading']
                # sec_label = sec['sectionLabel']

                sec_text = text[sec['begin']:sec['begin']+sec['length']]
                
                sec_sents = [sec_title] + sent_tokenize(sec_text, method=sent_tokenizer)
                for sidx, sent in enumerate(sec_sents):
                    sent_id = f'{id}_{j}_{sidx}'
                    assert sent_id not in sent_ids, f'Duplicate sentence id: {sent_id}'
                    sent_ids.add(sent_id)
                    
                    doc['ids'].append(sent_id)
                    doc['sentences'].append(sent)
                    doc['titles_mask'].append(1 if sidx == 0 else 0)
                    doc['labels'].append(1 if sidx == len(sec_sents) - 1 else 0)
                                
            yield doc

    
    data_dir = Path(data_dir)
    
    data_name = data_dir.name
    assert data_name in ['en_city', 'en_disease']
    
    split_raw_names = {
        'train': f'wikisection_{data_name}_train.json', 
        'eval': f'wikisection_{data_name}_validation.json',
        'predict': f'wikisection_{data_name}_test.json'
    }

    for split, raw_name in split_raw_names.items():
        split_path = data_dir / f"{split}-wt-{sent_tokenizer}.jsonl"
        if split_path.exists():
            print(f'Split file {str(split_path)} already exists. Skipping.')
            continue
        
        raw_path = data_dir / raw_name
        with open(raw_path, "r") as f:
            data = json.load(f)
        
        with open(split_path, 'w', encoding='utf-8') as f:
            print(f'Writing {split} split into {str(split_path)}')
            for i, doc in enumerate(parse_split(data)):
                f.write(json.dumps(doc) + '\n')
                
                # if i == 0:
                #     print(f'First {split} sample:')
                #     for key in doc:
                #         value = doc[key]
                #         if isinstance(value, list) and (_n := len(value)) > 6:
                #             value = f'{str(value[:3])[:-1]}, ..., {str(value[-3:])[1:]}'
                #         print(f'\t{key}: {value}')
        

def load_wikisection_dataset(data_dir=None, data_files=None, cache_dir=None, split=None):
    import datasets 

    if data_dir is not None:
        assert data_files is None
        data_files = {
            'train': str(Path(data_dir) / 'train.jsonl'),
            'eval': str(Path(data_dir) / 'eval.jsonl'),
            'predict': str(Path(data_dir) / 'predict.jsonl'),
        }

    assert split is None or split in data_files

    features = datasets.Features(
        {
            "id": datasets.Value("string"),  # document id --> [doc0, doc1, ...]
            "title": datasets.Value("string"),
            "ids": datasets.Sequence(        # document sentence ids --> [[doc0_sec0_sent0, doc0_sec0_sent1, ...], ...]
                datasets.Value("string")
            ),
            "sentences": datasets.Sequence(
                datasets.Value("string")
            ),
            "titles_mask": datasets.Sequence(
                datasets.Value("uint8")
            ),
            "labels": datasets.Sequence(
                datasets.ClassLabel(num_classes=2, names=['Negative', 'Positive'])
            ),
        }
    )

    dsets = datasets.load_dataset('json', data_files=data_files, cache_dir=cache_dir, features=features, split=split)

    return dsets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sent_tokenizer", type=str, default='nltk')
    args = parser.parse_args()
    
    prepare_wikisection_splits(args.data_dir, sent_tokenizer=args.sent_tokenizer)
