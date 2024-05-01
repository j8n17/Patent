from omegaconf import OmegaConf
import pandas as pd
import argparse
import os
import json
from tqdm.auto import tqdm
from pathlib import Path
import logging
from datasets import Dataset

def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/preprocess.yaml')
    args = parser.parse_args()
    return args

def get_config(args):
    return OmegaConf.load(args.config)

def load_raw_data(train_dir):

    def to_dataframe(source_dir):
        dataframes = []
        records = []
        for path in tqdm(sorted(Path(source_dir).glob("*.json"))):
            with open(path, encoding='utf-8') as file:
                data = json.load(file)
                records.extend(data['dataset'])
        df = pd.DataFrame(records)
        return df.fillna('')
    
    docs = to_dataframe(os.path.join(train_dir, 'docs'))
    labels = to_dataframe(os.path.join(train_dir, 'labels'))
    return docs, labels

def get_category_df(labels):
    category_data = labels[['SSno', 'SStext', 'Sno', 'Stext', 'Mno', 'Mtext',
        'Lno', 'Ltext', 'LLno', 'LLtext']].groupby('SSno').first()
    return category_data

def groupby_docId(docs, labels):
    SSnos_by_docId = (
        labels.groupby('documentId').SSno.unique()
        .str.join(' ').rename('SSnos')
    )
    texts_by_docId = docs.groupby('documentId').first()[
        ['invention_title', 'abstract', 'claims']
    ]
    data = pd.concat([texts_by_docId, SSnos_by_docId], axis=1)
    return data

def get_dataset(cfg, logger):
    category_csv = cfg.data.category_csv
    train_csv = cfg.data.train_csv
    if os.path.isfile(category_csv) and os.path.isfile(train_csv):
        pass
    else:
        logger.info(f"save category_df & train_df as csv file...")

        docs, labels = load_raw_data(cfg.data.train)
        category_df = get_category_df(labels)
        train_df = groupby_docId(docs, labels)

        category_df.to_csv(category_csv)
        train_df.to_csv(train_csv)
        
    category_df = pd.read_csv(category_csv, dtype={'SSno': str})
    train_df = pd.read_csv(train_csv)
    train_set = Dataset.from_pandas(train_df)
    return category_df, train_set

def main(cfg):
    global logger
    logger = get_logger()

    get_dataset(cfg, logger)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)