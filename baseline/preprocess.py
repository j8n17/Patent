from omegaconf import OmegaConf
import pandas as pd
import argparse
import os
import json
from tqdm.auto import tqdm
from pathlib import Path
import logging
from datasets import Dataset, load_from_disk, DatasetDict, concatenate_datasets, Features
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger

logger = get_logger()

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

def make_dataset(cfg, tokenizer):
    logger.info('make dataset...')
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

    category_df = pd.read_csv(category_csv, dtype=str)
    train_df = pd.read_csv(train_csv)
    train_set = Dataset.from_pandas(train_df)
    train_set = formatting_data(cfg, train_set, category_df)
    train_set = tokenize_data(cfg, train_set, tokenizer)
    return train_set, category_df

def cleaning_data(text):
    text = re.sub(r'[^가-힣A-Za-z0-9,\. ]', '', text)
    return text

def formatting_data(cfg, dataset, category_df=None):
    logger.info('formatting dataset...')

    # train & valid
    if 'train' in cfg:
        idx_to_SS = category_df.SSno.values
        SS_to_idx = {cat:idx for idx, cat in enumerate(idx_to_SS)}

        def formatting_fn(example):
            title = example['invention_title']
            abstract = example['abstract']
            claims = example['claims']

            texts = f"{title} 요약: {abstract} 청구항: {claims}"
            texts = cleaning_data(texts)
            labels = np.zeros(len(SS_to_idx), dtype=np.bool_)

            for SSno in example['SSnos'].split():
                labels[SS_to_idx[SSno]] = 1

            return {
                'texts': texts,
                'labels': labels,
            }
        
        formatted = dataset.map(
            formatting_fn,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ['documentId']
            ],
        )
        return formatted
    # pred
    else:
        def formatting_fn(example):
            title = example['invention_title']
            abstract = example['abstract']
            claims = example['claims']

            texts = f"{title} 요약: {abstract} 청구항: {claims}"
            texts = cleaning_data(texts)

            return {
                'texts': texts,
            }
        
        formatted = dataset.map(
            formatting_fn,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ['documentId']
            ],
        )

        return formatted

def tokenize_data(cfg, dataset, tokenizer):
    logger.info('tokenize dataset...')
    def batch_tokenize(batch):
        return tokenizer(
            batch,
            max_length=512,
            padding='max_length',
            truncation=True,
        )
    tokenized = dataset.map(
        batch_tokenize,
        input_columns='texts',
        batched=True,
    )

    return tokenized

def load_data(cfg, tokenizer):
    logger.info('load dataset...')

    # train & valid
    if 'train' in cfg:
        return make_dataset(cfg, tokenizer)

    # pred
    else:
        category_df = pd.read_csv(cfg.data.category_csv, dtype=str)
        df = pd.read_csv(cfg.data.test_csv)
        dataset = Dataset.from_pandas(df)

        dataset = formatting_data(cfg, dataset, category_df)
        dataset = tokenize_data(cfg, dataset, tokenizer)

        return dataset, category_df
    
def get_dataset(cfg, tokenizer=None):
    dataset_path = os.path.join(cfg.data.train, cfg.model.name)

    if os.path.isdir(dataset_path):
        logger.info('load dataset from disk!')
        dataset = load_from_disk(dataset_path)
        category_df = pd.read_csv(cfg.data.category_csv, dtype=str)
        return dataset, category_df
    
    dataset, category_df = load_data(cfg, tokenizer)
    dataset = convert_single_label_dataset(dataset)
    dataset = split_data(cfg, dataset)
    
    dataset.save_to_disk(dataset_path)
    logger.info('save dataset for next train!')

    return dataset, category_df

def make_kfold_indices(cfg, dataset):
    logger.info('make kfold indices...')
    '''
    Multi Label의 경우 Stratified KFold를 하기 어렵기 때문에,
    추후에 변경하더라도 일단은
    Multi Label은 KFold로 나누고 Single Label은 Stratified KFold로 나눈 후, 둘을 합쳐 사용한다.
    '''
    X = np.arange(len(dataset))
    labels = np.array(dataset['labels'])
    y = np.argmax(labels, axis=1)

    n_fold = cfg.data.n_fold
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.data.split_seed)

    kfold_indices = []
    ros = RandomOverSampler(random_state=cfg.setting.seed)
    for train_idx, valid_idx in skf.split(X, y):
        kfold_indices.append([upsample(ros, train_idx, y) if cfg.data.upsampling else train_idx, valid_idx])

    # np.save("../data/train/kfold_indices.npy", kfold_indices) # 필요하면 저장 후 분석
    # test = np.load('./train/KFold_indices.npy', allow_pickle=True) # npy load

    return kfold_indices

def upsample(ros, train_idx, labels):
    labels = labels[train_idx]
    train_idx = train_idx.reshape(-1, 1)

    resampled_train_idx, _ = ros.fit_resample(train_idx, labels)

    return resampled_train_idx.flatten()

def compute_pos_weights(dataset):
    logger.info('compute pos_weights...')
    train_size = len(dataset['train'])
    train_labels = np.array(dataset['train']['labels'])

    pos_counts = train_labels.sum(axis=0)
    neg_counts = np.ones_like(pos_counts) * train_size - pos_counts
    pos_weights = neg_counts / pos_counts

    return pos_weights

def split_data(cfg, dataset):
    logger.info('split dataset...')
    kfold_indices = make_kfold_indices(cfg, dataset)
    train_idx, valid_idx = kfold_indices[cfg.data.valid_fold]
    
    dataset = DatasetDict({
        "train": dataset.select(train_idx),
        "valid": dataset.select(valid_idx)
    })

    dataset = convert_multi_label_dataset(dataset)

    return dataset

def convert_multi_label_dataset(dataset):
    for key in dataset.keys():
        logger.info(f'convert to Multi Label Dataset for {key} dataset...')
        combined_data = {}
        
        for entry in dataset[key]:
            doc_id = entry['documentId']
            if doc_id not in combined_data:
                combined_data[doc_id] = entry.copy()
            else:
                combined_data[doc_id]['labels'] = np.logical_or(combined_data[doc_id]['labels'], entry['labels']).tolist()

        # Create the new dataset
        new_dataset = []
        for entry in combined_data.values():
            new_dataset.append(entry)

        new_dataset = Dataset.from_dict({key: [d[key] for d in new_dataset] for key in new_dataset[0]})

        dataset[key] = new_dataset
        
    return dataset

def convert_single_label_dataset(dataset):
    """Multi Label Dataset을 Single Label Dataset으로 변환."""
    logger.info('convert to Single Label Dataset...')
    expand_dataset = {feature: [] for feature in dataset.features}

    # 멀티 라벨 데이터 확장
    labels = np.array(dataset['labels'])
    sums = np.sum(labels, axis=1)
    remove_indices = np.where(sums != 1)[0]
    remain_indices = np.where(sums == 1)[0]

    num_class = len(labels[0])
    for i in remove_indices.tolist():
        example = dataset[i]
        labels = example['labels']
        for i, label in enumerate(labels):
            if label == True:
                for feature in dataset.features:
                    if feature == 'labels':
                        new_labels = [False] * num_class
                        new_labels[i] = True
                        expand_dataset[feature].append(new_labels)
                    else:
                        expand_dataset[feature].append(example[feature])

    expand_dataset = Dataset.from_dict(expand_dataset, features=Features(dataset.features))
    dataset = dataset.select(remain_indices)
    
    return concatenate_datasets([dataset, expand_dataset])

def add_hierarchical_labels(cfg, dataset, category_df, target_dataset, use_all=False):
    if use_all:
        extra_hierarchy = list(cfg.train.extra_hierarchy.keys())
    else:
        extra_hierarchy = [key for key, value in cfg.train.extra_hierarchy.items() if value]

    if not extra_hierarchy:
        logger.info('use just SSnos...')
        return dataset
    
    logger.info(f'extra hierarchical labels - {extra_hierarchy}')
    
    for key in target_dataset:
        logger.info(f'add extra hierarchical labels for {key} dataset...')
        labels = np.array(dataset[key]['labels'])

        num_classes = category_df[["SSno", "Sno", "Mno", "Lno", "LLno"]].nunique().to_dict()

        extra_labels = []
        for idx in labels:
            entry_df = category_df.loc[idx]
            extra_label = []
            for hierarchy in extra_hierarchy:
                onehot = np.full(num_classes[hierarchy], False)
                indices = np.searchsorted(category_df[hierarchy].unique(), entry_df[hierarchy].values)
                onehot[indices] = True
                extra_label.append(onehot)
            extra_labels.append(np.concatenate(extra_label))
        extra_labels = np.array(extra_labels)

        extended_labels = np.concatenate([labels, extra_labels], axis=1)
        new_dataset = dataset[key].remove_columns('labels')
        dataset[key] = new_dataset.add_column('labels', extended_labels.tolist())

    return dataset

def main(cfg):
    make_dataset(cfg)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)