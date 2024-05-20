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

def get_dataset(cfg, tokenizer):
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
    train_set = formatting_data(cfg, train_set, category_df)
    train_set = tokenize_data(cfg, train_set, tokenizer)
    return train_set, category_df

def cleaning_data(text):
    text = re.sub(r'[^가-힣A-Za-z0-9,\. ]', '', text)
    return text

def formatting_data(cfg, dataset, category_df=None):
    logger.info('formatting dataset...')
    
    if 'pred' in cfg:
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

    if 'pred' in cfg:
        return tokenized
    
    tokenized.save_to_disk(os.path.join(cfg.data.train, cfg.model.name))
    logger.info('save tokenize dataset for next train!')

    return tokenized

def load_data(cfg, tokenizer):

    logger.info('load dataset...')

    # pred
    if 'pred' in cfg:
        category_df = pd.read_csv(cfg.data.category_csv, dtype={'SSno': str})
        df = pd.read_csv(cfg.data.test_csv)
        dataset = Dataset.from_pandas(df)

        dataset = formatting_data(cfg, dataset, category_df)
        dataset = tokenize_data(cfg, dataset, tokenizer)

        return dataset, category_df
    
    # train
    else:
        if cfg.train.cls_head_only:
            return get_hiddenset(cfg, tokenizer)
        if os.path.isdir(os.path.join(cfg.data.train, cfg.model.name)):
            logger.info('load tokenized set from disk!')
            tokenized_set = load_from_disk(os.path.join(cfg.data.train, cfg.model.name))
            return tokenized_set, None
        return get_dataset(cfg, tokenizer)

def get_hiddenset(cfg, tokenizer):
    '''
    cls head only 학습을 위한 base model의 hidden vector로 이뤄진 dataset
    '''
    tokenizedset_path = os.path.join(cfg.data.train, cfg.model.name)
    hiddenset_path = tokenizedset_path + '_hidden'

    if os.path.isdir(hiddenset_path):
        logger.info('load hidden set from disk!')
        hidden_set = load_from_disk(hiddenset_path)
        return hidden_set, None
    elif os.path.isdir(tokenizedset_path):
        logger.info('load tokenized set from disk!')
        tokenized_set = load_from_disk(os.path.join(cfg.data.train, cfg.model.name))
    else:
        tokenized_set, _ = get_dataset(cfg, tokenizer)
    
    return compute_hidden(cfg, tokenized_set)

def compute_hidden(cfg, tokenized_set):
    logger.info('computing hidden vectors...')
    hidden_vectors = []
    base_model = AutoModel.from_pretrained(cfg.model.pretrained_model_name_or_path)
    dataset = TensorDataset(torch.tensor(tokenized_set['input_ids']))

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    # todo dataset.map으로 바꾸기
    for input_tokens in tqdm(dataloader):
        input_tokens = input_tokens[0].to(device)

        with torch.no_grad():  # 그래디언트 계산 비활성화
            output = base_model(input_tokens).last_hidden_state[:, 0, :].cpu().detach().numpy()
            hidden_vectors.append(output)  # GPU 메모리 해제 및 numpy 배열로 변환
        
        # del input_tokens, output  # 불필요한 텐서 삭제
        # torch.cuda.empty_cache() # GPU 캐시를 비워서 메모리 회수
    hidden_set = tokenized_set.add_column("hidden_vectors", list(np.concatenate(hidden_vectors)))
    hidden_set.save_to_disk(os.path.join(cfg.data.train, cfg.model.name) + '_hidden')
    
    return hidden_set, None

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

    train_test_indices = []
    for train_idx, test_idx in skf.split(X, y):
        train_test_indices.append([train_idx, test_idx])

    # np.save("../data/train/kfold_indices.npy", train_test_indices) # 필요하면 저장 후 분석
    # test = np.load('./train/KFold_indices.npy', allow_pickle=True) # npy load

    return train_test_indices

def split_data(cfg, dataset):
    logger.info('split dataset...')
    kfold_indices = make_kfold_indices(cfg, dataset)
    train_idx, test_idx = kfold_indices[cfg.data.test_fold]
    
    dataset = DatasetDict({
        "train": dataset.select(train_idx),
        "test": dataset.select(test_idx)
    })
    
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

def pass_add_labels(my_dict):
    # Check if the key 'SSno' exists and its value is True
    logger.info('use just SSnos...')
    if my_dict.get('SSno') == True:
        # Check if all other keys have the value False
        for key, value in my_dict.items():
            if key != 'SSno' and value != False:
                return False
        return True
    return False

def add_hierarchical_labels(cfg, dataset):
    if pass_add_labels(cfg.train.hierarchical):
        return dataset
    logger.info('add hierarchical labels...')
    category_df = pd.read_csv('../data/category.csv')
    labels = np.array(dataset['labels'])
    labels_idx = np.argmax(labels, axis=1)

    encoder = OneHotEncoder(sparse_output=False, dtype=bool)

    extended_labels = []
    hierarchical_counts = {}
    if cfg.train.hierarchical['SSno']:
        SSno_onehot = np.array(dataset['labels'])
        extended_labels.append(SSno_onehot)
        hierarchical_counts['SSno'] = SSno_onehot.shape[1]

    for key, value in cfg.train.hierarchical.items():
        if key=='SSno':
            continue
        elif value:
            hierarchical_labels = category_df.loc[labels_idx, key].values
            one_hot = encoder.fit_transform(hierarchical_labels.reshape(-1, 1))
            extended_labels.append(one_hot)
            hierarchical_counts[key] = one_hot.shape[1]

    extended_labels = np.concatenate(extended_labels, axis=1)

    dataset = dataset.remove_columns('labels')
    dataset = dataset.add_column('labels', extended_labels.tolist())
    logger.info(f'hierarchical_labels [name:num] - {hierarchical_counts}')

    return dataset

def main(cfg):
    get_dataset(cfg)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)