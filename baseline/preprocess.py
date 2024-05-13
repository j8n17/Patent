from omegaconf import OmegaConf
import pandas as pd
import argparse
import os
import json
from tqdm.auto import tqdm
from pathlib import Path
import logging
from datasets import Dataset, load_from_disk, DatasetDict
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset

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

    for input_tokens in tqdm(dataloader):
        input_tokens = input_tokens[0].to(device)

        with torch.no_grad():  # 그래디언트 계산 비활성화
            output = base_model(input_tokens).last_hidden_state[:, 0, :].cpu().detach().numpy()
            hidden_vectors.append(output)  # GPU 메모리 해제 및 numpy 배열로 변환
        
        # del input_tokens, output  # 불필요한 텐서 삭제
        # torch.cuda.empty_cache() # GPU 캐시를 비워서 메모리 회수
    hidden_set = tokenized_set.add_column("hidden_vectors", list(np.concatenate(hidden_vectors)))
    hidden_set.save_to_disk(os.path.join(cfg.data.train, cfg.model.name) + '_hidden')
    
    return hidden_set

def make_kfold_indices(cfg):
    logger.info('make kfold indices...')
    '''
    Multi Label의 경우 Stratified KFold를 하기 어렵기 때문에,
    추후에 변경하더라도 일단은
    Multi Label은 KFold로 나누고 Single Label은 Stratified KFold로 나눈 후, 둘을 합쳐 사용한다.
    '''

    df = pd.read_csv(cfg.data.train_csv)

    multi_X = df[df["SSnos"].apply(lambda x: len(x)!=5)].index.to_numpy() # indice
    single_X = df[df["SSnos"].apply(lambda x: len(x)==5)].index.to_numpy()
    single_y = df.loc[single_X, "SSnos"].values

    n_fold = cfg.data.n_fold
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=cfg.data.split_seed)
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.data.split_seed)

    train_test_indices = []
    for (single_train_idx, single_test_idx), (multi_train_idx, multi_test_idx) in zip(skf.split(single_X, single_y), kf.split(multi_X)):
        train_test_indices.append([np.concatenate([single_train_idx, multi_train_idx]), np.concatenate([single_test_idx, multi_test_idx])])

    # NumPy 배열로 변환
    train_test_indices = np.array(train_test_indices, dtype=object)

    # np.save("../data/train/kfold_indices.npy", train_test_indices) # 필요하면 저장 후 분석
    # test = np.load('./train/KFold_indices.npy', allow_pickle=True) # npy load

    return train_test_indices

def split_data(cfg, dataset):
    logger.info('split dataset...')
    # dataset = dataset.train_test_split(
    #     test_size = 1/cfg.data.n_fold,
    #     seed = cfg.data.split_seed,
    # )

    kfold_indices = make_kfold_indices(cfg)
    train_idx, test_idx = kfold_indices[cfg.data.test_fold]
    
    dataset = DatasetDict({
        "train": dataset.select(train_idx),
        "test": dataset.select(test_idx)
    })
    
    return dataset

def main(cfg):
    get_dataset(cfg)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)