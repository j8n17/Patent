from modules.loss import get_loss_fn

from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import argparse
import logging
import os
import json
from tqdm.auto import tqdm
from pathlib import Path

import random
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tokenizers.processors import TemplateProcessing
from transformers import Trainer, TrainingArguments, default_data_collator
from preprocess import get_dataset, add_hierarchical_labels
from sklearn.metrics import f1_score
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from pred import inference, prediction, load_model
from sklearn.metrics import classification_report

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
    parser.add_argument('--train_config', default='./config/train.yaml')
    parser.add_argument('--pred_config', default='./config/valid.yaml')
    args = parser.parse_args()
    return args

def get_config(args):
    train_cfg = OmegaConf.load(args.train_config)
    pred_cfg = OmegaConf.load(args.pred_config)

    return OmegaConf.merge(train_cfg, pred_cfg)

def set_seed(cfg):
    if cfg.setting.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed = cfg.setting.seed

    logger.info(f"set seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_valset_info(dataset, tokenizer, category_df):
    dir_path = "../analysis/validation"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    valset_info_path = dir_path + "/valset_info.csv"
    if os.path.isfile(valset_info_path):
        return
    logger.info(f'save {valset_info_path}...')

    ids = dataset['documentId']
    input_texts, num_pad = get_input_texts(dataset, tokenizer)
    labels = np.array(dataset['labels'])[:, :564]

    valset_info = pd.DataFrame({
        'documentId': ids,
        'input_text': input_texts,
        'num_pads': num_pad,
        'SSno': label2text('SSno', category_df, labels),
        'Sno': label2text('Sno', category_df, labels),
        'Mno': label2text('Mno', category_df, labels),
        'Lno': label2text('Lno', category_df, labels),
        'LLno': label2text('LLno', category_df, labels),
    })

    valset_info.to_csv(valset_info_path, index=False)

def get_input_texts(dataset, tokenizer):
    encoded_texts = np.array(dataset['input_ids'])
    num_pad = np.sum(encoded_texts == 3, axis=1)
    return tokenizer.batch_decode(encoded_texts, skip_special_tokens=True), num_pad

def label2text(hierarchy, category_df, labels):
    idx_to_class = category_df[hierarchy].values
    classes = [
        ' '.join(idx_to_class[idx] for idx in label.nonzero()[0])
        for label in labels
    ]
    return classes

def save_probs(cfg, ids, probs, columns):
    # 예측된 SSno의 상위 계층으로 상위 계층의 F1 Score을 계산할 것이므로 상위 계층에 대한 확률은 필요 없음.
    logger.info('save pred probs...')
    dir_path = f"../analysis/validation/{cfg.model.pretrained_model_name_or_path.split('/')[-1]}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    probs_path = dir_path + "/probs.csv"

    df = pd.DataFrame(probs, columns=columns)
    df['documentId'] = ids
    df = df[['documentId'] + columns]

    df.to_csv(probs_path, index=False)

def make_columns(cfg, category_df):
    columns = {'SSno': [i for i in category_df["SSno"].unique()]}

    extra_hierarchy = [key for key, value in cfg.train.extra_hierarchy.items() if value]
    for hierarchy in extra_hierarchy:
        columns[hierarchy] = [i for i in category_df[hierarchy].unique()]

    return columns

def save_score_df(cfg, preds, labels, columns, category_df):
    save_path = f"../analysis/validation/{cfg.model.pretrained_model_name_or_path.split('/')[-1]}/score.csv"

    extra_hierarchy = [key for key, value in cfg.train.extra_hierarchy.items() if value]
    num_classes = category_df[["SSno", "Sno", "Mno", "Lno", "LLno"]].nunique().to_dict()

    ssno_preds = preds[:, :564]
    ssno_labels = labels[:, :564]

    report = classification_report(ssno_labels, ssno_preds, target_names=columns['SSno'], output_dict=True)
    all_score_df = pd.DataFrame(report).drop(columns=['macro avg', 'weighted avg', 'samples avg']).rename(columns={'micro avg': 'SSno_micro avg'})

    num_data = preds.shape[0]
    start = num_classes['SSno']
    for hierarchy in extra_hierarchy:
        hierarchy_classes = category_df[hierarchy].unique()
        hierarchy_preds = np.zeros((num_data, num_classes[hierarchy]), dtype=bool)
        # 'SSno' 라벨을 기준으로 'Lno' 라벨을 생성
        for i in range(num_data):
            hierarchy_preds[i, np.searchsorted(hierarchy_classes, category_df.loc[ssno_labels[i], hierarchy].values)] = True

        end = start + num_classes[hierarchy]
        hierarchy_labels = labels[:, start:end]
        start = end
        report = classification_report(hierarchy_labels, hierarchy_preds, target_names=columns[hierarchy], output_dict=True)
        hierarchy_score_df = pd.DataFrame(report).drop(columns=['macro avg', 'weighted avg', 'samples avg']).rename(columns={'micro avg': f'{hierarchy}_micro avg'})
        all_score_df = pd.concat([all_score_df, hierarchy_score_df], axis=1)

    all_score_df['hierarchy'] = ['SSno'] + extra_hierarchy
    # 'micro'가 포함된 열들을 앞으로 이동
    micro_cols = [col for col in all_score_df.columns if 'micro' in col]
    other_cols = [col for col in all_score_df.columns if 'micro' not in col]
    all_score_df = all_score_df[['hierarchy'] + micro_cols + other_cols]

    all_score_df.to_csv(save_path, index=False)

def main(cfg):
    set_seed(cfg)

    tokenizer, model = load_model(cfg)

    dataset, category_df = get_dataset(cfg, tokenizer)
    dataset = add_hierarchical_labels(cfg, dataset, category_df)
    dataset = dataset['valid']
    logger.info(dataset)

    save_valset_info(dataset, tokenizer, category_df)
    cols = make_columns(cfg, category_df)
    
    ids, probs = inference(cfg, dataset, model)
    save_probs(cfg, ids, probs, cols['SSno'])

    preds = prediction(cfg, probs) # True, False로 pred
    labels = np.array(dataset['labels'])

    save_score_df(cfg, preds, labels, cols, category_df)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)