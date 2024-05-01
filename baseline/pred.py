from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import argparse
import logging
import os
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import default_data_collator

def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/pred.yaml')
    args = parser.parse_args()
    return args

def get_config(args):
    return OmegaConf.load(args.config)

def load_data(cfg):
    logger.info('load dataset...')
    category_df = pd.read_csv(cfg.data.category_csv, dtype={'SSno': str})
    test_df = pd.read_csv(cfg.data.test_csv)
    test_set = Dataset.from_pandas(test_df)
    if cfg.data.debug:
        logger.info(f'debug: {cfg.data.debug}')
        test_set = test_set.train_test_split(test_size=0.05, seed=42)['test']
    return test_set, category_df

def load_model(cfg):
    logger.info('load model...')
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path,
    )
    return tokenizer, model

def preprocess_data(cfg, dataset):
    logger.info('preprocess dataset...')

    def preprocess_fn(example):
        title = example['invention_title']
        abstract = example['abstract']
        claims = example['claims']

        texts = f"{title} 요약: {abstract} 청구항: {claims}"

        return {
            'texts': texts,
        }
    
    preprocessed = dataset.map(
        preprocess_fn,
        remove_columns=[
            col
            for col in dataset.column_names
            if col not in ['documentId']
        ],
    )
    return preprocessed

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

def pred(cfg, dataset, model, tokenizer):
    device = cfg.pred.device
    threshold = cfg.pred.threshold

    test_loader = DataLoader(
        dataset,
        batch_size = cfg.pred.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    logger.info('model to device...')
    model.to(device)
    model.eval()

    logger.info('predict...')
    result_ids = []
    result_logits = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            outputs = model(
                input_ids = batch['input_ids'].to(device),
                attention_mask = batch['attention_mask'].to(device),
            )
            result_ids.append(batch['documentId'].numpy())
            result_logits.append(outputs.logits.detach().cpu().numpy())
    
    ids = np.concatenate(result_ids)
    logits = np.concatenate(result_logits)
    preds = (logits > threshold)

    return ids, preds

def save_submission(cfg, ids, preds, category_df):
    idx_to_SSno = category_df.SSno.values
    SSnos = [
        ' '.join(idx_to_SSno[idx] for idx in pred.nonzero()[0])
        for pred in preds
    ]
    submission = pd.DataFrame({
        'documentId': ids,
        'SSnos': SSnos,
    })
    submission.to_csv(cfg.pred.submission_csv, index=False)

def main(cfg):
    dataset, category_df = load_data(cfg)
    tokenizer, model = load_model(cfg)

    dataset = preprocess_data(cfg, dataset)
    dataset = tokenize_data(cfg, dataset, tokenizer)

    ids, preds = pred(cfg, dataset, model, tokenizer)
    save_submission(cfg, ids, preds, category_df)

if __name__ == '__main__':
    global logger
    logger = get_logger()
    args = parse_args()
    cfg = get_config(args)
    main(cfg)
