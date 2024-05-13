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
from tokenizers.processors import TemplateProcessing
from transformers import default_data_collator
from preprocess import tokenize_data, formatting_data, load_data

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

def load_model(cfg):
    logger.info('load model...')
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
    )
    if 'kobart' in cfg.model.name:
        tokenizer.bos_token_id = 0
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor =TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", tokenizer.bos_token_id), 
                (f"{eos}", tokenizer.eos_token_id)
            ],
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path,
    )
    return tokenizer, model

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
    preds = (torch.sigmoid(logits) > threshold)

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
    tokenizer, model = load_model(cfg)

    dataset, category_df = load_data(cfg, tokenizer)

    ids, preds = pred(cfg, dataset, model, tokenizer)
    save_submission(cfg, ids, preds, category_df)

if __name__ == '__main__':
    global logger
    logger = get_logger()
    args = parse_args()
    cfg = get_config(args)
    main(cfg)
