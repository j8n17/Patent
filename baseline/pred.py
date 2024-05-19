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

    logits = torch.from_numpy(logits)

    if cfg.pred.method == 'prob_threshold':
        preds = torch.sigmoid(logits) > threshold

    elif cfg.pred.method == 'argmax':
        max_values = logits.max(dim=1, keepdim=True).values
        preds = logits == max_values

    elif cfg.pred.method == 'auto_ratio_threshold':
        # 특징 : 높은 확률로 예측한 클래스가 있는 데이터인지와 상관없이, multi label이 될 수 있다?
        threshold = 1
        n_single_pred = logits.shape[0] * 0.8476
        max_values = logits.max(dim=1, keepdim=True).values

        while (logits >= threshold * max_values).sum(dim=1).eq(1).sum() > n_single_pred:
            threshold -= 0.0005

        print(f"ratio_threshold : {threshold}")

        preds = logits >= threshold * max_values

    elif cfg.pred.method == 'auto_prob_threshold':
        # single predict 개수에 맞게 예측하도록 threshold를 찾는 경우
        # 특징 : 높은 확률로 예측한 클래스가 있는 데이터가 multi label일 확률이 높다?
        threshold = 1
        n_single_pred = logits.shape[0] * 0.8476

        def count_single_predictions(logits, threshold):
            preds = torch.sigmoid(logits) > threshold
            pred_counts = preds.sum(dim=1)
            return (pred_counts == 0).sum() + (pred_counts == 1).sum()

        while count_single_predictions(logits, threshold) > n_single_pred:
            threshold -= 0.0005

        print(f"prob_threshold : {threshold}")

        preds_prob = torch.sigmoid(logits) > threshold
        max_values = logits.max(dim=1, keepdim=True).values
        preds_argmax = logits == max_values

        preds = preds_prob | preds_argmax

    else:
        raise ValueError

    return ids, preds

def save_submission(cfg, ids, preds, category_df):
    idx_to_SSno = category_df.SSno.values
    SSnos = [
        ' '.join(idx_to_SSno[idx] for idx in pred.nonzero())
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
