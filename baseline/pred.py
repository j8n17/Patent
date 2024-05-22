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
from preprocess import load_data

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

def inference(cfg, dataset, model):
    device = cfg.pred.device

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
            result_logits.append(outputs.logits.detach().cpu()[:, :564]) # 계층적 라벨 학습이든 아니든 SSno 출력만 선택.
    
    ids = np.concatenate(result_ids)
    probs = torch.sigmoid(torch.concat(result_logits))

    return ids, probs

def prediction(cfg, probs):
    threshold = cfg.pred.threshold

    if cfg.pred.method == 'prob_threshold':
        preds = probs > threshold

    elif cfg.pred.method == 'argmax':
        max_values = probs.max(dim=1, keepdim=True).values
        preds = probs == max_values

    elif cfg.pred.method == 'auto_ratio_threshold':
        # 특징 : 높은 확률로 예측한 클래스가 있는 데이터인지와 상관없이, multi label이 될 수 있다?
        threshold = 1
        n_single_pred = probs.shape[0] * 0.8476
        max_values = probs.max(dim=1, keepdim=True).values

        while (probs >= threshold * max_values).sum(dim=1).eq(1).sum() > n_single_pred:
            threshold -= 0.0005

        print(f"ratio_threshold : {threshold}")

        preds = probs >= threshold * max_values

    elif cfg.pred.method == 'auto_prob_threshold':
        # single predict 개수에 맞게 예측하도록 threshold를 찾는 경우
        # 특징 : 높은 확률로 예측한 클래스가 있는 데이터가 multi label일 확률이 높다?
        threshold = 1
        n_single_pred = probs.shape[0] * 0.8476

        def count_single_predictions(probs, threshold):
            preds = probs > threshold
            pred_counts = preds.sum(dim=1)
            return (pred_counts == 0).sum() + (pred_counts == 1).sum()

        while count_single_predictions(probs, threshold) > n_single_pred:
            threshold -= 0.0005

        print(f"prob_threshold : {threshold}")

        preds_prob = probs > threshold
        max_values = probs.max(dim=1, keepdim=True).values
        preds_argmax = probs == max_values

        preds = preds_prob | preds_argmax

    else:
        raise ValueError
    
    return preds

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

    ids, logits = inference(cfg, dataset, model)
    preds = prediction(cfg, logits)
    save_submission(cfg, ids, preds, category_df)

if __name__ == '__main__':
    global logger
    logger = get_logger()
    args = parse_args()
    cfg = get_config(args)
    main(cfg)
