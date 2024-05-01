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
from transformers import Trainer, TrainingArguments
from preprocess import get_dataset

def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/train.yaml')
    args = parser.parse_args()
    return args

def get_config(args):
    return OmegaConf.load(args.config)

def set_seed(cfg):
    if cfg.train.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed = cfg.train.seed

    logger.info(f"set seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_gpu(cfg):
    logger.info(f"CUDA_VISIBLE_DEVICES: {cfg.train.gpu_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.train.gpu_id)

def load_model(cfg):
    logger.info('load model...')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path,
        num_labels = cfg.model.num_labels,
    )
    return tokenizer, model

def formatting_data(dataset, category_df):
    logger.info('formatting dataset...')

    idx_to_SS = category_df.SSno.values
    SS_to_idx = {cat:idx for idx, cat in enumerate(idx_to_SS)}

    def formatting_fn(example):
        title = example['invention_title']
        abstract = example['abstract']
        claims = example['claims']

        texts = f"{title} 요약: {abstract} 청구항: {claims}"
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

def tokenize_data(dataset, tokenizer):
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
    if os.path.isdir(cfg.data.load_from_disk):
        logger.info('loaded from disk!')
        tokenized = load_from_disk(cfg.data.load_from_disk)
        return tokenized
    
    category_df, train_set = get_dataset(cfg, logger)
    train_set = formatting_data(train_set, category_df)
    train_set = tokenize_data(train_set, tokenizer)
    
    return train_set

def split_data(cfg, dataset):
    logger.info('split dataset...')
    dataset = dataset.train_test_split(
        test_size = cfg.data.val_size,
        seed = cfg.data.split_seed,
    )
    return dataset

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn, **kargs):
        super().__init__(*args, **kargs)
        self.loss_fn = loss_fn
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        loss = self.loss_fn(outputs.logits, inputs['labels'])
        return (loss, outputs) if return_outputs else loss

def get_trainer(cfg, model, tokenizer, dataset):
    logger.info(f"build trainer...")

    training_args = TrainingArguments(
        evaluation_strategy='epoch',
        save_strategy='epoch',

        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        optim=cfg.train.optim,
        learning_rate=cfg.train.learning_rate,
        warmup_steps=cfg.train.warmup_steps,

        output_dir=cfg.train.output_dir,
        save_total_limit=cfg.train.save_total_limit,

        report_to = list(cfg.train.report_to),
    )

    loss_fn = get_loss_fn(cfg)
    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        tokenizer = tokenizer,
        loss_fn=loss_fn
    )

    logger.info(f"trainer: {trainer.args}")
    return trainer

def main(cfg):
    global logger
    logger = get_logger()
    set_seed(cfg)
    set_gpu(cfg)

    tokenizer, model = load_model(cfg)

    dataset = load_data(cfg, tokenizer)
    dataset = split_data(cfg, dataset)
    logger.info(dataset)
    
    trainer = get_trainer(cfg, model, tokenizer, dataset)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)