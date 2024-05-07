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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer
from transformers import Trainer, TrainingArguments
from preprocess import get_dataset, formatting_data, tokenize_data, load_data, split_data
from sklearn.metrics import f1_score
import math

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
    tokenizer = BartTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path) if 'kobart' in cfg.data.save_tokenized_set else AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path,
        num_labels = cfg.model.num_labels,
    )

    if cfg.model.finetune:
        for name, param in model.named_parameters():
            if not any(k in name for k in ['classifier', 'classification_head']):
                param.requires_grad = False
        
        def check_freezing(model):
            trainable_weights = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_weights.append(name)
            logger.info(f'trainable_weights: {trainable_weights}')

        check_freezing(model)
    else:
        logger.info('not fine-tuning...')

    return tokenizer, model

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn, **kargs):
        super().__init__(*args, **kargs)
        self.loss_fn = loss_fn
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        # print(self.tokenizer.decode(inputs['input_ids'][0])) # 모델 입력이 어떤 순서로 나오는지 확인을 위함
        loss = self.loss_fn(outputs.logits, inputs['labels'])
        return (loss, outputs) if return_outputs else loss

def get_trainer(cfg, model, tokenizer, dataset):
    logger.info(f"build trainer...")

    if cfg.train.use_step:
        max_steps=math.ceil(len(dataset['train']) / cfg.train.batch_size) * cfg.train.epochs
        train_fold = cfg.data.n_fold - 1

        training_args = TrainingArguments(
            save_strategy='epoch',
            evaluation_strategy='steps',
            logging_strategy='steps',
            
            max_steps=max_steps,
            eval_steps=int((max_steps//cfg.train.epochs)*2/train_fold),
            logging_steps=int((max_steps//cfg.train.epochs)*2/train_fold),

            per_device_train_batch_size=cfg.train.batch_size,
            per_device_eval_batch_size=cfg.train.batch_size,
            optim=cfg.train.optim,
            learning_rate=cfg.train.learning_rate,
            warmup_steps=max_steps if cfg.train.warmup_steps==-1 else cfg.train.warmup_steps,
            lr_scheduler_type=cfg.train.lr_scheduler_type,

            output_dir=cfg.train.output_dir,
            save_total_limit=cfg.train.save_total_limit,

            report_to = list(cfg.train.report_to),
        )
    else:
        training_args = TrainingArguments(
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',

            num_train_epochs=cfg.train.epochs,
            per_device_train_batch_size=cfg.train.batch_size,
            per_device_eval_batch_size=cfg.train.batch_size,
            optim=cfg.train.optim,
            learning_rate=cfg.train.learning_rate,
            warmup_steps=cfg.train.warmup_steps,
            lr_scheduler_type=cfg.train.lr_scheduler_type,

            output_dir=cfg.train.output_dir,
            save_total_limit=cfg.train.save_total_limit,

            report_to = list(cfg.train.report_to),
        )

    def compute_metrics(pred):
        labels = pred.label_ids
        outputs = pred.predictions

        # [multi cls pred] sigmoid 값 0.5 이상 예측, micro F1-score 계산
        preds = torch.from_numpy(outputs)
        preds = (torch.sigmoid(preds) > 0.5)
        f1_sig05 = f1_score(labels, preds, average='micro')
        
        # [single cls pred] argmax로 가장 높은 클래스로 예측
        preds = outputs.argmax(-1)
        preds_onehot = np.full(outputs.shape, False)
        preds_onehot[np.arange(preds.shape[0]), preds] = True
        f1_argmax = f1_score(labels, preds_onehot, average='micro')
        
        return {
            "f1_sig05": f1_sig05,
            "f1_argmax": f1_argmax
            }

    loss_fn = get_loss_fn(cfg)
    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        tokenizer = tokenizer,
        loss_fn=loss_fn,
        compute_metrics=compute_metrics,
    )

    logger.info(f"trainer: {trainer.args}")
    return trainer

def main(cfg):
    global logger
    logger = get_logger()
    set_seed(cfg)
    set_gpu(cfg)

    tokenizer, model = load_model(cfg)

    dataset, _ = load_data(cfg, tokenizer)
    dataset = split_data(cfg, dataset)
    logger.info(dataset)
    
    trainer = get_trainer(cfg, model, tokenizer, dataset)
    
    if os.path.isdir(cfg.train.checkpoint_path):
        logger.info(f'resume from {cfg.train.checkpoint_path}...')
        trainer.train(resume_from_checkpoint=cfg.train.checkpoint_path)
    else:
        logger.info(f'train new model...')
        trainer.train()
    
    # trainer.train()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
    main(cfg)