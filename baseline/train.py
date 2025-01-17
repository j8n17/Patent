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
from transformers import Trainer, TrainingArguments
from preprocess import load_data, split_data, convert_single_label_dataset, add_hierarchical_labels, get_dataset, compute_pos_weights
from sklearn.metrics import f1_score
import math
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW

from peft import get_peft_model, LoraConfig, TaskType

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
    parser.add_argument('--config', default='./config/train.yaml')
    args = parser.parse_args()
    return args

def get_config(args):
    return OmegaConf.load(args.config)

def set_seed(cfg):
    if cfg.setting.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed = cfg.setting.seed

    logger.info(f"set seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_gpu(cfg):
    logger.info(f"CUDA_VISIBLE_DEVICES: {cfg.train.gpu_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.train.gpu_id)

def set_model_name(cfg):
    model_name = cfg.model.name

    if cfg.train.restart_path:
        cfg.model.pretrained_model_name_or_path = cfg.train.restart_path
    elif model_name == 'kobart':
        cfg.model.pretrained_model_name_or_path = 'gogamza/kobart-base-v2'
    elif model_name == 'koelectra':
        cfg.model.pretrained_model_name_or_path = 'monologg/koelectra-small-v3-discriminator'
    elif model_name == 'kopatelectra':
        cfg.model.pretrained_model_name_or_path = './KIPIKorPatELECTRA/KorPatELECTRA/PT'
    else:
        raise ValueError

    logger.info(f'model_path : {cfg.model.pretrained_model_name_or_path}')

def load_tokenizer(cfg):
    logger.info('load tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    if 'kobart' in cfg.model.name:
        tokenizer.bos_token_id = 0
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", tokenizer.bos_token_id), 
                (f"{eos}", tokenizer.eos_token_id)
            ],
        )
    return tokenizer

def load_model(cfg, dataset):
    logger.info('load model...')
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path,
        num_labels = len(dataset['train'][0]['labels'])
    )

    if cfg.train.fine_tune.enable:
        if cfg.model.name == 'kobart':
            N = cfg.train.fine_tune.n_layer
            logger.info(f'kobart classification_head & {N} layers fine-tuning...')

            if N != -1:
                for param in model.parameters():
                    param.requires_grad = False

                # 마지막 N개의 Decoder Layer의 파라미터를 unfreeze
                if N != 0:
                    for layer in model.model.decoder.layers[-N:]:
                        for param in layer.parameters():
                            param.requires_grad = True

                # Classification head의 파라미터를 학습 가능하게 설정
                for param in model.classification_head.parameters():
                    param.requires_grad = True
        
        elif cfg.model.name == 'kopatelectra':
            N = cfg.train.fine_tune.n_layer
            logger.info(f'kopatelectra classifier & {N} layers fine-tuning...')

            for param in model.parameters():
                param.requires_grad = False

            if N != 0:
                # 마지막 N개의 Decoder Layer의 파라미터를 unfreeze
                for layer in model.electra.decoder.layer[-N:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Classification head의 파라미터를 학습 가능하게 설정
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        if cfg.train.fine_tune.lora:
            model = get_lora_model(model)

        def check_freezing(model):
            trainable_weights = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_weights.append(name)
            logger.info(f'trainable_weights: {trainable_weights}')

        check_freezing(model)
    else:
        logger.info('train entire model...')

    return model

def get_lora_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc", "embed_positions"]
    )

    model = get_peft_model(model, peft_config)

    # Classification head의 파라미터를 학습 가능하게 설정
    for param in model.classification_head.parameters():
        param.requires_grad = True

    return model

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

def get_trainer(cfg, model, tokenizer, dataset, pos_weights):
    logger.info(f"build trainer...")

    training_args = TrainingArguments(
        save_strategy='epoch',
        evaluation_strategy='no' if cfg.data.n_fold == 1 else 'steps',
        logging_strategy='steps',
        
        num_train_epochs=cfg.train.epochs,
        eval_steps=0.5/cfg.train.epochs,
        eval_on_start=cfg.train.valid.eval_first,
        eval_accumulation_steps=1280//cfg.train.batch_size if cfg.train.valid.compute_metrics else None,
        logging_steps=0.5/cfg.train.epochs,

        dataloader_num_workers=1,
        dataloader_persistent_workers=True,

        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=math.ceil(cfg.model.num_labels/cfg.train.batch_size) if cfg.train.grad_accumulation else 1,
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
        # 계층적 학습이든 아니든 SSno output만 선택
        labels = pred.label_ids[:, :564]
        outputs = pred.predictions[0][:, :564]

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
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        validation OOM 문제 해결 방법 - https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
        위 링크에서는 argmax를 통해 미리 처리하는 방식으로 메모리 사용률을 줄였지만, 일단 cpu로 보내서 할 수 있는지 테스트하기 위한 코드로 작성했음.
        eval_accumulation_step으로 compute_metrics OOM 문제 해결했으므로 이 함수는 사용중지
        """
        logits = (logits[0].to('cpu'), logits[1].to('cpu'))
        return logits
    

    loss_fn = get_loss_fn(cfg, pos_weights)

    if cfg.train.find_optimal_lr:
        optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate) if cfg.train.optim == "adamw_torch" else None
        scheduler = LambdaLR(optimizer, lambda epoch: 2 ** epoch)

    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset['valid'],
        tokenizer = tokenizer,
        loss_fn=loss_fn,
        optimizers=(optimizer, scheduler) if cfg.train.find_optimal_lr else (None, None),
        compute_metrics=compute_metrics if cfg.train.valid.compute_metrics else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    logger.info(f"trainer: {trainer.args}")
    return trainer

def main(cfg):
    set_seed(cfg)
    set_gpu(cfg)
    set_model_name(cfg)

    tokenizer = load_tokenizer(cfg)

    dataset, category_df = get_dataset(cfg, tokenizer)
    dataset = add_hierarchical_labels(cfg, dataset, category_df, dataset.keys())
    logger.info(dataset)

    pos_weights = compute_pos_weights(dataset)
    
    model = load_model(cfg, dataset)
    trainer = get_trainer(cfg, model, tokenizer, dataset, pos_weights)
    
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