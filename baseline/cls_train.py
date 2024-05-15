import logging
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from modules.loss import get_loss_fn
import numpy as np
from sklearn.metrics import f1_score

def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger

logger = get_logger()

import torch
from torch.utils.data import Sampler
import numpy as np

class ClsHeadTrainer():
    def __init__(self, cfg, model, dataset):
        logger.info('getting cls head trainer...') 
        self.model = model
        self.epoch = cfg.train.epochs
        # sampler = SequentialUpsamplingSampler(dataset, dataset['labels'])
        dataset.set_format(type="torch", columns=["hidden_vectors", "labels"])
        self.train_loader = DataLoader(dataset['train'], batch_size=cfg.train.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset['test'], batch_size=cfg.train.batch_size, shuffle=False)
        self.optim = AdamW(self.model.parameters(), lr=1e-5)
        self.loss_fn = get_loss_fn(cfg)
    
    def compute_loss(self, model, inputs, labels, return_outputs=False):
        outputs = model(inputs)
        loss = self.loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, outputs, labels):
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
    
    def train(self, resume_from_checkpoint=None):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        for epoch in range(self.epoch):
            train_loss = 0
            self.model.train()
            for batch in tqdm(self.train_loader):
                self.optim.zero_grad()
                inputs = batch['hidden_vectors'].to(device)
                labels = batch['labels'].to(device)
                loss = self.compute_loss(self.model, inputs, labels)
                train_loss += loss
                loss.backward()
                self.optim.step()
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss}')

            val_loss = 0
            self.model.eval() # Switch to evaluation mode for validation
            with torch.no_grad():
                output_ls = []
                label_ls = []
                for batch in self.val_loader:
                    inputs = batch['hidden_vectors'].to(device)
                    labels = batch['labels'].to(device)
                    loss, outputs = self.compute_loss(self.model, inputs, labels, return_outputs=True)
                    output_ls.append(outputs.cpu().detach().numpy())
                    label_ls.append(labels.cpu().detach().numpy())
                    val_loss += loss.item()
            output_ls = np.concatenate(output_ls)
            label_ls = np.concatenate(label_ls)

            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, {self.compute_metrics(output_ls, label_ls)}')
            torch.save(self.model.state_dict(), f'./results/cls_head/model_weights_epoch_{epoch+1}.pth')
