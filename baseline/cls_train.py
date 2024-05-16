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

class SequentialUpsamplingSampler(Sampler):
    def __init__(self, data_source, labels):
        self.data_source = data_source
        self.labels = labels
        self.indices = self._make_sequential_indices()

    def _make_sequential_indices(self):
        # todo - 계층적 라벨 학습시 어떤 계층 라벨을 기준으로 업샘플링할 것인지?
        # 원핫 인코딩된 레이블에서 각 클래스의 인덱스 추출
        class_indices = [np.where(self.labels[:, i] == 1)[0] for i in range(self.labels.shape[1])]

        # 가장 많은 데이터를 가진 클래스의 데이터 수
        max_size = max(len(indices) for indices in class_indices)

        # 모든 클래스를 가장 큰 클래스 크기로 업샘플링
        upsampled_indices = [np.random.choice(indices, max_size, replace=True) for indices in class_indices]

        # 각 클래스에서 순차적으로 하나씩 인덱스 추출
        sequential_indices = []
        for idx in range(max_size):
            for class_group in upsampled_indices:
                sequential_indices.append(class_group[idx])

        return sequential_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
class ClsHeadTrainer():
    def __init__(self, cfg, model, dataset):
        logger.info('getting cls head trainer...') 
        self.model = model
        self.epoch = cfg.train.epochs
        dataset.set_format(type="torch", columns=["hidden_vectors", "labels"])
        sampler = SequentialUpsamplingSampler(dataset['train'], dataset['train']['labels'])
        self.train_loader = DataLoader(dataset['train'], batch_size=cfg.train.batch_size, sampler=sampler)
        self.val_loader = DataLoader(dataset['test'], batch_size=cfg.train.batch_size, shuffle=False)
        self.optim = AdamW(self.model.parameters(), lr=cfg.train.learning_rate)
        self.loss_fn = get_loss_fn(cfg, len(dataset['train'][0]['labels']) - 1)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
    
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
        self.validation()
        
        for epoch in range(self.epoch):
            train_loss = 0
            self.model.train()
            for batch in tqdm(self.train_loader):
                self.optim.zero_grad()
                inputs = batch['hidden_vectors'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss = self.compute_loss(self.model, inputs, labels)
                train_loss += loss
                loss.backward()
                self.optim.step()
            print(f'Epoch {epoch + 1}, Training Loss: {train_loss}')

            output_ls, label_ls, val_loss = self.validation()

            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, {self.compute_metrics(output_ls, label_ls)}')
            torch.save(self.model.state_dict(), f'./results/cls_head/model_weights_epoch_{epoch+1}.pth')

    def validation(self, checkpoint_path=None):
        val_loss = 0
        self.model.eval() # Switch to evaluation mode for validation
        with torch.no_grad():
            output_ls = []
            label_ls = []
            for batch in self.val_loader:
                inputs = batch['hidden_vectors'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss, outputs = self.compute_loss(self.model, inputs, labels, return_outputs=True)
                output_ls.append(outputs.cpu().detach().numpy())
                label_ls.append(labels.cpu().detach().numpy())
                val_loss += loss.item()
        output_ls = np.concatenate(output_ls)
        label_ls = np.concatenate(label_ls)

        probalities = torch.sigmoid(torch.from_numpy(output_ls[0])).numpy()
        print(np.mean(probalities), np.var(probalities))
        
        return output_ls, label_ls, val_loss
