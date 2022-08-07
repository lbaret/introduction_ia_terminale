from typing import Any, Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import RobertaModel


class FakeNewsDetector(LightningModule):
    def __init__(self, **lightning_module: Any) -> None:
        super(FakeNewsDetector, self).__init__(**lightning_module)
        self.roberta_encoder = RobertaModel.from_pretrained('roberta-base')
        self.regressor = nn.Linear(in_features=768, out_features=1)

        self._freeze_encoder_parameters()
    
    def _freeze_encoder_parameters(self) -> None:
        for name, params in self.roberta_encoder.named_parameters():
            if 'pooler' in name:
                continue
            
            params.requires_grad = False

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        y = self.roberta_encoder(**x)['pooler_output']
        y = self.regressor(y)
        return y
    
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        input_ids, input_attentions, targets = train_batch
        scores = self.forward({'input_ids': input_ids, 'attention_mask': input_attentions})
        probas = torch.sigmoid(scores).squeeze()
        loss = F.binary_cross_entropy(probas, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, valid_batch, batch_idx) -> torch.Tensor:
        input_ids, input_attentions, targets = valid_batch
        scores = self.forward({'input_ids': input_ids, 'attention_mask': input_attentions})
        probas = torch.sigmoid(scores).squeeze()
        loss = F.binary_cross_entropy(probas, targets)
        self.log('valid_loss', loss)
    
    def test_step(self, test_batch, batch_idx) -> torch.Tensor:
        input_ids, input_attentions, targets = test_batch
        scores = self.forward({'input_ids': input_ids, 'attention_mask': input_attentions})
        probas = torch.sigmoid(scores).squeeze()
        loss = F.binary_cross_entropy(probas, targets)
        self.log('test_loss', loss)
    
    def predict_step(self, predict_batch, batch_idx) -> torch.Tensor:
        input_ids, input_attentions, targets = predict_batch
        scores = self.forward({'input_ids': input_ids, 'attention_mask': input_attentions})
        probas = torch.sigmoid(scores).squeeze()
        return probas >= 0.5, targets
