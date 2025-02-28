# src/model_commands_classification.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class CommandsClassificationModel(pl.LightningModule):
    def __init__(
        self, 
        model_name,
        num_commands,
        command_list,
        learning_rate=5e-5, 
        weight_decay=0.01,
        warmup_ratio=0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Lista de comandos disponibles
        self.command_list = command_list
        
        # Cargar modelo pre-entrenado para clasificación
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_commands
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Calcular precisión
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["labels"]).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Configurar optimizador
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Calcular steps totales
        train_loader = self.trainer.train_dataloader
        if train_loader is not None:
            total_steps = (
                len(train_loader.dataset) // 
                (self.trainer.accumulate_grad_batches * train_loader.batch_size) *
                self.trainer.max_epochs
            )
            
            warmup_steps = int(total_steps * self.hparams.warmup_ratio)
            
            # Configurar scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        return optimizer
    
    def predict_command(self, query, tokenizer):
        """Predice un comando a partir de una consulta"""
        # Tokenizar la consulta
        inputs = tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Predecir
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        # Obtener el comando correspondiente
        predicted_command = self.command_list[pred_idx]
        
        return predicted_command, confidence
