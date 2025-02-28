import os
import hydra
import torch
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.utils import get_original_cwd

from model import SentimentClassifier
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        print(f"Iniciando entrenamiento rápido de {cfg.project.name}")
        
        # Configuración para reproducibilidad
        seed_everything(42)
        
        # Obtener el directorio de trabajo original
        orig_cwd = get_original_cwd()
        
        # Cargar y preparar dataset con ruta absoluta
        csv_path = os.path.join(orig_cwd, "data/mini_sentiment.csv")
        print(f"Cargando dataset desde: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Dataset cargado con {len(df)} ejemplos")
        
        # Establecer valores predeterminados si no están en la configuración
        max_train_samples = getattr(cfg.training, "max_train_samples", 0)
        max_eval_samples = getattr(cfg.training, "max_eval_samples", 0)
        
        # Asegurar que tengamos al menos 2 muestras para entrenamiento y 1 para validación
        total_samples = len(df)
        if total_samples < 3:
            raise ValueError(f"El dataset debe tener al menos 3 ejemplos, pero tiene {total_samples}")
        
        # Limitar muestras para prueba rápida pero garantizar mínimos
        if max_train_samples > 0:
            train_size = min(max(max_train_samples, 2), total_samples - 1)
            eval_size = min(max(max_eval_samples, 1), total_samples - train_size)
        else:
            train_size = max(int(0.8 * total_samples), 2)
            eval_size = total_samples - train_size
        
        print(f"Usando {train_size} muestras para entrenamiento y {eval_size} para validación")
        
        # Cargar tokenizador
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        
        # Crear dataset completo
        dataset = SentimentDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=cfg.model.max_length
        )
        
        # Dividir en train/val
        train_dataset, val_dataset = random_split(
            dataset, [train_size, eval_size]
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(cfg.training.batch_size, train_size),  # Batch size no mayor que nº de ejemplos
            shuffle=True,
            num_workers=4  # Ajustar según el número de núcleos disponibles
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(cfg.training.batch_size, eval_size),  # Batch size no mayor que nº de ejemplos
            shuffle=False,
            num_workers=4  # Ajustar según el número de núcleos disponibles
        )
        
        # Crear modelo
        model = SentimentClassifier(
            model_name=cfg.model.name,
            num_labels=cfg.model.num_labels,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        
        # Configurar callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(orig_cwd, "outputs"),
            filename="mini-sentiment-{epoch:02d}",
            save_top_k=1,
            monitor="val_loss"
        )
        
        # Configurar logger
        logger = TensorBoardLogger(os.path.join(orig_cwd, "logs"), name=cfg.project.name)
        
        # Configurar trainer
        trainer = Trainer(
            max_epochs=cfg.training.epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            log_every_n_steps=cfg.training.logging_steps,
            deterministic=True,
            accelerator="auto",  # Automáticamente detecta CPU/GPU
        )
        
        # Entrenar modelo
        print("Iniciando entrenamiento...")
        trainer.fit(model, train_loader, val_loader)
        
        # Guardar modelo final
        output_dir = os.path.join(orig_cwd, "outputs/final")
        os.makedirs(output_dir, exist_ok=True)
        model.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Entrenamiento completado. Modelo guardado en {output_dir}")
    
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()