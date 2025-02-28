# src/train_commands_simple.py
import os
import torch
import json
import random
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model_commands import CommandsModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class CommandsDataset(Dataset):
    def __init__(self, queries, commands, tokenizer, max_length_input, max_length_output):
        self.queries = queries
        self.commands = commands
        self.tokenizer = tokenizer
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        command = self.commands[idx]
        
        # Tokenizar entrada y salida
        inputs = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length_input,
            padding="max_length",
            return_tensors="pt"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                command,
                truncation=True,
                max_length=self.max_length_output,
                padding="max_length",
                return_tensors="pt"
            )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

def main():
    print("Iniciando entrenamiento de asistente de comandos Ubuntu")
    
    # Configuración hardcoded
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    max_length_input = 128
    max_length_output = 64
    epochs = 5
    batch_size = 16
    learning_rate = 3e-5
    weight_decay = 0.01
    max_train_samples = 24
    max_eval_samples = 6
    
    # Configuración para reproducibilidad
    seed_everything(42)
    
    # Cargar y preparar dataset
    dataset_path = "data/ubuntu_commands/commands_dataset.json"
    print(f"Cargando dataset desde: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer queries y commands
    queries = [item["query"] for item in data]
    commands = [item["command"] for item in data]
    
    print(f"Dataset cargado con {len(queries)} ejemplos")
    
    # Asegurar que tengamos al menos 2 muestras para entrenamiento y 1 para validación
    total_samples = len(queries)
    if total_samples < 3:
        raise ValueError(f"El dataset debe tener al menos 3 ejemplos, pero tiene {total_samples}")
    
    # Limitar muestras para prueba rápida pero garantizar mínimos
    train_size = min(max(max_train_samples, 2), total_samples - 1)
    eval_size = min(max(max_eval_samples, 1), total_samples - train_size)
    
    print(f"Usando {train_size} muestras para entrenamiento y {eval_size} para validación")
    
    # Mezclar los datos
    indices = list(range(total_samples))
    random.seed(42)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:train_size+eval_size]
    
    train_queries = [queries[i] for i in train_indices]
    train_commands = [commands[i] for i in train_indices]
    
    eval_queries = [queries[i] for i in eval_indices]
    eval_commands = [commands[i] for i in eval_indices]
    
    # Cargar tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Crear datasets
    train_dataset = CommandsDataset(
        train_queries,
        train_commands,
        tokenizer,
        max_length_input=max_length_input,
        max_length_output=max_length_output
    )
    
    eval_dataset = CommandsDataset(
        eval_queries,
        eval_commands,
        tokenizer,
        max_length_input=max_length_input,
        max_length_output=max_length_output
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, train_size),
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        eval_dataset,
        batch_size=min(batch_size, eval_size),
        shuffle=False,
        num_workers=0
    )
    
    # Crear modelo
    model = CommandsModel(
        model_name=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Configurar callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/ubuntu_assistant",
        filename="ubuntu-assistant-{epoch:02d}",
        save_top_k=1,
        monitor="val_loss"
    )
    
    # Configurar logger
    logger = TensorBoardLogger("logs", name="ubuntu_assistant")
    
    # Configurar trainer
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        accelerator="auto",  # Automáticamente detecta CPU/GPU
    )
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    trainer.fit(model, train_loader, val_loader)
    
    # Guardar modelo final
    output_dir = "outputs/ubuntu_assistant/final"
    os.makedirs(output_dir, exist_ok=True)
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Entrenamiento completado. Modelo guardado en {output_dir}")
    
if __name__ == "__main__":
    main()
