# src/train_commands_classification.py
import os
import torch
import json
import random
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model_commands_classification import CommandsClassificationModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class CommandsClassificationDataset(Dataset):
    def __init__(self, queries, command_indices, tokenizer, max_length):
        self.queries = queries
        self.command_indices = command_indices
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        command_idx = self.command_indices[idx]
        
        # Tokenizar entrada
        inputs = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(command_idx, dtype=torch.long)
        }

def main():
    print("Iniciando entrenamiento de asistente de comandos Ubuntu (Clasificación)")
    
    # Configuración hardcoded
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    max_length = 128
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
    
    # Crear una lista única de comandos para clasificación
    unique_commands = sorted(list(set(commands)))
    cmd_to_idx = {cmd: idx for idx, cmd in enumerate(unique_commands)}
    command_indices = [cmd_to_idx[cmd] for cmd in commands]
    
    print(f"Número de comandos únicos: {len(unique_commands)}")
    
    # Asegurar que tengamos al menos 2 muestras para entrenamiento y 1 para validación
    total_samples = len(queries)
    if total_samples < 3:
        raise ValueError(f"El dataset debe tener al menos 3 ejemplos, pero tiene {total_samples}")
    
    # Limitar muestras para prueba rápida pero garantizar mínimos
    train_size = min(max(max_train_samples, 2), total_samples - 1)
    eval_size = min(max(max_eval_samples, 1), total_samples - train_size)
    
    print(f"Usando {train_size} muestras para entrenamiento y {eval_size} para validación")
    
    # Mezclar los datos manteniendo correspondencia entre queries y command_indices
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:train_size+eval_size]
    
    train_queries = [queries[i] for i in train_indices]
    train_cmd_indices = [command_indices[i] for i in train_indices]
    
    eval_queries = [queries[i] for i in eval_indices]
    eval_cmd_indices = [command_indices[i] for i in eval_indices]
    
    # Cargar tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Crear datasets
    train_dataset = CommandsClassificationDataset(
        train_queries,
        train_cmd_indices,
        tokenizer,
        max_length=max_length
    )
    
    eval_dataset = CommandsClassificationDataset(
        eval_queries,
        eval_cmd_indices,
        tokenizer,
        max_length=max_length
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
    model = CommandsClassificationModel(
        model_name=model_name,
        num_commands=len(unique_commands),
        command_list=unique_commands,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Crear directorios para outputs
    os.makedirs("outputs/ubuntu_assistant", exist_ok=True)
    
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
    
    # Guardar la lista de comandos
    with open(os.path.join(output_dir, "commands.json"), "w", encoding="utf-8") as f:
        json.dump(unique_commands, f, ensure_ascii=False, indent=2)
    
    print(f"Entrenamiento completado. Modelo guardado en {output_dir}")
    print(f"Número de comandos clasificados: {len(unique_commands)}")
    
if __name__ == "__main__":
    main()
