Este proyecto implementa un asistente de comandos para Ubuntu que convierte consultas en lenguaje natural a comandos de terminal.

## Características

- Entrenado con BERT en español
- Clasificación de consultas para 30 comandos comunes de Ubuntu
- Interfaz interactiva por línea de comandos
- Framework modular basado en PyTorch Lightning y Hydra

## Estructura

- `data/ubuntu_commands/`: Dataset con ejemplos de consultas y comandos
- `conf/`: Configuraciones modularizadas
  - `model/bert_commands.yaml`: Configuración del modelo
  - `training/ubuntu_assistant.yaml`: Configuración del entrenamiento
- `src/`: Código fuente
  - `model_commands_classification.py`: Modelo para clasificación de comandos
  - `train_commands_classification.py`: Script de entrenamiento
  - `predict_commands_classification.py`: Script de inferencia interactiva
- `outputs/ubuntu_assistant/`: Modelos entrenados (generados durante el entrenamiento)

## Uso

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo
python src/train_commands_classification.py

# Usar el asistente interactivamente
python src/predict_commands_classification.py
