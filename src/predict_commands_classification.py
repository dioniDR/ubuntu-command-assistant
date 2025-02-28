# src/predict_commands_classification.py
import os
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_data(model_path):
    """Carga el modelo entrenado, el tokenizador y la lista de comandos"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Cargar la lista de comandos
    with open(os.path.join(model_path, "commands.json"), "r", encoding="utf-8") as f:
        commands = json.load(f)
    
    return model, tokenizer, commands

def predict_command(query, model, tokenizer, commands):
    """Predice un comando a partir de una consulta"""
    # Tokenizar la consulta
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Predecir
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    # Obtener el comando correspondiente
    predicted_command = commands[pred_idx]
    
    return predicted_command, confidence

def main():
    parser = argparse.ArgumentParser(description="Asistente de Comandos Ubuntu (Clasificación)")
    parser.add_argument("--query", type=str, required=False, help="Consulta para generar un comando")
    parser.add_argument("--model_path", type=str, default="./outputs/ubuntu_assistant/final", help="Ruta al modelo entrenado")
    args = parser.parse_args()
    
    # Cargar modelo y datos
    print(f"Cargando modelo desde {args.model_path}...")
    model, tokenizer, commands = load_model_and_data(args.model_path)
    model.eval()
    
    print(f"Modelo cargado. Conoce {len(commands)} comandos.")
    
    if args.query:
        # Predecir para la consulta proporcionada
        command, confidence = predict_command(args.query, model, tokenizer, commands)
        print(f"\nConsulta: '{args.query}'")
        print(f"Comando: {command}")
        print(f"Confianza: {confidence:.2%}")
    else:
        # Modo interactivo
        print("\n=== Asistente de Comandos Ubuntu ===")
        print("Escribe 'salir' para terminar\n")
        
        while True:
            query = input("¿Qué quieres hacer? ")
            if query.lower() in ["salir", "exit", "quit"]:
                break
            
            command, confidence = predict_command(query, model, tokenizer, commands)
            print(f"Comando: {command} (confianza: {confidence:.2%})\n")
    
if __name__ == "__main__":
    main()
