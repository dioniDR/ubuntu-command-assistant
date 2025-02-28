# src/predict_commands.py
import os
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from model_commands import CommandsModel

def load_model(model_path):
    """Carga el modelo entrenado y el tokenizador"""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_command(query, model, tokenizer):
    """Predice un comando a partir de una consulta"""
    # Tokenizar la consulta
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generar comando
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )
    
    # Decodificar el comando
    command = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return command

def main():
    parser = argparse.ArgumentParser(description="Asistente de Comandos Ubuntu")
    parser.add_argument("--query", type=str, required=False, help="Consulta para generar un comando")
    parser.add_argument("--model_path", type=str, default="./outputs/ubuntu_assistant/final", help="Ruta al modelo entrenado")
    args = parser.parse_args()
    
    # Cargar modelo
    print(f"Cargando modelo desde {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    model.eval()
    
    if args.query:
        # Predecir para la consulta proporcionada
        command = predict_command(args.query, model, tokenizer)
        print(f"\nConsulta: '{args.query}'")
        print(f"Comando: {command}")
    else:
        # Modo interactivo
        print("\n=== Asistente de Comandos Ubuntu ===")
        print("Escribe 'salir' para terminar\n")
        
        while True:
            query = input("¿Qué quieres hacer? ")
            if query.lower() in ["salir", "exit", "quit"]:
                break
            
            command = predict_command(query, model, tokenizer)
            print(f"Comando: {command}\n")
    
if __name__ == "__main__":
    main()
