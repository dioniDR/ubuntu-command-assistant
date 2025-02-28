# src/predict.py
import os
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    """Carga el modelo entrenado y el tokenizador"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Predice el sentimiento de un texto dado"""
    # Tokenizar el texto
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    
    # Obtener predicción
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Obtener etiqueta y confianza
    label_id = predictions.argmax().item()
    confidence = predictions[0, label_id].item()
    
    # Mapear etiqueta a significado
    sentiment = "positivo" if label_id == 1 else "negativo"
    
    return sentiment, confidence

def main():
    parser = argparse.ArgumentParser(description="Predicción de sentimiento")
    parser.add_argument("--text", type=str, required=False, help="Texto para analizar")
    parser.add_argument("--model_path", type=str, default="./outputs/final", help="Ruta al modelo entrenado")
    args = parser.parse_args()
    
    # Cargar modelo
    print(f"Cargando modelo desde {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    model.eval()
    
    if args.text:
        # Predecir para el texto proporcionado
        sentiment, confidence = predict_sentiment(args.text, model, tokenizer)
        print(f"\nTexto: '{args.text}'")
        print(f"Sentimiento: {sentiment}")
        print(f"Confianza: {confidence:.2%}")
    else:
        # Modo interactivo
        print("\n=== Análisis de Sentimiento Interactivo ===")
        print("Escribe 'salir' para terminar\n")
        
        while True:
            text = input("Ingresa un texto: ")
            if text.lower() in ["salir", "exit", "quit"]:
                break
            
            sentiment, confidence = predict_sentiment(text, model, tokenizer)
            print(f"Sentimiento: {sentiment} (confianza: {confidence:.2%})\n")
    
if __name__ == "__main__":
    main()
