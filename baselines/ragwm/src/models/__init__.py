from .GPT import GPT
import json
# Note: PaLM2/Vicuna/Llama are imported lazily inside create_model so that the
# GPT (OpenAI-compatible, e.g. local Ollama) path works without google.generativeai/fastchat.

def load_json(file_path):
    print(f'load json: {file_path}')
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        from .PaLM2 import PaLM2
        model = PaLM2(config)
    elif provider == 'vicuna':
        from .Vicuna import Vicuna
        model = Vicuna(config)
    elif provider == 'gpt':
        model = GPT(config)
    elif provider == 'llama':
        from .Llama import Llama
        model = Llama(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model


