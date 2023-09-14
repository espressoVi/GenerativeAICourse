import os, toml, json
import torch
from transformers import BertTokenizer
from model import Bert
from train import train, write_predictions
from processors.eval import CoQAEvaluator

config = toml.load("config.toml")

def manager(evaluate:bool, output_directory):
    if os.path.exists(output_directory):
        if not evaluate and len(os.listdir(output_directory))>0:
            raise ValueError(f"Output directory {output_directory}  already exists, Change output_directory name")
    else:
        os.makedirs(output_directory)
    path = os.path.join(output_directory, "weights.pth")
    tokenizer = BertTokenizer.from_pretrained(config['model'])
    device = torch.device("cuda")
    model = Bert()
    if not evaluate:
        model.to(device)
        model = train(model, tokenizer, output_directory)
        torch.save(model.state_dict(), path)
    else:
        model.load_state_dict(torch.load(path))
        model.to(device)
        write_predictions(model, tokenizer, output_directory = output_directory)

def print_results():
    evaluator = CoQAEvaluator(config['data']['TEST_FILE'])
    prediction_file = os.path.join(config['data']['OUTPUT'], "predictions.json")
    with open(prediction_file) as f:
        pred_data = CoQAEvaluator.preds_to_dict(prediction_file)
    print(json.dumps(evaluator.model_performance(pred_data), indent=2))

def main():
    manager(evaluate=False, output_directory = config['data']['OUTPUT']) 
    manager(evaluate=True, output_directory = config['data']['OUTPUT'])
    print_results()

if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
