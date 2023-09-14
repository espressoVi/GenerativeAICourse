import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import toml,os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from processors.coqa import Extract_Features, Processor, Result
from processors.metrics import get_predictions

device = torch.device("cuda")
config = toml.load("config.toml")

def train(model, tokenizer, output_directory):
    epochs = config['train']['EPOCHS']
    train_dataset, _, _ = load_dataset(tokenizer, evaluate=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config['train']['TRAIN_BATCH'], shuffle = True)
    optimizer_parameters = [{"params": [p for n, p in model.named_parameters()
                                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],"weight_decay": 0.01,},
                            {"params": [p for n, p in model.named_parameters() 
                                if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_parameters, lr=config['train']['LR'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps=config['train']['WARMUP'],
                                        num_training_steps=(len(train_dataloader)//epochs))
    counter,train_loss, loss = 1, 0.0, 0.0
    model.zero_grad()
    for ep in range(epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { "input_ids": batch[0],"segment_ids": batch[1],
                  "input_masks": batch[2],"start_positions": batch[3],
                  "end_positions": batch[4],"rationale_mask": batch[5],"cls_idx": batch[6]}
            loss = model(**inputs)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description(f"Epoch {ep+1}/{epochs} | Loss : {(train_loss/counter)}")
            epoch_iterator.refresh()
            if counter % 1000 == 0:
                save_path = os.path.join(output_directory, "weights.pth")
                torch.save(model.state_dict(), save_path)
    return model


def write_predictions(model, tokenizer, output_directory = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    evaluation_dataloader = DataLoader(dataset, batch_size=config['train']['EVAL_BATCH'], shuffle = False)
    mod_results = []
    for batch in tqdm(evaluation_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2]}
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [convert_to_list(output[i]) for output in outputs]
            start_logits, end_logits, yes_logits, no_logits, unk_logits = output
            result = Result(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits, yes_logits=yes_logits, no_logits=no_logits, unk_logits=unk_logits)
            mod_results.append(result)

    output_prediction_file = os.path.join(output_directory, "predictions.json")
    get_predictions(examples, features, mod_results, 20, 30, True, output_prediction_file, False, tokenizer)

def load_dataset(tokenizer, evaluate=False):
    processor = Processor()
    examples = processor.get_examples(evaluate, 2, threads=16)
    features, dataset = Extract_Features(examples=examples, tokenizer=tokenizer,
                    max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=12)
    return dataset, examples, features

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()
