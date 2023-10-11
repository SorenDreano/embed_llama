from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import uuid
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-hf"

def segment_level_scoring(hyps, refs, embeddings, layers, tokenizer):
    scores = []
    for hyp, ref in zip(hyps, refs):
        tokenized = tokenizer([hyp, ref], padding="longest", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            e = embeddings(tokenized)
            for i in range(len(layers)):
                e = layers[i](e)[0]
        score = torch.nn.CosineEmbeddingLoss()(e[0], e[1], torch.full((e[0].shape[0],), -1, device=device)).item()
        scores.append(score)

    return scores

def system_level_scoring(scores):
    return sum(scores)/len(scores)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-R",
        "--reference",
        help="reference translation",
        required=True,
        type=str)
    argParser.add_argument(
        "-H",
        "--hypothesis",
        help="hypothesis translation",
        required=True,
        type=str)
    argParser.add_argument(
        "-n",
        "--number",
        help="number of layers",
        required=True,
        type=str)
    args = argParser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = uuid.uuid4()
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = model.embed_tokens
    layers = model.layers[:int(args.number)]
    del model
    embeddings.to(device)
    for i in range(len(layers)):
        layers[i].to(device)
    with open(args.reference, "r", encoding="utf-8", errors="replace") as f:
        refs = f.read()
    with open(args.hypothesis, "r", encoding="utf-8", errors="replace") as f:
        hyps = f.read()

    scores = segment_level_scoring(hyps, refs, embeddings, layers, tokenizer)
    system_score = system_level_scoring(scores)
    print(system_score)

if __name__ == "__main__":
    main()