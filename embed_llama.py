from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import uuid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
embeddings = model.embed_tokens
l0 = model.layers[0]
l1 = model.layers[1]
del model
embeddings.to(device)
l0.to(device)
l1.to(device)
tokenizer.pad_token = uuid.uuid4()

def segment_level_scoring(hyps, refs, model, tokenizer):
    scores = []
    for hyp, ref in zip(hyps, refs):
        t = tokenizer([hyp, ref], padding="longest", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            e = embeddings(t)
            e1 = l0(e)[0]
            e2 = l1(e1)[0]
        score = torch.nn.CosineEmbeddingLoss()(e2[0], e2[1], torch.full((e2[0].shape[0],), -1, device=device)).item()
        scores.append(score)

    return scores

def system_level_scoring(hyps, refs, model, tokenizer):
    scores = segment_level_scoring(hyps, refs, model, tokenizer)
    return sum(scores)/len(scores)