import os
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

from dataset_muril import MuRILDataset
from model_muril import MuRILSeq2Seq
import config


def load_data():
    with open(os.path.join(config.DATA_PROCESSED, "train.pkl"), "rb") as f:
        return pickle.load(f)


def generate(model, tokenizer, text, device, max_len=50):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        enc_outputs = model.encoder(**inputs)
        enc_hidden = enc_outputs.last_hidden_state[:, 0]

        h = model.init_fc(enc_hidden).unsqueeze(0)
        c = torch.zeros_like(h)

        input_token = torch.tensor([[tokenizer.cls_token_id]]).to(device)

        generated = []

        for _ in range(max_len):
            emb = model.embedding(input_token)
            out, (h, c) = model.decoder(emb, (h, c))
            logits = model.fc_out(out[:, -1, :])

            next_token = torch.argmax(logits, dim=-1)

            if next_token.item() == tokenizer.sep_token_id:
                break

            generated.append(next_token.item())
            input_token = next_token.unsqueeze(0)

    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_rouge(model, tokenizer, dataset, device, num_samples=50):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    scores = []

    for i in range(min(num_samples, len(dataset))):
        item = dataset.data[i]

        src = " ".join(item["src_tokens"])
        ref = " ".join(item["tgt_tokens"])

        pred = generate(model, tokenizer, src, device)

        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    print(f"\nROUGE-L (avg over {num_samples} samples): {sum(scores)/len(scores):.4f}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    train_data = load_data()

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    dataset = MuRILDataset(train_data, tokenizer)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = MuRILSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch+1}")

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, decoder_input_ids)

            outputs = outputs.reshape(-1, outputs.size(-1))
            labels = labels.reshape(-1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print("Model saved!")

    # 🔥 Evaluate
    evaluate_rouge(model, tokenizer, dataset, device)


if __name__ == "__main__":
    train()