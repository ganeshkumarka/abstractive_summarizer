from torch.utils.data import Dataset


class MuRILDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        src_text = " ".join(item["src_tokens"])
        tgt_text = " ".join(item["tgt_tokens"])

        inputs = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        labels = targets["input_ids"].squeeze(0)

        # 🔥 IMPORTANT: shift for teacher forcing
        decoder_input_ids = labels[:-1]
        labels = labels[1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }