from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd

# load elon musk tweets
df = pd.read_csv('elon_musk_tweets.csv')

# isloate the text column
tweet_data = df['text']

# convert the tweets to a list
tweet_data = tweet_data.values.tolist()


class TweetDataset(Dataset):
    def __init__(self, tweets, tokenizer, max_length=128):
        self.encodings = tokenizer(
            tweets, truncation=True, max_length=max_length, padding="max_length")

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# You can choose another existing token if you prefer
tokenizer.pad_token = tokenizer.eos_token


# Tokenize and preprocess the dataset
dataset = TweetDataset(tweet_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fine-tune the model on the tweet dataset
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
accumulation_steps = 4
for epoch in range(num_epochs):
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["input_ids"].to(device)

        outputs = model(inputs, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (progress_bar.n + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Display loss in the progress bar
        progress_bar.set_postfix(loss=loss.item())


# Save the fine-tuned model for later use
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")


# Generate tweets without seed text
generated_tweets = model.generate(
    max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
decoded_tweets = tokenizer.decode(
    generated_tweets[0], skip_special_tokens=True)

print("Generated Tweet:", decoded_tweets)
