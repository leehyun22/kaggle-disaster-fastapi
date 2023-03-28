import re
import string
import uvicorn

from typing import Tuple
from multiprocessing import set_start_method

from fastapi import FastAPI

import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from torchmetrics import Accuracy

import pytorch_lightning as pl

from pydantic import BaseModel
import pandas as pd
import logging

# logging
logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)


formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)
logger.addHandler(stream_hander)


class TweetNCTInput(BaseModel):
    text: str


class TweetNCTOutput(BaseModel):
    result: str


TweetResult = {
    0: "normal",
    1: "disaster"
}

app = FastAPI()

device = 'cpu'

# Load torch script vocab, input: str, output: list[int]
vocab_path = "./models/vocab.pt"
vocab = torch.jit.load(vocab_path, map_location=device)

# Limit cpu usage
torch.set_num_threads(1)

# Define neural text classification model


class RNNClassifier(pl.LightningModule):

    def __init__(
        self,
        input_size: int,
        word_vec_size: int,
        hidden_size: int,
        n_classes: int = 2,
        n_layers: int = 4,
        dropout_p: float = .4,
    ):
        super().__init__()
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.crit = nn.NLLLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=2, average='macro', top_k=1)
        self.valid_acc = Accuracy(task="multiclass", num_classes=2, average='macro', top_k=1)

        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )
        self.generator = nn.Linear(hidden_size*2, n_classes)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)
        return y

    def training_step(self, batch: Tuple, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.crit(y_hat, y.squeeze())
        y_hat = torch.argmax(y_hat, dim=-1)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_acc = self.train_acc.compute()
        self.log("train_accuracy", train_acc)
        self.log("train_epoch_loss", avg_loss)
        return

    def validation_step(self, batch: Tuple, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.crit(y_hat, y.squeeze())
        y_hat = torch.argmax(y_hat, dim=-1)
        self.valid_acc(y_hat, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        valid_acc = self.valid_acc.compute()
        self.log("valid_accuracy", valid_acc)
        self.log("valid_epoch_loss", avg_loss)
        self.log("epoch", self.current_epoch)

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = torch.argmax(self(x), dim=-1)
        return y_hat

    def test_epoch_end(self, outputs):
        y_hat = torch.cat(outputs, dim=0)
        y_hat = y_hat.detach().cpu().numpy()
        
        test_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
        test_df["target"] = y_hat
        test_df.to_csv("/kaggle/working/submission.csv", index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer


# Load simple neural text classification model using pytorch-lightning
model = RNNClassifier.load_from_checkpoint("./models/rnn.ckpt", map_location=device)
model.eval()


# single inference
async def inference(decoded_text: torch.Tensor):
    result = model(decoded_text)
    return result.squeeze()


# define preprocess
def preprocess(text: str) -> str:
    url = re.compile(r'https?://\S+|www\.\S+')
    html = re.compile(r'<.*?>')
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    table = str.maketrans('', '', string.punctuation)

    text = url.sub(r'', text)
    text = html.sub(r'', text)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(table)
    return text


@app.post("/classification", response_model=TweetNCTOutput)
async def post(request: TweetNCTInput):
    logger.info("post classification")

    # preprocess input text
    normalized_text = preprocess(request.text)

    # decode text and convert list[int] to tensor
    decoded_text = torch.IntTensor(vocab(normalized_text)).unsqueeze(0)
    with torch.no_grad():
        logger.info("start model inference")
        result = await inference(decoded_text)
        logger.info("finish model inference")
        result = TweetResult[int(torch.argmax(result.squeeze(), dim=-1).detach())]
        logger.info(f"result : {result} , type : {type(result)}")
        return TweetNCTOutput(
            result=result
        )


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        print(e)
    uvicorn.run('main:app', host='0.0.0.0', port=9999, reload=True)
