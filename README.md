# kaggle-disaster(Simple RNN model) + FastAPI
This repo is made to implement the 'FastAPI' example through a simple RNN model.

# folder structure

data: train/test data(csv file)

main.py: model(simple rnn) + serving(FastAPI) example code

simple-rnn-model.ipynb: kaggle notebook file (model train code using pytorch-lightning) 

<pre>

├── data
│   ├── test.csv
│   └── train.csv
├── models
│   ├── rnn.ckpt
│   └── vocab.pt
├── README.md
├── requirements.txt
├── simple-rnn-model.ipynb
├── src
│    └── main.py

</pre>
# Example code operation method
ex)
python3 ./src/main.py 
### or
ex)
uvicorn src.main:app --host=0.0.0.0 --port=9999

# Service
---

POST /recognition
Request(JSON):
```json
{
    "text": "input string"
}
```

Response(JSON):
```json
{
    "result": "result string",
}
```
# Summary
This is a text classification model using a simple LSTM structure.

A model that classifies whether it is a disaster or not based on input text.

Vocabulary is extracted from the torch script model.

Model training was conducted using kaggle notebook.

I used the 'FastAPI' web framework for torch model serving.



# Reference
[Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/overview)

[FastAPI](https://github.com/tiangolo/fastapi)

[PyTorch Lightning](https://github.com/PyTorchLightning)