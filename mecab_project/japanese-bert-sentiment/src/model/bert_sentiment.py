from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BertSentiment:
    def __init__(self, model_name='cl-tohoku/bert-base-japanese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def get_sentiment(self, text):
        probabilities = self.predict(text)
        sentiment = torch.argmax(probabilities, dim=1).item()
        return sentiment, probabilities.numpy()