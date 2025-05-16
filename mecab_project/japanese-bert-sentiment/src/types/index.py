from typing import List, Dict, Any

class SentimentInput:
    def __init__(self, text: str):
        self.text = text

class SentimentOutput:
    def __init__(self, label: str, score: float):
        self.label = label
        self.score = score

class BatchSentimentOutput:
    def __init__(self, results: List[SentimentOutput]):
        self.results = results

class ModelConfig:
    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length

class PredictionResult:
    def __init__(self, input: SentimentInput, output: SentimentOutput):
        self.input = input
        self.output = output

class BatchPredictionResult:
    def __init__(self, inputs: List[SentimentInput], outputs: List[SentimentOutput]):
        self.inputs = inputs
        self.outputs = outputs