from typing import List
from sklearn.metrics import fbeta_score
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

# define your custom evaluation metric
class F05Score():
    def __init__(self) -> None:
        self.name = "F05"
        self.is_maximized = True

    def calculate(self, true_labels: List[List[str]], predicted_labels: List[List[str]]) -> float:
        beta = 0.5
        flat_true_labels = [label for sentence_labels in true_labels for label in sentence_labels]
        flat_predicted_labels = [label for sentence_labels in predicted_labels for label in sentence_labels]
        return fbeta_score(flat_true_labels, flat_predicted_labels, beta=beta, average='micro')

