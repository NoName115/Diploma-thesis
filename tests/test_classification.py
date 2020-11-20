from unittest import TestCase
from torch.utils.data import DataLoader

from src.evaluation import evaluate_actions
from src.loader import load_config_file, ActionDatasetIterative

from tests import get_test_path
from tests.mocked_model import MockedClassificationBiRNN


class TestClassification(TestCase):

    def setUp(self) -> None:
        self.model = MockedClassificationBiRNN(
            75,
            512,
            64
        )

    def _evaluate(self):
        return evaluate_actions(
            trained_model=self.model,
            model_config=load_config_file(get_test_path("data/config_model.yaml")),
            evaluation_loader=DataLoader(
                ActionDatasetIterative(
                    action_file=get_test_path("data/test_actions.data"),
                    meta_file=get_test_path("data/test_meta.txt"),
                    train_mode=False
                )
            )
        )

    def test_100_accuracy(self):
        self.model.set_predicted_labels(
            [5, 11, 1, 32, 3, 51, 39, 15]
        )
        correct, total = self._evaluate()
        self.assertEqual(8, correct)
        self.assertEqual(8, total)

    def test_50_accuracy(self):
        self.model.set_predicted_labels(
            [5, 1, 1, 1, 3, 1, 39, 1]
        )
        correct, total = self._evaluate()
        self.assertEqual(4, correct)
        self.assertEqual(8, total)

    def test_0_accuracy(self):
        self.model.set_predicted_labels(
            [1, 1, 2, 1, 1, 1, 1, 1]
        )
        correct, total = self._evaluate()
        self.assertEqual(0, correct)
        self.assertEqual(8, total)
