from unittest import TestCase
from torch.utils.data import DataLoader

from src.evaluation import evaluate_sequences
from src.loader import load_config_file, ActionDataset, SequenceDataset

from tests import get_test_path
from tests.mocked_model import MockedDetectionBiRNN


class TestDetection(TestCase):

    def setUp(self) -> None:
        self.model = MockedDetectionBiRNN(
            75,
            512,
            64
        )

    def _evaluate(self, frame_size: int):
        model_config = load_config_file(get_test_path("data/config_model.yaml"))
        model_config['evaluation']['frame_size'] = frame_size

        return evaluate_sequences(
            trained_model=self.model,
            model_config=model_config,
            sequence_loader=DataLoader(
                SequenceDataset(
                    sequence_file=get_test_path("data/test_sequences.data"),
                    meta_file=get_test_path("data/test_meta.txt"),
                    train_mode=False
                )
            ),
            action_dataset=ActionDataset(
                action_file=get_test_path("data/test_actions.data"),
                meta_file=get_test_path("data/test_meta.txt"),
                train_mode=False
            )
        )

    def test_recall_100_full_frame(self):
        self.model.set_predicted_labels(
            [
                [
                    (5, 1.0),
                    (11, 1.0),
                    (1, 1.0),
                    (32, 1.0),
                    (3, 1.0),
                    (51, 1.0),
                    (39, 1.0),
                    (15, 1.0)
                ]
            ]
        )
        result = self._evaluate(frame_size=512)

        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(194, data["correct"])
            self.assertEqual(1.0, data["recall"])

    def test_recall_100_frame_20(self):
        self.model.set_predicted_labels(
            [
                [(5, 1.0)],  # 1 - 20
                [(5, 1.0)],  # 21 - 40
                [(5, 1.0)],  # 41 - 60
                [],  # 61 - 80
                [(11, 1.0)],  # 81 - 100
                [(11, 1.0)],  # 101 - 120
                [(1, 1.0)],  # 121 - 140
                [(1, 1.0)],  # 141 - 160
                [],  # 161 - 180
                [(32, 1.0)],  # 181 - 200
                [(32, 1.0)],  # 201 - 220
                [],  # 221 - 240
                [(3, 1.0)],  # 241 - 260
                [(3, 1.0)],  # 261 - 280
                [(3, 1.0)],  # 281 - 300
                [],  # 301 - 320
                [],  # 321 - 340
                [],  # 341 - 360
                [(51, 1.0)],  # 361 - 380
                [(51, 1.0)],  # 381 - 400
                [(39, 1.0)],  # 401 - 420
                [(39, 1.0)],  # 421 - 440
                [],  # 441 - 460
                [(15, 1.0)],  # 461 - 480
                [(15, 1.0)],  # 481 - 500
                []
            ]
        )
        result = self._evaluate(frame_size=20)

        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(194, data["correct"])
            self.assertEqual(1.0, data["recall"])
