from unittest import TestCase
from typing import Tuple, Optional
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
            ),
            keep_short_memory=False
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
        th_list = [
            (0.2, '0.1'),
            (0.3, '0.2'),
            (0.4, '0.3'),
            (0.5, '0.4'),
            (0.6, '0.5'),
            (0.7, '0.6'),
            (0.8, '0.7'),
            (0.9, '0.8'),
            (1.0, '0.9')
        ]
        for th_data, th_test in th_list:
            self._test_recall_100_frame_20_th(th_data, th_test)

    def _test_recall_100_frame_20_th(self, th_data: float, th_test: str):
        self.model.set_predicted_labels(
            [
                [(5, th_data)],  # 1 - 20
                [(5, th_data)],  # 21 - 40
                [(5, th_data)],  # 41 - 60
                [],  # 61 - 80
                [(11, th_data)],  # 81 - 100
                [(11, th_data)],  # 101 - 120
                [(1, th_data)],  # 121 - 140
                [(1, th_data)],  # 141 - 160
                [],  # 161 - 180
                [(32, th_data)],  # 181 - 200
                [(32, th_data)],  # 201 - 220
                [],  # 221 - 240
                [(3, th_data)],  # 241 - 260
                [(3, th_data)],  # 261 - 280
                [(3, th_data)],  # 281 - 300
                [],  # 301 - 320
                [],  # 321 - 340
                [],  # 341 - 360
                [(51, th_data)],  # 361 - 380
                [(51, th_data)],  # 381 - 400
                [(39, th_data)],  # 401 - 420
                [(39, th_data)],  # 421 - 440
                [],  # 441 - 460
                [(15, th_data)],  # 461 - 480
                [(15, th_data)],  # 481 - 500
                []  # 501 -
            ]
        )
        result = self._evaluate(frame_size=20)

        self.assertEqual(194, result["total_frames"])

        data = result["thresholds"][th_test]
        self.assertEqual(194, data["correct"])
        self.assertEqual(1.0, data["recall"])

    def test_recall_100_frame_50(self):
        self.model.set_predicted_labels(
            [
                [(5, 1.0)],  # 1 - 50
                [(5, 1.0), (11, 1.0)],  # 51 - 100
                [(11, 1.0), (1, 1.0)],  # 101 - 150
                [(1, 1.0), (32, 1.0)],  # 151 - 200
                [(32, 1.0)],  # 201 - 250
                [(3, 1.0)],  # 251 - 300
                [],  # 301 - 350
                [(51, 1.0)],  # 351 - 400
                [(39, 1.0)],  # 401 - 450
                [(15, 1.0)],  # 451 - 500
                [],  # 501 -
            ]
        )
        result = self._evaluate(frame_size=50)

        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(194, data["correct"])
            self.assertEqual(1.0, data["recall"])

    def test_precision_100_frame_1(self):
        def append_labels(number_of_labels: int, inner_value: Optional[Tuple[int, float]]):
            return [[inner_value] if inner_value else [] for _ in range(number_of_labels)]

        predict_labels = []

        predict_labels += append_labels(19, None)
        predict_labels += append_labels(40, (5, 1.0))
        predict_labels += append_labels(30, None)
        predict_labels += append_labels(20, (11, 1.0))
        predict_labels += append_labels(20, None)
        predict_labels += append_labels(30, (1, 1.0))
        predict_labels += append_labels(40, None)
        predict_labels += append_labels(20, (32, 1.0))
        predict_labels += append_labels(40, None)
        predict_labels += append_labels(40, (3, 1.0))
        predict_labels += append_labels(80, None)
        predict_labels += append_labels(20, (51, 1.0))
        predict_labels += append_labels(20, None)
        predict_labels += append_labels(11, (39, 1.0))
        predict_labels += append_labels(39, None)
        predict_labels += append_labels(13, (15, 1.0))
        predict_labels += append_labels(30, None)
        self.model.set_predicted_labels(predict_labels)

        assert len(predict_labels) == 512
        result = self._evaluate(frame_size=1)

        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(194, data["above"])
            self.assertEqual(1.0, data["precision"])

    def test_precision_above_frame_10(self):
        self.model.set_predicted_labels(
            [
                [],  # 1 - 10
                [(5, 1.0)],  # 11 - 20
                [(5, 1.0)],  # 21 - 30
                [(5, 1.0)],  # 31 - 40
                [(5, 1.0)],  # 41 - 50
                [(5, 1.0)],  # 51 - 60
                [],  # 61 - 70
                [],  # 71 - 80
                [(11, 1.0)],  # 81 - 90
                [(11, 1.0)],  # 91 - 100
                [(11, 1.0)],  # 101 - 110
                [],  # 111 - 120
                [(1, 1.0)],  # 121 - 130
                [(1, 1.0)],  # 131 - 140
                [(1, 1.0)],  # 141 - 150
                [(1, 1.0)],  # 151 - 160
                [],  # 161 - 170
                [],  # 171 - 180
                [],  # 181 - 190
                [(32, 1.0)],  # 191 - 200
                [(32, 1.0)],  # 201 - 210
                [(32, 1.0)],  # 211 - 220
                [],  # 221 - 230
                [],  # 231 - 240
                [],  # 241 - 250
                [(3, 1.0)],  # 251 - 260
                [(3, 1.0)],  # 261 - 270
                [(3, 1.0)],  # 271 - 280
                [(3, 1.0)],  # 281 - 290
                [(3, 1.0)],  # 291 - 300
                [],  # 301 - 310
                [],  # 311 - 320
                [],  # 321 - 330
                [],  # 331 - 340
                [],  # 341 - 350
                [],  # 351 - 360
                [],  # 361 - 370
                [(51, 1.0)],  # 371 - 380
                [(51, 1.0)],  # 381 - 390
                [(51, 1.0)],  # 391 - 400
                [],  # 401 - 410
                [(39, 1.0)],  # 411 - 420
                [(39, 1.0)],  # 421 - 430
                [],  # 431 - 440
                [],  # 441 - 450
                [],  # 451 - 460
                [(15, 1.0)],  # 461 - 470
                [(15, 1.0)],  # 471 - 480
                [(15, 1.0)],  # 481 - 490
                [],  # 491 - 500
                [],  # 501 - 510
                [],  # 511 -
            ]
        )

        result = self._evaluate(frame_size=10)
        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(280, data["above"])
            self.assertEqual(194, data["correct"])

    def test_precision_above_frame_100(self):
        self.model.set_predicted_labels(
            [
                [(5, 1.0), (11, 1.0)],  # 1 - 100
                [(11, 1.0), (1, 1.0), (32, 1.0)],  # 101 - 200
                [(32, 1.0), (3, 1.0)],  # 201 - 300
                [(51, 1.0)],  # 301 - 400
                [(39, 1.0), (15, 1.0)],  # 401 - 500
                [],  # 501 -
            ]
        )

        result = self._evaluate(frame_size=100)
        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(1000, data["above"])
            self.assertEqual(194, data["correct"])
            self.assertEqual(1.0, data["recall"])
            self.assertEqual(0.194, data["precision"])

    def test_precision_recall_above_frame_100(self):
        self.model.set_predicted_labels(
            [
                [(5, 1.0)],  # 1 - 100
                [(11, 1.0), (48, 1.0), (32, 1.0)],  # 101 - 200
                [(3, 1.0)],  # 201 - 300
                [(51, 1.0)],  # 301 - 400
                [(1, 1.0), (15, 1.0)],  # 401 - 500
                [],  # 501 -
            ]
        )

        result = self._evaluate(frame_size=100)
        self.assertEqual(194, result["total_frames"])
        for th, data in result["thresholds"].items():
            self.assertEqual(800, data["above"])
            self.assertEqual(123, data["correct"])
            assert data["recall"] > 0.634
            self.assertEqual(0.15375, data["precision"])
