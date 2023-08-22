from unittest import TestCase

from src.model.classifier import load_trained_model
from src.util.definitions import TRAINED_MODEL_DIR


class TestTrainedModelLoading(TestCase):
    def setUp(self):
        self.ffn = load_trained_model(
            "FFN", TRAINED_MODEL_DIR / "2022-12-16-144509_863758" / "best.ckpt"
        )
        self.ffn.eval()
        self.dmpnn1d = load_trained_model(
            "D-MPNN", TRAINED_MODEL_DIR / "2022-12-16-145840_448790" / "best.ckpt"
        )
        self.dmpnn1d.eval()

    def test_ffn_first_linear_layer_has_correct_input_size(self):
        self.assertEqual(self.ffn.decoder._modules["ffn"][1].in_features, 132)

    def test_ffn_second_linear_layer_has_correct_hidden_size(self):
        self.assertEqual(self.ffn.decoder._modules["ffn"][4].in_features, 59)

    def test_ffn_dropout_has_correct_value(self):
        self.assertAlmostEqual(
            self.ffn.decoder._modules["ffn"][3].p, 3.67e-05, places=7
        )
