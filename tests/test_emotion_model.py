from pathlib import Path

import numpy as np
import pytest

from app.emotion_model import CLASS_LABELS, EmotionClassifier, build_model3

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "model3.weights.h5"


def test_architecture_param_count_matches_notebook_summary() -> None:
    model = build_model3()
    # Matches the param count printed by model3.summary() in the original notebook run.
    assert model.count_params() == 1_782_340


def test_predict_returns_valid_softmax_distribution() -> None:
    clf = EmotionClassifier()  # no weights path -> random init, shape/plumbing test only
    fake_face = (np.random.rand(48, 48) * 255).astype("uint8")
    label, confidence, per_class = clf.predict(fake_face)

    assert label in CLASS_LABELS
    assert 0.0 <= confidence <= 1.0
    assert set(per_class.keys()) == set(CLASS_LABELS)
    assert abs(sum(per_class.values()) - 1.0) < 1e-4
    assert clf.weights_loaded is False


def test_predict_handles_non_square_input() -> None:
    clf = EmotionClassifier()
    odd_shaped_image = (np.random.rand(96, 64) * 255).astype("uint8")
    label, confidence, per_class = clf.predict(odd_shaped_image)
    assert label in CLASS_LABELS


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="real weights not present in this checkout")
def test_real_weights_load_and_report_loaded() -> None:
    clf = EmotionClassifier(weights_path=WEIGHTS_PATH)
    assert clf.weights_loaded is True


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="real weights not present in this checkout")
def test_real_weights_are_deterministic_and_confident() -> None:
    """Not an accuracy claim (no labeled held-out face set ships in this repo --
    that evaluation lives in the original notebook, see README). This just checks
    the loaded weights behave like a genuinely trained model rather than a random
    one: identical input -> identical output (no dropout at inference), and the
    top class carries meaningfully more mass than the 1-in-4 chance baseline.
    """
    clf = EmotionClassifier(weights_path=WEIGHTS_PATH)
    rng = np.random.default_rng(7)
    image = (rng.uniform(0, 1, size=(48, 48)) * 255).astype("uint8")

    label_a, confidence_a, per_class_a = clf.predict(image)
    label_b, confidence_b, per_class_b = clf.predict(image)

    assert label_a == label_b
    assert confidence_a == pytest.approx(confidence_b, abs=1e-6)
    assert confidence_a > 0.35  # meaningfully above the 0.25 random-guess baseline


def test_real_weights_match_the_originally_saved_model_bit_for_bit() -> None:
    """Regression guard for the weight-transfer step (full Colab model.save() ->
    save_weights() -> load_weights() into this app's own build_model3()). If this
    ever drifts, it means the shipped weights file stopped matching the actual
    trained model, silently. Reference values captured once from the verified
    transfer (see PR/commit that added models/model3.weights.h5) and pinned here
    rather than re-loading the original 21MB full-model file on every test run.
    """
    if not WEIGHTS_PATH.exists():
        pytest.skip("real weights not present in this checkout")

    clf = EmotionClassifier(weights_path=WEIGHTS_PATH)
    rng = np.random.default_rng(42)
    image = (rng.uniform(0, 1, size=(48, 48, 3)) * 255).astype("uint8")  # exercise the 3-channel path too
    _, _, per_class = clf.predict(image)

    reference = {
        "happy": 0.12968087196350098,
        "sad": 0.4333098232746124,
        "neutral": 0.4288417398929596,
        "surprise": 0.008167491294443607,
    }
    for label, expected in reference.items():
        assert per_class[label] == pytest.approx(expected, abs=1e-5)
