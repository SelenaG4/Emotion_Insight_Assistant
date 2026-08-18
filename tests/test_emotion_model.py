import numpy as np

from app.emotion_model import CLASS_LABELS, EmotionClassifier, build_model3


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
