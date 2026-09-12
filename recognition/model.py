"""
Loads and runs the TRAINED offline activity model produced by
train_model.py. This is the "trained AI model" deliverable: a
scikit-learn classifier trained on recorded keypoint sequences,
running entirely locally -- no internet connection or cloud API
required at inference time.

Falls back gracefully (is_available == False) until a model has
actually been trained, so the rest of the app still runs -- using
activity_classifier.classify() as a placeholder -- before you've
collected data and trained one.
"""

import pickle
from pathlib import Path

from recognition.features import to_feature_vector

MODEL_PATH = Path(__file__).resolve().parent / "activity_model.pkl"


class ActivityModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self._bundle = None
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self._bundle = pickle.load(f)

    @property
    def is_available(self):
        return self._bundle is not None

    def predict(self, landmarks):
        """Returns a predicted activity label string, or 'no_person'
        if no person was detected in the frame."""
        features = to_feature_vector(landmarks)
        if features is None:
            return "no_person"
        if not self.is_available:
            raise RuntimeError(
                "No trained model found -- run recognition/train_model.py "
                "first, or call activity_classifier.classify() as a fallback."
            )
        clf = self._bundle["model"]
        return clf.predict([features])[0]
