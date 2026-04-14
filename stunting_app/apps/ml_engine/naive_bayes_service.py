import numpy as np
import json
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import pickle, os


LABEL_ENCODERS = {}
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model.pkl')

FEATURES = ['jenis_kelamin', 'umur', 'status_bbu', 'status_tbu', 'status_gizi', 'status_asi']
TARGET = 'status_stunting'

CATEGORIES = {
    'jenis_kelamin': ['L', 'P'],
    'umur': ['1-34 Bulan', '35-69 Bulan'],
    'status_bbu': ['Sangat Kurang', 'Kurang', 'Normal', 'Berat Badan Normal', 'Risiko Lebih'],
    'status_tbu': ['Sangat Pendek', 'Pendek', 'Normal', 'Tinggi'],
    'status_gizi': ['Gizi Buruk', 'Gizi Kurang', 'Gizi Baik', 'Risiko Gizi Lebih', 'Gizi Lebih', 'Obesitas'],
    'status_asi': ['Tidak', 'Ya'],
    'status_stunting': ['Tidak', 'Potensi Stunting', 'Stunting'],
}


def get_label_encoders():
    encoders = {}
    for col, cats in CATEGORIES.items():
        le = LabelEncoder()
        le.fit(cats)
        encoders[col] = le
    return encoders


def prepare_dataframe(qs):
    """Convert queryset to encoded numpy array"""
    from apps.balita.models import Balita
    encoders = get_label_encoders()
    rows = []
    labels = []

    for b in qs:
        try:
            row = []
            for feat in FEATURES:
                val = getattr(b, feat, '').strip() if isinstance(getattr(b, feat, ''), str) else getattr(b, feat, '')
                row.append(encoders[feat].transform([val])[0])
            rows.append(row)
            if b.status_stunting:
                labels.append(encoders['status_stunting'].transform([b.status_stunting])[0])
        except Exception:
            continue

    return np.array(rows), np.array(labels), encoders


def train_model(training_qs):
    """Train Naive Bayes model and return metrics"""
    X, y, encoders = prepare_dataframe(training_qs)
    if len(X) == 0:
        raise ValueError('Tidak ada data training yang valid.')

    model = CategoricalNB()
    model.fit(X, y)

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'encoders': encoders}, f)

    return model, encoders, X, y


def evaluate_model(model, encoders, testing_qs):
    """Evaluate model on test data"""
    X_test, y_test, _ = prepare_dataframe(testing_qs)
    if len(X_test) == 0:
        raise ValueError('Tidak ada data testing yang valid.')

    y_pred = model.predict(X_test)
    classes = CATEGORIES['status_stunting']
    label_enc = encoders['status_stunting']

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes, zero_division=0, output_dict=True)

    # Build confusion matrix dict
    cm_dict = {
        'labels': classes,
        'matrix': cm.tolist(),
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
    }

    return {
        'akurasi': acc,
        'presisi': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm_dict,
        'classification_report': cr,
        'jumlah_testing': len(X_test),
    }


def load_model():
    """Load saved model from disk"""
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoders']


def predict_single(input_data: dict):
    """
    input_data keys: jenis_kelamin, umur, status_bbu, status_tbu, status_gizi, status_asi
    Returns: predicted class, probabilities dict
    """
    model, encoders = load_model()
    if model is None:
        raise ValueError('Model belum dilatih. Silakan latih model terlebih dahulu.')

    row = []
    for feat in FEATURES:
        val = str(input_data.get(feat, '')).strip()
        row.append(encoders[feat].transform([val])[0])

    X = np.array([row])
    pred_encoded = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label_enc = encoders['status_stunting']
    predicted_class = label_enc.inverse_transform([pred_encoded])[0]

    classes = label_enc.classes_
    proba_dict = {cls: float(proba[i]) for i, cls in enumerate(classes)}

    return predicted_class, proba_dict


def get_naive_bayes_explanation(encoders):
    """Return prior probabilities for display"""
    model, encoders = load_model()
    if model is None:
        return {}

    label_enc = encoders['status_stunting']
    classes = label_enc.classes_
    
    # Log prior probs -> convert to probabilities
    log_priors = model.class_log_prior_
    priors = np.exp(log_priors)
    
    result = {}
    for i, cls in enumerate(classes):
        result[cls] = {
            'prior': float(priors[i]),
            'log_prior': float(log_priors[i]),
        }
    return result
