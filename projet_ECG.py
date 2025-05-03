# IMPORTATIONS
import sys
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # Ajout SMOTE

# 1. VERSIONS
print('\n=== VERSIONS DES LIBRAIRIES ===')
print(f"Python: {sys.version.split()[0]}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"wfdb: {wfdb.__version__}")

# 2. CHARGEMENT DES DONNEES
def load_ecg_data(record_names):
    signals = []
    labels = []
    for record in record_names:
        try:
            signal, _ = wfdb.rdsamp(record, pn_dir='mitdb')
            annotation = wfdb.rdann(record, 'atr', pn_dir='mitdb')
            for i in range(len(annotation.sample)):
                start = max(0, annotation.sample[i] - 100)
                end = min(len(signal), annotation.sample[i] + 100)
                beat = signal[start:end, 0]
                if len(beat) < 200:
                    beat = np.pad(beat, (0, 200 - len(beat)), mode='constant')
                else:
                    beat = beat[:200]
                if annotation.symbol[i] in ['N', 'V', 'A']:
                    signals.append(beat)
                    labels.append(annotation.symbol[i])
        except Exception as e:
            print(f"Erreur avec {record}: {e}")
    return np.array(signals), np.array(labels)

print(' CHARGEMENT DES DONNEES ECG Réussi')
# Charger plus d'enregistrements pour équilibrer
record_names = [str(i) for i in range(100, 110)]
X, y = load_ecg_data(record_names)

# 3. PRETRAITEMENT
print('\n=== PRETRAITEMENT ===')
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_normalized = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])

# SMOTE pour équilibrer les classes
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_normalized, y_encoded)

# Recalcul des poids après SMOTE (peu utile mais gardé si besoin)
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
weights_dict = dict(zip(np.unique(y_resampled), class_weights))

# 4. EXPLORATION
print('\n=== EXPLORATION ===')
print(f"Nombre de battements (après SMOTE): {len(X_resampled)}")
print("Distribution des classes:")
print(pd.Series(le.inverse_transform(y_resampled)).value_counts())

# Visualisation
plt.figure(figsize=(10, 4))
for i, class_name in enumerate(le.classes_):
    idx = np.where(y_resampled == i)[0][0]
    plt.plot(X_resampled[idx], label=class_name)
plt.title("Exemples de battements par classe (équilibrés)")
plt.legend()
plt.show()

# 5. MODELISATION
print('\n=== EVALUATION DES MODELES ===')
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

models = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight=weights_dict, random_state=42)),
    ('SVM', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
]

results = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_scores)
    print(f"{name}: Accuracy = {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

plt.boxplot(results, tick_labels=[name for name, _ in models])
plt.title("Performance des modèles")
plt.ylabel("Accuracy")
plt.show()

# 6. ENTRAINEMENT DU MODELE FINAL
print('\n=== ENTRAINEMENT DU MODELE FINAL ===')
best_model = RandomForestClassifier(n_estimators=100, class_weight=weights_dict, random_state=42)
best_model.fit(X_train, y_train)

# 7. EVALUATION FINALE
print('\n=== RESULTATS FINAUX ===')
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatrice de confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. VISUALISATION DES PREDICTIONS
plt.figure(figsize=(15, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(X_test[i])
    pred_class = le.inverse_transform([y_pred[i]])[0]
    true_class = le.inverse_transform([y_test[i]])[0]
    proba = y_proba[i].max()
    color = 'green' if pred_class == true_class else 'red'
    plt.title(f"Prédiction: {pred_class} ({proba:.2f})\nVérité: {true_class}", color=color)
plt.tight_layout()
plt.show()
