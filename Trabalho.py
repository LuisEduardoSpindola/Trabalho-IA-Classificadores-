import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carrega o dataset CIFAR-10
(XTreino, yTreino), (XTeste, yTeste) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Pré-processamento
XTreino_reshaped = XTreino.reshape((XTreino.shape[0], -1))
XTeste_reshaped = XTeste.reshape((XTeste.shape[0], -1))

XTreino_normalized = XTreino_reshaped / 255.0
XTeste_normalized = XTeste_reshaped / 255.0

yTreino_flat = yTreino.flatten()
yTeste_flat = yTeste.flatten()

# 3. Reduz dimensionalidade com PCA
pca = PCA(n_components=100)
XTreino_pca = pca.fit_transform(XTreino_normalized)
XTeste_pca = pca.transform(XTeste_normalized)

# 4. Define e treina três classificadores

# a) Regressão Logística
clf_log = LogisticRegression(max_iter=1000, solver='saga', random_state=42)
clf_log.fit(XTreino_pca, yTreino_flat)

# b) Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_rf.fit(XTreino_pca, yTreino_flat)

# c) Support Vector Machine
clf_svm = SVC(kernel='rbf', random_state=42)
clf_svm.fit(XTreino_pca, yTreino_flat)

# 5. Avalia os classificadores
modelos = {
    "Logistic Regression": clf_log,
    "Random Forest": clf_rf,
    "SVM": clf_svm
}

print("="*40)
print("AVALIAÇÃO DOS MODELOS NO CIFAR-10 (com PCA)")
print("="*40)

for nome, modelo in modelos.items():
    print(f"\n--- Resultados para: {nome} ---")

    y_pred = modelo.predict(XTeste_pca)

    # Cálculo das métricas solicitadas
    acc = accuracy_score(yTeste_flat, y_pred)
    rec = recall_score(yTeste_flat, y_pred, average='macro')
    prec = precision_score(yTeste_flat, y_pred, average='macro')
    f1 = f1_score(yTeste_flat, y_pred, average='macro')

    print(f"Acurácia (Accuracy): {acc:.4f}")
    print(f"Recall (Sensibilidade): {rec:.4f}")
    print(f"Precisão (Precision): {prec:.4f}")
    print(f"F1 Score (Pontuação F1): {f1:.4f}")
    
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(yTeste_flat, y_pred, target_names=class_names))

    # Item 2: Geração da Matriz de Confusão
    print(f"\nMatriz de Confusão (numérica) - {nome}:")
    cm = confusion_matrix(yTeste_flat, y_pred)
    print(cm)

    # Plotando a Matriz de Confusão (visual)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {nome}')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.show()

print("\nAvaliação concluída.")