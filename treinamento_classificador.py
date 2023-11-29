from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np

data = pd.read_csv('C:/Users/Paula/Documents/avaliacao_semestral/breast-cancer.csv', sep=',', encoding='iso-8859-1')

# Separar dados numéricos e categóricos
numeric_columns = ['deg-malig']
categorical_columns = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'breast', 'breast-quad', 'irradiat']

# Normalizar dados numéricos
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Converter dados categóricos para numéricos
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Converter dados categóricos
one_hot_encoder = OneHotEncoder(sparse_output=False)
categorical_data_one_hot = one_hot_encoder.fit_transform(data[categorical_columns])

# Combinar dados normalizados numéricos e categóricos
normalized_data = np.concatenate([data[numeric_columns].values, categorical_data_one_hot], axis=1)

columns = numeric_columns + list(one_hot_encoder.get_feature_names_out(categorical_columns))
normalized_df = pd.DataFrame(data=normalized_data, columns=columns)

# Balanceamento
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(normalized_df, data['Class'])

balanced_df = pd.DataFrame(data=X_resampled, columns=normalized_df.columns)
balanced_df['Class'] = y_resampled

# Separar os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df.drop('Class', axis=1),  # features
    balanced_df['Class'],  # target
    test_size=0.2,  # proporção para teste
    random_state=42
)

# Inicializar o classificador de árvore de decisão
model = DecisionTreeClassifier(random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Realizar previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Dados Normalizados:")
print(normalized_df)

print("\nDados Balanceados:")
print(balanced_df)

print("\nAcurácia:", accuracy)
print("\nMatriz de Confusão:")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(class_report)