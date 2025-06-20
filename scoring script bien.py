import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
from sklearn.impute import KNNImputer
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
# Importation du jeu de donnée
df = pd.read_csv("D:/Projet road map/projet scoring bancaire/loan_sanction_train.csv")
pd.set_option('display.max_columns',None)
print(df)
df.drop(columns="Loan_ID",axis=1,inplace= True)
## Analyse exploratoire des données
####Analyse des valeurs manquantes 
df.isna().sum()
df.isnull().any(axis=1).sum()
df.isnull().any(axis=1).sum()/df.any(axis=1).sum()
# cela montre que le jeu de données contient 21% de valeurs manquantes nous ne pouvons donc pasz les supprimer directement nous allons les imputer
msn.heatmap(df)
"""
 ce graphique montre qu'il ya pas vraiment de relation entre les valeurs manquantes dans le jeu de donnée à part pour la variable dependents 
 et married nous allons donc imputer par la mediane et le mode les autres variables tanids que l'imputation de dependents se fera par le plus
 proche voisins
 
"""
for col in ['Gender', 'Married', 'Self_Employed', 'Credit_History']:
    df[col] = df[col].fillna(df[col].mode()[0])
df['LoanAmount']= df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term']= df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
## avec le cas plus proche voisin
imputer= KNNImputer(n_neighbors=3)
df['Dependents'] = df['Dependents'].replace('3+', 4).astype(float)
df[['Dependents']] = imputer.fit_transform(df[['Dependents']])
df.isna().sum()
df['Dependents'] = df['Dependents'].replace(4, '3+').astype(str)
#pour la vrai valeur de loan Amout 
df["LoanAmount"]=df["LoanAmount"]*1000
df["CoapplicantIncome"]=df["CoapplicantIncome"]*1000
## Nous allons couper les variables quantitatives en classe pour les avoirs en qualitatives
for col in df.select_dtypes(include=["int", "float"]).drop(columns="Credit_History").columns:
    print(f"\nDécoupage de la variable '{col}':")

    try:
        # Découpage en quartiles sans label
        bins = pd.qcut(df[col], q=4, duplicates='drop')
        
        # Récupération des bornes des intervalles
        intervals = bins.cat.categories

        # Création d'un dictionnaire de mapping : intervalle -> string des bornes
        mapping = {interval: f"[{interval.left} ; {interval.right}]" for interval in intervals}

        # Application du mapping sur la variable
        df[col] = bins.map(mapping)

        # Affichage des bornes utilisées
        for i, interval in enumerate(intervals):
            print(f"  Classe {i+1} : [{interval.left} ; {interval.right}]")

    except ValueError as e:
        print(f"  Erreur pour la variable '{col}': {e}")

for col in df:
    print(f"Valeurs uniques de {col} : {df[col].unique()}")

## remplaçons la variable Credit_History par respecté et non respecté
df['Credit_History'] = df['Credit_History'].map({1.0: 'Respecté', 0.0: 'Non respecté'})

# 1. Sélection des variables qualitatives sauf la cible Loan_Status
variables_qualitatives = df.select_dtypes(include=['object', 'category']).columns.drop('Loan_Status')

# 2. Fonction pour faire le test chi2
def chi2_test(var, target='Loan_Status'):
    table = pd.crosstab(df[var], df[target])
    chi2, p, dof, expected = chi2_contingency(table)
    return p

# 3. Boucle pour tester chaque variable et afficher résultat
significance_level = 0.05
results = {}

for col in variables_qualitatives:
    p_val = chi2_test(col)
    results[col] = p_val
    statut = 'lié' if p_val < significance_level else 'non lié'
    print(f"{col}: p-value = {p_val:.4f} -> {statut}")

# 4. Sélection des variables liées à Loan_Status
variables_significatives = [var for var, p in results.items() if p < significance_level]

print("\nVariables liées à Loan_Status :")
print(variables_significatives)

# 5. Création d'un DataFrame réduit avec seulement ces variables et la cible
df_reduit = df[variables_significatives + ['Loan_Status']]
df_reduit
# des variables ne sont pas significatives mais au vu du metier nous allons les garder ces variables sont "ApplicantIncome" et "LoanAmount"
df_reduit = df_reduit.copy()

df_reduit["LoanAmount"] = df["LoanAmount"]
df_reduit["ApplicantIncome"] = df["ApplicantIncome"]

# passons à la regression logistique
X = df_reduit.drop(columns=["Loan_Status"])
y = df_reduit["Loan_Status"].map({'Y':1, 'N':0})  # Convertir en 0/1

# On encode toutes les variables catégorielles dans X avant séparation train/test
X_encoded = pd.get_dummies(X, drop_first=True)

# Puis on refait train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# On refait l'entraînement
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

#recuperons les coefficients afin de passer au scoring
# Coefficients
coefficients = model.coef_[0]  # car coef_ est une matrice (1, n_features)
intercept = model.intercept_[0]

# Associer coefficients à leurs variables
feature_names = X_train.columns
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

print(f"Intercept (constante): {intercept:.4f}")
#### calcul du score 
# 1. Récupérer les modalités d'origine
levels = {}
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        levels[col] = X[col].unique().tolist()

# 2. Récupérer les coefficients du modèle déjà entraîné
coefficients = model.coef_[0]  # coefficients (array)
intercept = model.intercept_[0]  # intercept (float)

# 3. Récupérer les noms des variables encodées
feature_names = X_encoded.columns.tolist()

# 4. Construire tableau complet 
rows = []

# Intercept
rows.append(['(Intercept)', '', '(Intercept)', intercept])

for var, modalities in levels.items():
    ref_mod = modalities[0]  # modalité de référence supprimée (drop_first)
    # Modalité de référence : coefficient 0
    rows.append([var, ref_mod, f'{var}_{ref_mod}', 0.0])
    
    # Pour les autres modalités : utiliser les vrais coefficients
    for mod in modalities[1:]:
        col_name = f"{var}_{mod}"
        if col_name in feature_names:
            coef = coefficients[feature_names.index(col_name)]
        else:
            coef = 0.0  # Sécurité (si jamais absente)
        rows.append([var, mod, col_name, coef])

# 5. DataFrame final
param = pd.DataFrame(rows, columns=['VARIABLE', 'MODALITE', 'NOMVAR', 'COEF'])

print(param)

## Calculons le poids total
# mini : min coef par variable
mini = param.groupby('VARIABLE')['COEF'].min().reset_index().rename(columns={'COEF':'min'})

# maxi : max coef par variable
maxi = param.groupby('VARIABLE')['COEF'].max().reset_index().rename(columns={'COEF':'max'})

# fusion
total = pd.merge(mini, maxi, on='VARIABLE')

# diff
total['diff'] = total['max'] - total['min']

# poids total
poids_total = total['diff'].sum()
print("Poids total =", poids_total)

#calculons les poids pour chaque variables
grille = pd.merge(param, mini, on='VARIABLE', how='left')

# Calcul du delta
grille['delta'] = grille['COEF'] - grille['min']

# Calcul du poids/scoring sur 1000
grille['POIDS'] = np.round(1000 * grille['delta'] / poids_total).astype(int)

# Filtrer les lignes où VARIABLE n'est pas vide
resultat = grille[grille['VARIABLE'] != ""][['VARIABLE', 'MODALITE', 'POIDS']]

print(resultat)

# calculons le score pour chaque individus
# Initialiser un tableau de score à zéro pour chaque individu
scores = np.zeros(X_encoded.shape[0])

# Ajouter pour chaque modalité présente dans grille
for idx, row in grille.iterrows():
    nomvar = row['NOMVAR']
    poids = row['POIDS']
    if nomvar in X_encoded.columns:
        scores += poids * X_encoded[nomvar].values

# Ajouter la colonne score à df_reduit
df_reduit['SCORE'] = scores
df_reduit
#pour que ce soit les individus ayant un gros score qui soit les moins risqué
df_reduit['SCORE'] = 1000 - scores
df_reduit

# Calcul des probabilités (colonne 1 = probabilité de défaut)
df_reduit['PD'] = model.predict_proba(X_encoded)[:, 1]

# 2. Calcul des quantiles pour découper en 20 classes (vingtiles)
quantiles = np.quantile(df_reduit['PD'], np.linspace(0, 1, 21))

# 3. Création des classes de risques (0 à 19)
df_reduit['classe_risque'] = pd.cut(df_reduit['PD'], bins=quantiles, labels=False, include_lowest=True)

# Si tu préfères les classes numérotées 1 à 20 :
# df_reduit['classe_risque'] = pd.cut(df_reduit['PD'], bins=quantiles, labels=range(1,21), include_lowest=True)

# 4. Affichage du nombre d’observations par classe
print(df_reduit['classe_risque'].value_counts().sort_index())

# 5. Affichage de la moyenne de PD par classe de risque
print(df_reduit.groupby('classe_risque')['PD'].mean())


def niveau_risque(pd):
    if pd <= 0.0399:
        return "Risque Très Faible"
    elif pd <= 0.105:
        return "Risque Faible"
    elif pd <= 0.199:
        return "Risque Moyen"
    elif pd <= 0.378:
        return "Risque Elevé"
    else:
        return "Risque Très Elevé"

df_reduit['niveau_risque'] = df_reduit['PD'].apply(niveau_risque)

# Vérifions la distribution
print(df_reduit['niveau_risque'].value_counts())


