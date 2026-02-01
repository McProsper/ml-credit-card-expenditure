# Application de Machine Learning - Prédiction des Dépenses par Carte de Crédit

Application Streamlit pour prédire les dépenses (Expenditure) en utilisant un modèle KNN optimisé.

## 📋 Description

Cette application utilise un modèle K-Nearest Neighbors (KNN) optimisé avec GridSearchCV pour prédire les dépenses des clients basées sur leurs caractéristiques (âge, revenu, statut de propriétaire, etc.).

## 🚀 Installation et Utilisation Locale

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation

1. Cloner ou télécharger ce dossier
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 🌐 Déploiement sur Streamlit Cloud

### Méthode 1 : Via Streamlit Cloud (Recommandé)

1. **Créer un compte** sur [Streamlit Cloud](https://streamlit.io/cloud)

2. **Connecter votre repository GitHub** :
   - Créez un repository GitHub avec vos fichiers
   - Assurez-vous que les fichiers suivants sont présents :
     - `app.py`
     - `requirements.txt`
     - `AER_credit_card_data.csv`
     - `README.md` (optionnel)

3. **Déployer** :
   - Allez sur [share.streamlit.io](https://share.streamlit.io)
   - Cliquez sur "New app"
   - Sélectionnez votre repository
   - Spécifiez le fichier principal : `app.py`
   - Cliquez sur "Deploy"

### Méthode 2 : Via Streamlit CLI

```bash
# Installer Streamlit CLI
pip install streamlit

# Se connecter à Streamlit Cloud
streamlit login

# Déployer l'application
streamlit deploy app.py
```

## 📁 Structure du Projet

```
.
├── app.py                          # Application Streamlit principale
├── AER_credit_card_data.csv        # Dataset
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation
└── knn_model.pkl                  # Modèle sauvegardé (généré après entraînement)
```

## 🎯 Fonctionnalités

L'application comprend 4 sections principales :

1. **📊 Exploration des données** :
   - Statistiques descriptives
   - Visualisations interactives
   - Matrice de corrélation

2. **🤖 Entraînement du modèle** :
   - Optimisation automatique avec GridSearchCV
   - Ajustement des hyperparamètres (nombre de voisins, poids, algorithme)
   - Métriques de performance (RMSE, MAE, R²)
   - Visualisation des prédictions

3. **🔮 Prédictions** :
   - Interface interactive pour faire des prédictions
   - Saisie des caractéristiques du client
   - Affichage des dépenses prédites

4. **📈 Évaluation du modèle** :
   - Métriques détaillées
   - Analyse des erreurs
   - Importance des features

## 🔧 Optimisation du Modèle

Le modèle KNN est optimisé via GridSearchCV avec validation croisée (5 folds) sur les paramètres suivants :

- **n_neighbors** : Nombre de voisins (par défaut : 3 à 20)
- **weights** : Type de pondération ('uniform' ou 'distance')
- **algorithm** : Algorithme de recherche ('auto', 'ball_tree', 'kd_tree', 'brute')

## 📊 Dataset

Le dataset contient 1319 observations avec les features suivantes :
- `card` : Possession d'une carte de crédit (yes/no)
- `reports` : Nombre de rapports
- `age` : Âge
- `income` : Revenu
- `share` : Part
- `expenditure` : Dépenses (target)
- `owner` : Statut de propriétaire (yes/no)
- `selfemp` : Travailleur indépendant (yes/no)
- `dependents` : Nombre de dépendants
- `months` : Nombre de mois
- `majorcards` : Cartes majeures
- `active` : Nombre de cartes actives

## 🛠️ Technologies Utilisées

- **Streamlit** : Framework pour l'interface web
- **Scikit-learn** : Machine Learning (KNN, GridSearchCV)
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **Plotly** : Visualisations interactives

## 📝 Notes

- Le modèle est sauvegardé automatiquement après l'entraînement
- Les données sont mises en cache pour améliorer les performances
- L'application supporte le preprocessing automatique des variables catégorielles

## 👤 Auteur

Application développée pour le projet de Machine Learning - Prédiction des Dépenses par Carte de Crédit

## 📄 Licence

Ce projet est fourni à des fins éducatives.
