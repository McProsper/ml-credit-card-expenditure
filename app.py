import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Dépenses par Carte de Crédit",
    page_icon="💳",
    layout="wide"
)

# Titre principal
st.title("Projet de Machine Learning - Prédiction des Dépenses")
st.markdown("**Modèle : KNN (K-Nearest Neighbors) optimisé**")
st.markdown("**Target : Expenditure (Dépenses)**")

# Chargement des données
@st.cache_data
def load_data():
    """Charge le dataset depuis le fichier CSV"""
    try:
        df = pd.read_csv('AER_credit_card_data.csv')
        return df
    except FileNotFoundError:
        st.error("Fichier AER_credit_card_data.csv non trouvé!")
        return None

# Fonction de preprocessing
def preprocess_data(df):
    """Prépare les données pour l'entraînement"""
    df_processed = df.copy()
    
    # Encodage des variables catégorielles
    le_card = LabelEncoder()
    le_owner = LabelEncoder()
    le_selfemp = LabelEncoder()
    
    df_processed['card_encoded'] = le_card.fit_transform(df_processed['card'])
    df_processed['owner_encoded'] = le_owner.fit_transform(df_processed['owner'])
    df_processed['selfemp_encoded'] = le_selfemp.fit_transform(df_processed['selfemp'])
    
    # Sélection des features
    features = ['reports', 'age', 'income', 'share', 'dependents', 
                'months', 'majorcards', 'active', 'card_encoded', 
                'owner_encoded', 'selfemp_encoded']
    
    X = df_processed[features]
    y = df_processed['expenditure']
    
    return X, y, le_card, le_owner, le_selfemp, features

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une section",
    ["Exploration des données", "Entraînement du modèle", "Prédictions", "Évaluation du modèle"]
)

# Chargement des données
df = load_data()

if df is not None:
    if page == "📊 Exploration des données":
        st.header("Exploration des Données")
        
        # Statistiques générales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre d'observations", len(df))
        col2.metric("Nombre de features", len(df.columns) - 1)
        col3.metric("Dépenses moyennes", f"${df['expenditure'].mean():.2f}")
        col4.metric("Dépenses max", f"${df['expenditure'].max():.2f}")
        
        # Aperçu des données
        st.subheader("Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualisations
        st.subheader("Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des dépenses
            fig = px.histogram(df, x='expenditure', nbins=50, 
                             title="Distribution des Dépenses",
                             labels={'expenditure': 'Dépenses ($)', 'count': 'Fréquence'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Relation income vs expenditure
            fig = px.scatter(df, x='income', y='expenditure', 
                           color='card', title="Income vs Expenditure",
                           labels={'income': 'Revenu', 'expenditure': 'Dépenses ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot des dépenses par propriétaire
            fig = px.box(df, x='owner', y='expenditure', 
                        title="Dépenses par Statut de Propriétaire",
                        labels={'owner': 'Propriétaire', 'expenditure': 'Dépenses ($)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Relation age vs expenditure
            fig = px.scatter(df, x='age', y='expenditure', 
                           color='owner', title="Age vs Expenditure",
                           labels={'age': 'Âge', 'expenditure': 'Dépenses ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Matrice de corrélation
        st.subheader("Matrice de corrélation")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Matrice de corrélation",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Entraînement du modèle":
        st.header("Entraînement et Optimisation du Modèle KNN")
        
        # Préprocessing
        X, y, le_card, le_owner, le_selfemp, features = preprocess_data(df)
        
        # Séparation train/test
        test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20, 5)
        random_state = st.number_input("Random state", min_value=0, max_value=100, value=42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.info(f"Données d'entraînement : {len(X_train)} échantillons | Données de test : {len(X_test)} échantillons")
        
        # Paramètres d'optimisation
        st.subheader("Paramètres d'optimisation")
        
        col1, col2 = st.columns(2)
        with col1:
            optimize = st.checkbox("Effectuer une optimisation (GridSearchCV)", value=True)
            n_neighbors_min = st.number_input("Nombre de voisins (min)", min_value=1, max_value=50, value=3)
            n_neighbors_max = st.number_input("Nombre de voisins (max)", min_value=1, max_value=50, value=20)
        
        with col2:
            weights_options = ['uniform', 'distance']
            weights = st.multiselect("Types de poids", weights_options, default=weights_options)
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            algorithm = st.multiselect("Algorithmes", algorithms, default=['auto'])
        
        if st.button("Entraîner et Optimiser le Modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                if optimize:
                    # GridSearchCV pour optimisation
                    param_grid = {
                        'n_neighbors': range(n_neighbors_min, n_neighbors_max + 1),
                        'weights': weights,
                        'algorithm': algorithm
                    }
                    
                    knn = KNeighborsRegressor()
                    grid_search = GridSearchCV(
                        knn, 
                        param_grid, 
                        cv=5, 
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    st.success("✅ Modèle optimisé avec succès!")
                    
                    # Affichage des meilleurs paramètres
                    st.subheader("Meilleurs paramètres trouvés")
                    st.json(best_params)
                    
                    st.metric("Meilleur score (CV)", f"{grid_search.best_score_:.4f}")
                else:
                    # Modèle simple sans optimisation
                    best_model = KNeighborsRegressor(n_neighbors=5)
                    best_model.fit(X_train_scaled, y_train)
                    best_params = {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}
                    st.success("✅ Modèle entraîné avec succès!")
                
                # Prédictions
                y_train_pred = best_model.predict(X_train_scaled)
                y_test_pred = best_model.predict(X_test_scaled)
                
                # Métriques
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Affichage des métriques
                st.subheader("📊 Métriques de performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE (Train)", f"${train_rmse:.2f}")
                    st.metric("RMSE (Test)", f"${test_rmse:.2f}")
                with col2:
                    st.metric("MAE (Train)", f"${train_mae:.2f}")
                    st.metric("MAE (Test)", f"${test_mae:.2f}")
                with col3:
                    st.metric("R² (Train)", f"{train_r2:.4f}")
                    st.metric("R² (Test)", f"{test_r2:.4f}")
                
                # Visualisation des prédictions
                st.subheader("Visualisation des prédictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prédictions vs Réalité (Train)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_train, 
                        y=y_train_pred,
                        mode='markers',
                        name='Prédictions',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[y_train.min(), y_train.max()],
                        y=[y_train.min(), y_train.max()],
                        mode='lines',
                        name='Ligne parfaite',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Prédictions vs Réalité (Train)",
                        xaxis_title="Valeurs réelles",
                        yaxis_title="Prédictions",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Prédictions vs Réalité (Test)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test, 
                        y=y_test_pred,
                        mode='markers',
                        name='Prédictions',
                        marker=dict(color='green', opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Ligne parfaite',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Prédictions vs Réalité (Test)",
                        xaxis_title="Valeurs réelles",
                        yaxis_title="Prédictions",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarde du modèle
                st.subheader("Sauvegarde du modèle")
                if st.button("Sauvegarder le modèle"):
                    with open('knn_model.pkl', 'wb') as f:
                        pickle.dump(best_model, f)
                    with open('scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    with open('label_encoders.pkl', 'wb') as f:
                        pickle.dump({'card': le_card, 'owner': le_owner, 'selfemp': le_selfemp}, f)
                    st.success("Modèle sauvegardé avec succès!")
                
                # Stockage en session state
                st.session_state['model'] = best_model
                st.session_state['scaler'] = scaler
                st.session_state['label_encoders'] = {'card': le_card, 'owner': le_owner, 'selfemp': le_selfemp}
                st.session_state['features'] = features
                st.session_state['best_params'] = best_params
    
    elif page == "Prédictions":
        st.header("Faire une Prédiction")
        
        # Vérifier si le modèle est entraîné
        if 'model' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord entraîner le modèle dans la section 'Entraînement du modèle'")
        else:
            st.info("Remplissez les informations ci-dessous pour prédire les dépenses")
            
            col1, col2 = st.columns(2)
            
            with col1:
                card = st.selectbox("Carte de crédit", ['yes', 'no'])
                owner = st.selectbox("Propriétaire", ['yes', 'no'])
                selfemp = st.selectbox("Travailleur indépendant", ['yes', 'no'])
                reports = st.number_input("Nombre de rapports", min_value=0, max_value=20, value=0)
                age = st.number_input("Âge", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
                income = st.number_input("Revenu", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
            
            with col2:
                share = st.number_input("Part (share)", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.6f")
                dependents = st.number_input("Nombre de dépendants", min_value=0, max_value=10, value=2)
                months = st.number_input("Nombre de mois", min_value=0, max_value=200, value=50)
                majorcards = st.selectbox("Cartes majeures", [0, 1])
                active = st.number_input("Nombre de cartes actives", min_value=0, max_value=50, value=6)
            
            if st.button("Prédire les Dépenses", type="primary"):
                # Préparation des données
                le_card = st.session_state['label_encoders']['card']
                le_owner = st.session_state['label_encoders']['owner']
                le_selfemp = st.session_state['label_encoders']['selfemp']
                scaler = st.session_state['scaler']
                model = st.session_state['model']
                
                # Encodage
                card_encoded = le_card.transform([card])[0]
                owner_encoded = le_owner.transform([owner])[0]
                selfemp_encoded = le_selfemp.transform([selfemp])[0]
                
                # Création du vecteur de features
                features_array = np.array([[
                    reports, age, income, share, dependents,
                    months, majorcards, active,
                    card_encoded, owner_encoded, selfemp_encoded
                ]])
                
                # Normalisation
                features_scaled = scaler.transform(features_array)
                
                # Prédiction
                prediction = model.predict(features_scaled)[0]
                
                # Affichage du résultat
                st.success(f"### 💰 Dépenses prédites : **${prediction:.2f}**")
                
                # Informations supplémentaires
                st.subheader("Détails de la prédiction")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Revenu", f"${income:.2f}")
                with col2:
                    st.metric("Âge", f"{age:.1f} ans")
                with col3:
                    st.metric("Cartes actives", active)
    
    elif page == "📈 Évaluation du modèle":
        st.header("Évaluation Détaillée du Modèle")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord entraîner le modèle dans la section 'Entraînement du modèle'")
        else:
            # Préprocessing
            X, y, le_card, le_owner, le_selfemp, features = preprocess_data(df)
            
            # Séparation train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalisation
            scaler = st.session_state['scaler']
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = st.session_state['model']
            
            # Prédictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Métriques détaillées
            st.subheader("Métriques de Performance")
            
            metrics_train = {
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'R²': r2_score(y_train, y_train_pred),
                'MAPE': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
            }
            
            metrics_test = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'R²': r2_score(y_test, y_test_pred),
                'MAPE': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Ensemble d'entraînement")
                for metric, value in metrics_train.items():
                    if metric == 'MAPE':
                        st.metric(metric, f"{value:.2f}%")
                    elif metric == 'R²':
                        st.metric(metric, f"{value:.4f}")
                    else:
                        st.metric(metric, f"${value:.2f}")
            
            with col2:
                st.markdown("### 📊 Ensemble de test")
                for metric, value in metrics_test.items():
                    if metric == 'MAPE':
                        st.metric(metric, f"{value:.2f}%")
                    elif metric == 'R²':
                        st.metric(metric, f"{value:.4f}")
                    else:
                        st.metric(metric, f"${value:.2f}")
            
            # Distribution des erreurs
            st.subheader("Analyse des Erreurs")
            
            errors_train = y_train - y_train_pred
            errors_test = y_test - y_test_pred
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    x=errors_train,
                    nbins=50,
                    title="Distribution des Erreurs (Train)",
                    labels={'x': 'Erreur ($)', 'count': 'Fréquence'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    x=errors_test,
                    nbins=50,
                    title="Distribution des Erreurs (Test)",
                    labels={'x': 'Erreur ($)', 'count': 'Fréquence'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Importance des features (basée sur la corrélation)
            st.subheader("Importance des Features")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Corrélation avec Expenditure': [abs(df[feat].corr(df['expenditure'])) if feat in df.columns else 0 for feat in features]
            })
            feature_importance = feature_importance.sort_values('Corrélation avec Expenditure', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Corrélation avec Expenditure',
                y='Feature',
                orientation='h',
                title="Corrélation des Features avec les Dépenses"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Meilleurs paramètres
            if 'best_params' in st.session_state:
                st.subheader("Paramètres du Modèle Optimisé")
                st.json(st.session_state['best_params'])

else:
    st.error("Impossible de charger les données. Veuillez vérifier que le fichier AER_credit_card_data.csv est présent.")
