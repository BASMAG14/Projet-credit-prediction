import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import joblib

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")


# Configuration de la page
st.set_page_config(
    page_title="Credit Guard - Analyse de Risque",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS pour un look "Banque Digitale"
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; background-color: #1a237e; color: white; border-radius: 10px; height: 3em; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #1a237e; text-align: center; }
    .result-card { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 20px; }
    .good-risk { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .bad-risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

# Chargement du modèle et du scaler
@st.cache_resource
def load_assets():
    try:
        # Assurez-vous de sauvegarder ces fichiers dans votre notebook avec joblib
        model = joblib.load('credit_model.pkl')
        scaler = joblib.load('scaler.pkl')
        selected_features = joblib.load('selected_features.pkl')
        return model, scaler, selected_features
    except:
        return None, None, None

model, scaler, selected_features = load_assets()

# Header
st.markdown("<h1>🏦 CREDIT GUARD AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Système d'aide à la décision pour l'octroi de prêt bancaire</p>", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ **Fichiers manquants :** Veuillez d'abord entraîner et sauvegarder votre modèle (`credit_model.pkl`, `scaler.pkl`) dans votre notebook.")
    st.stop()

# --- SIDEBAR : INFORMATIONS CLIENT ---
st.sidebar.header("📋 Profil du Client")

# Variables numériques
st.sidebar.subheader("Données chiffrées")
duration = st.sidebar.slider("Durée du crédit (mois)", 4, 72, 24)
credit_amount = st.sidebar.number_input("Montant du crédit (DM)", 250, 20000, 3000)
age = st.sidebar.slider("Âge du demandeur", 18, 75, 35)

# Variables catégorielles (Nous utilisons les codes du dataset UCI)
st.sidebar.subheader("Statut financier")
checking_status = st.sidebar.selectbox("Statut compte courant", ["A11 : < 0 DM", "A12 : 0-200 DM", "A13 : > 200 DM", "A14 : Pas de compte"])
savings_status = st.sidebar.selectbox("Épargne", ["A61 : < 100 DM", "A62 : 100-500 DM", "A63 : 500-1000 DM", "A64 : > 1000 DM", "A65 : Inconnu"])
employment = st.sidebar.selectbox("Ancienneté emploi", ["A71 : Sans emploi", "A72 : < 1 an", "A73 : 1-4 ans", "A74 : 4-7 ans", "A75 : > 7 ans"])

# Bouton de prédiction
predict_btn = st.sidebar.button("ANALYSER LE RISQUE")

# --- MAIN PAGE : RÉSULTATS ---
if predict_btn:
    # 1. Préparation des données (Simulation de l'encodage One-Hot)
    # Note: Dans un vrai projet, il faut reproduire exactement les colonnes de df_ml
    
    # Création d'un dictionnaire avec toutes les features à 0
    input_dict = {feat: 0 for feat in selected_features}
    
    # Remplissage des valeurs numériques
    if 'duration' in input_dict: input_dict['duration'] = duration
    if 'credit_amount' in input_dict: input_dict['credit_amount'] = credit_amount
    if 'age' in input_dict: input_dict['age'] = age
    
    # Activation de la colonne One-Hot correspondante
    check_col = f"checking_status_{checking_status.split(' : ')[0]}"
    if check_col in input_dict: input_dict[check_col] = 1
    
    # Transformation en DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 2. Scaling
    num_cols = ['duration', 'credit_amount', 'age']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # 3. Prédiction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    risk_score = probability[1] * 100 # Probabilité de "Mauvais" (1)

    # 4. Affichage créatif
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Verdict du Système")
        if prediction == 0:
            st.markdown(f"""
                <div class="result-card good-risk">
                    <h3>✅ ACCORDÉ</h3>
                    <p style="font-size: 24px;"><b>Risque Faible</b></p>
                    <p>Le profil présente des garanties suffisantes.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card bad-risk">
                    <h3>❌ REFUSÉ</h3>
                    <p style="font-size: 24px;"><b>Risque Élevé</b></p>
                    <p>Le profil présente un risque de défaut important.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("Score de Probabilité")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Indice de Risque (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1a237e"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#f8d7da"}
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Détails techniques
    with st.expander("🔍 Voir les détails de l'analyse"):
        st.write(f"**Montant par mois :** {round(credit_amount/duration, 2)} DM")
        st.write(f"**Variables clés détectées :** {', '.join(selected_features[:5])}")

else:
    # Page d'accueil avant prédiction
    st.image("https://images.unsplash.com/photo-1550565118-3d1428df7300?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Modèle", "Random Forest")
    col2.metric("Précision", "76.5%")
    col3.metric("Variables RFE", "10")
    
    st.info("💡 **Conseil :** Utilisez la barre latérale pour saisir les informations du client et cliquez sur le bouton d'analyse.")

# --- ENREGISTREMENT DES FICHIERS ---
st.write("✅ **Fichiers sauvegardés :**")
st.write(f"credit_model.pkl")
st.write(f"scaler.pkl")
st.write(f"selected_features.pkl")