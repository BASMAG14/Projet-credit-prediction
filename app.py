import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go



st.set_page_config(
    page_title="LoanGuard ML - Analyse de Risque",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; background-color: #1a237e; color: white; border-radius: 10px; height: 3em; font-weight: bold; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 { color: #1a237e; text-align: center; }
    .result-card { padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 20px; }
    .good-risk { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .bad-risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .hero {
        position: relative;
        min-height: 560px;
        border-radius: 18px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 15px;
        margin-bottom: 20px;
        background-image:
            linear-gradient(rgba(8, 30, 70, 0.65), rgba(8, 30, 70, 0.65)),
            url('https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
    }
    .hero-content {
        text-align: center;
        color: white;
        max-width: 820px;
        padding: 20px;
    }
    .hero-content h1 {
        color: white;
        margin-bottom: 10px;
        font-size: 3rem;
    }
    .hero-content p {
        font-size: 1.15rem;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    .hero-btn {
        display: inline-block;
        margin-top: 14px;
        background: #1a4fb7;
        color: white !important;
        text-decoration: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
    }
    .top-nav {
        position: absolute;
        top: 18px;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        color: #dbe7ff;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .top-nav a {
        color: #dbe7ff;
        text-decoration: none;
    }
    .top-nav a:hover {
        color: #ffffff;
    }
    .brand {
        color: #ffffff;
        font-weight: 700;
    }
    .sub-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 28px;
        margin-bottom: 16px;
        font-weight: 600;
    }
    .sub-nav a {
        color: #1a237e;
        text-decoration: none;
    }
    .sub-nav a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Chargement du modèle et du scaler
@st.cache_resource
def load_assets():
    try:
   
        model = joblib.load('credit_model.pkl')
        scaler = joblib.load('scaler.pkl')
        selected_features = joblib.load('selected_features.pkl')
        return model, scaler, selected_features
    except:
        return None, None, None

model, scaler, selected_features = load_assets()

if model is None:
    st.warning("⚠️ **Fichiers manquants :** Veuillez d'abord entraîner et sauvegarder votre modèle (`credit_model.pkl`, `scaler.pkl`) dans votre notebook.")
    st.stop()

page_from_url = st.query_params.get("page", "home")
allowed_pages = {"home", "predict", "team", "contact"}
if page_from_url not in allowed_pages:
    page_from_url = "home"
st.session_state.page = page_from_url

if st.session_state.page == "home":
    st.markdown(
        """
        <div class="hero">
            <div class="top-nav">
                <a class="brand" href="?page=home">LoanGuard ML</a>
                <a href="?page=home">Home</a>
                <a href="?page=team">Team</a>
                <a href="?page=contact">Contact</a>
            </div>
            <div class="hero-content">
                <h1>Estimez le risque de crédit avec LoanGuard ML</h1>
                <p>Prédiction rapide • Score de risque • Aide à la décision instantanée</p>
                <a class="hero-btn" href="?page=predict">Lancer la prédiction</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Renseigne ton profil via la page prédiction puis clique sur **ANALYSER LE RISQUE**.")

# --- PAGE TEAM ---
elif st.session_state.page == "team":
    st.markdown(
        """
        <div class="sub-nav">
            <a href="?page=home">LoanGuard ML</a>
            <a href="?page=home">Home</a>
            <a href="?page=team">Team</a>
            <a href="?page=contact">Contact</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1>👥 Équipe LoanGuard ML</h1>", unsafe_allow_html=True)
    st.write("Notre équipe développe une solution IA pour évaluer le risque de crédit de façon rapide et fiable.")
    st.write("Compétences clés : Data Science, Machine Learning, Développement Streamlit et analyse financière.")

# --- PAGE CONTACT ---
elif st.session_state.page == "contact":
    st.markdown(
        """
        <div class="sub-nav">
            <a href="?page=home">LoanGuard ML</a>
            <a href="?page=home">Home</a>
            <a href="?page=team">Team</a>
            <a href="?page=contact">Contact</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1>📩 Contact</h1>", unsafe_allow_html=True)
    st.write("Pour toute démonstration ou collaboration autour de LoanGuard ML :")
    st.write("- Email : contact@loanguard-ml.com")
    st.write("- Téléphone : +212 6 00 00 00 00")

# --- PAGE PRÉDICTION ---
else:
    st.markdown(
        """
        <div class="sub-nav">
            <a href="?page=home">LoanGuard ML</a>
            <a href="?page=home">Home</a>
            <a href="?page=team">Team</a>
            <a href="?page=contact">Contact</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1> LOANGUARD ML</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Système d'aide à la décision pour l'octroi de prêt bancaire</p>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1, 1])
    with right:
        if st.button("← Retour à l'accueil"):
            st.query_params["page"] = "home"
            st.session_state.page = "home"
            st.rerun()

    # --- SIDEBAR : INFORMATIONS CLIENT ---
    st.sidebar.header("📋 Profil du Client")

    # Variables numériques
    st.sidebar.subheader("Données chiffrées")
    duration = st.sidebar.slider("Durée du crédit (mois)", 4, 72, 24)
    credit_amount = st.sidebar.number_input("Montant du crédit (DM)", 250, 20000, 3000)
    age = st.sidebar.slider("Âge du demandeur", 18, 75, 35)

    # Variables catégorielles (codes du dataset UCI)
    st.sidebar.subheader("Statut financier")
    checking_status = st.sidebar.selectbox("Statut compte courant", ["A11 : < 0 DM", "A12 : 0-200 DM", "A13 : > 200 DM", "A14 : Pas de compte"])
    savings_status = st.sidebar.selectbox("Épargne", ["A61 : < 100 DM", "A62 : 100-500 DM", "A63 : 500-1000 DM", "A64 : > 1000 DM", "A65 : Inconnu"])
    employment = st.sidebar.selectbox("Ancienneté emploi", ["A71 : Sans emploi", "A72 : < 1 an", "A73 : 1-4 ans", "A74 : 4-7 ans", "A75 : > 7 ans"])

    # Btn
    predict_btn = st.sidebar.button("ANALYSER LE RISQUE")



    if predict_btn:
        
        expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else selected_features
        input_dict = {feat: 0 for feat in expected_features}

        
        input_dict['duration'] = duration
        input_dict['credit_amount'] = credit_amount
        input_dict['age'] = age

        if 'age_group' in input_dict:
            if age < 25:
                input_dict['age_group'] = 0
            elif age < 45:
                input_dict['age_group'] = 1
            else:
                input_dict['age_group'] = 2
        if 'credit_per_duration' in input_dict:
            input_dict['credit_per_duration'] = credit_amount / max(duration, 1)
        if 'high_credit' in input_dict:
            input_dict['high_credit'] = 1 if credit_amount > 5000 else 0
        if 'long_term_credit' in input_dict:
            input_dict['long_term_credit'] = 1 if duration > 24 else 0

        check_col = f"checking_status_{checking_status.split(' : ')[0]}"
        if check_col in input_dict:
            input_dict[check_col] = 1

        input_df = pd.DataFrame([input_dict])

        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        if hasattr(scaler, "feature_names_in_"):
            num_cols = list(scaler.feature_names_in_)
        else:
            num_cols = ['age', 'credit_amount', 'duration']
        num_cols = [c for c in num_cols if c in input_df.columns]
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        risk_score = probability[1] * 100
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Verdict du Système")
            if prediction == 0:
                st.markdown(
                    """
                    <div class="result-card good-risk">
                        <h3>✅ ACCORDÉ</h3>
                        <p style="font-size: 24px;"><b>Risque Faible</b></p>
                        <p>Le profil présente des garanties suffisantes.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-card bad-risk">
                        <h3>❌ REFUSÉ</h3>
                        <p style="font-size: 24px;"><b>Risque Élevé</b></p>
                        <p>Le profil présente un risque de défaut important.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            st.subheader("Score de Probabilité")
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': "Indice de Risque (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1a237e"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d4edda"},
                            {'range': [30, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#f8d7da"}
                        ],
                    }
                )
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander(" Voir les détails de l'analyse"):
            st.write(f"**Montant par mois :** {round(credit_amount/duration, 2)} DM")
            st.write(f"**Variables clés détectées :** {', '.join(selected_features[:5])}")
    else:
        st.info("Utilise la barre latérale pour saisir les informations puis clique sur **ANALYSER LE RISQUE**.")
