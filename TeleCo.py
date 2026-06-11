
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(page_title='Telco Churn Dashboard', layout='wide')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global Reset & Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

.stApp {
    background: #070d1a !important;
    color: #e2e8f4 !important;
}

.block-container {
    width: 100% !important;
    max-width: 100% !important;
    padding: 2rem 4rem !important;
    background: transparent !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0b1528 !important;
    border-right: 1px solid #1a2d50 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem !important;
}
[data-testid="stSidebar"] label {
    color: #7a9cc4 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #3a8fff !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Headings ────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: #e2e8f4 !important;
    text-align: center !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 2.2rem !important; font-weight: 800 !important; }

/* ── Tabs ────────────────────────────────────────────────────────── */
[data-testid="stTabs"] {
    border-bottom: 1px solid #1a2d50 !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #5a7a9e !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #3a8fff !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #3a8fff !important;
    border-bottom: 2px solid #3a8fff !important;
    background: transparent !important;
}
[data-testid="stTabsContent"] {
    background: transparent !important;
    padding-top: 1.5rem !important;
}

/* ── Metric Cards ────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0d1f3c, #0f2448) !important;
    border: 1px solid #1e3660 !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.2rem !important;
    text-align: center !important;
    margin: auto !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(58,143,255,0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(58,143,255,0.15), inset 0 1px 0 rgba(58,143,255,0.15) !important;
}
[data-testid="metric-container"] label {
    color: #5a8fbf !important;
    font-size: 0.72rem !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f4 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #3a8fff !important;
}

/* ── Dividers ────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #1a2d50 !important;
    margin: 1.5rem 0 !important;
    opacity: 0.7 !important;
}

/* ── Charts ──────────────────────────────────────────────────────── */
.stPlotlyChart {
    background: #0d1f3c !important;
    border: 1px solid #1e3660 !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    margin: auto !important;
    overflow: hidden !important;
}

/* ── Form & Inputs ───────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: #0b1528 !important;
    border: 1px solid #1a2d50 !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4) !important;
}

.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div {
    background: #0f2040 !important;
    border: 1px solid #1e3660 !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
    transition: border-color 0.2s ease !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div:hover {
    border-color: #3a8fff !important;
}

.stSelectbox label,
.stSlider label,
.stNumberInput label {
    color: #7a9cc4 !important;
    font-size: 0.8rem !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.04em !important;
    font-weight: 600 !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: #3a8fff !important;
}
[data-testid="stSlider"] > div > div > div {
    background: #1a2d50 !important;
}

/* ── Section headers inside form ─────────────────────────────────── */
[data-testid="stForm"] h4 {
    color: #3a8fff !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    text-align: left !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #1a2d50 !important;
}

/* ── Submit Button ───────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #1a5fd4, #3a8fff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 20px rgba(58,143,255,0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #2268e0, #55a0ff) !important;
    box-shadow: 0 6px 28px rgba(58,143,255,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: #0d1f3c !important;
    border: 1px solid #1e3660 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Prediction Result Boxes ─────────────────────────────────────── */
.predict-box-churn {
    background: linear-gradient(135deg, rgba(255,75,75,0.08), rgba(255,75,75,0.15));
    border: 1px solid rgba(255,75,75,0.5);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 12px;
    box-shadow: 0 0 40px rgba(255,75,75,0.1), inset 0 1px 0 rgba(255,75,75,0.15);
    backdrop-filter: blur(4px);
}
.predict-box-no-churn {
    background: linear-gradient(135deg, rgba(0,200,83,0.08), rgba(0,200,83,0.15));
    border: 1px solid rgba(0,200,83,0.5);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 12px;
    box-shadow: 0 0 40px rgba(0,200,83,0.1), inset 0 1px 0 rgba(0,200,83,0.15);
    backdrop-filter: blur(4px);
}
.predict-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
    color: #e2e8f4;
}
.predict-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #9ab4d4;
    line-height: 1.6;
}

/* ── Page title accent ───────────────────────────────────────────── */
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #3a8fff, #7ac0ff, #e2e8f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.dash-subtitle {
    text-align: center;
    color: #4a6a8a;
    font-size: 0.85rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: 'Syne', sans-serif;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('Telco_Customer_Churn_Modified.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop_duplicates(inplace=True)

# ── Chart theme ────────────────────────────────────────────────────────────────
CHURN_COLORS = {'Yes': '#ff4b4b', 'No': '#3a8fff'}
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#9ab4d4', size=12),
    title_font=dict(family='Syne, sans-serif', color='#c8d8f0', size=15),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e3660', borderwidth=1,
                font=dict(color='#9ab4d4')),
    xaxis=dict(gridcolor='#1a2d50', linecolor='#1e3660', tickfont=dict(color='#7a9cc4')),
    yaxis=dict(gridcolor='#1a2d50', linecolor='#1e3660', tickfont=dict(color='#7a9cc4')),
    margin=dict(l=20, r=20, t=50, b=20),
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('xgb_churn_model.pkl')

model = load_model()

# ── Sidebar Filters ────────────────────────────────────────────────────────────
contract_filter = st.sidebar.multiselect('Filter By Contract Type', options=df['Contract'].unique())
contract_filter = df['Contract'].unique() if contract_filter == [] else contract_filter

payment_filter = st.sidebar.multiselect('Filter By Payment Method', options=df['PaymentMethod'].unique())
payment_filter = df['PaymentMethod'].unique() if payment_filter == [] else payment_filter

internet_filter = st.sidebar.multiselect('Filter By Internet Service', options=df['InternetService'].unique())
internet_filter = df['InternetService'].unique() if internet_filter == [] else internet_filter

df_filtered = df[
    (df['Contract'].isin(contract_filter)) &
    (df['PaymentMethod'].isin(payment_filter)) &
    (df['InternetService'].isin(internet_filter))
]

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(['📊 Dashboard', '🤖 Churn Predictor'])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='dash-title'>Telco Customer Churn</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-subtitle'>Real-time customer retention intelligence</div>", unsafe_allow_html=True)
    st.markdown('---')

    churned   = df_filtered[df_filtered['Churn'] == 'Yes'].shape[0]
    churn_rate = round(churned / len(df_filtered) * 100, 2) if len(df_filtered) > 0 else 0
    avg_tenure = round(df_filtered['tenure'].mean(), 1)

    left_space, col1, col2, col3, col4, right_space = st.columns([2, 3, 3, 3, 3, 2])
    with col1: st.metric('Total Customers',     len(df_filtered))
    with col2: st.metric('Total Churned',        churned)
    with col3: st.metric('Churn Rate',           f'{churn_rate}%')
    with col4: st.metric('Avg Tenure (months)',  avg_tenure)

    st.markdown('---')

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(df_filtered, x='Contract', color='Churn', barmode='group',
                           title='Churn by Contract Type', color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_filtered, x='PaymentMethod', color='Churn', barmode='group',
                           title='Churn by Payment Method', color_discrete_map=CHURN_COLORS)
        fig.update_layout(xaxis_tickangle=-20, **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.histogram(df_filtered, x='InternetService', color='Churn', barmode='group',
                           title='Churn by Internet Service', color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(df_filtered, x='tenure', color='Churn', nbins=30,
                           title='Churn by Tenure', opacity=0.8, color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_filtered, x='MonthlyCharges', color='Churn', nbins=30,
                           title='Churn by Monthly Charges', opacity=0.8, color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.histogram(df_filtered, x='SeniorCitizen', color='Churn', barmode='group',
                           title='Churn by Senior Citizen', color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(df_filtered, x='OnlineSecurity', color='Churn', barmode='group',
                           title='Churn by Online Security', color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_filtered, x='TechSupport', color='Churn', barmode='group',
                           title='Churn by Tech Support', color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.histogram(df_filtered, x='PaperlessBilling', color='Churn', barmode='group',
                           title='Churn by Paperless Billing', color_discrete_map=CHURN_COLORS)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='dash-title'>🤖 Churn Predictor</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#4a6a8a; font-family:Syne,sans-serif; font-size:0.85rem; letter-spacing:0.1em; text-transform:uppercase;'>Fill in the customer details below and click <b style=\"color:#3a8fff\">Predict</b></p>", unsafe_allow_html=True)
    st.markdown('---')

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form('prediction_form'):

        st.markdown("#### 👤 Customer Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender        = st.selectbox('Gender',         ['Female', 'Male'])
        with c2:
            senior        = st.selectbox('Senior Citizen', ['NO', 'Yes'])
        with c3:
            partner       = st.selectbox('Partner',        ['Yes', 'No'])
        with c4:
            dependents    = st.selectbox('Dependents',     ['No', 'Yes'])

        st.markdown("#### 📞 Phone & Internet Services")
        c1, c2, c3 = st.columns(3)
        with c1:
            phone_service = st.selectbox('Phone Service',   ['No', 'Yes'])
        with c2:
            multi_lines   = st.selectbox('Multiple Lines',  ['No phone service', 'No', 'Yes'])
        with c3:
            internet      = st.selectbox('Internet Service',['DSL', 'Fiber optic', 'No'])

        st.markdown("#### 🔒 Online Add-ons")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            online_sec    = st.selectbox('Online Security',   ['No', 'Yes', 'No internet service'])
        with c2:
            online_bkp    = st.selectbox('Online Backup',     ['Yes', 'No', 'No internet service'])
        with c3:
            device_prot   = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        with c4:
            tech_support  = st.selectbox('Tech Support',      ['No', 'Yes', 'No internet service'])

        c1, c2 = st.columns(2)
        with c1:
            streaming_tv  = st.selectbox('Streaming TV',     ['No', 'Yes', 'No internet service'])
        with c2:
            streaming_mv  = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

        st.markdown("#### 💳 Billing & Contract")
        c1, c2, c3 = st.columns(3)
        with c1:
            contract      = st.selectbox('Contract',         ['Month-to-month', 'One year', 'Two year'])
        with c2:
            paperless     = st.selectbox('Paperless Billing',['Yes', 'No'])
        with c3:
            payment       = st.selectbox('Payment Method',   ['Electronic check', 'Mailed check',
                                                               'Bank transfer (automatic)',
                                                               'Credit card (automatic)'])

        st.markdown("#### 💰 Charges & Tenure")
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure         = st.slider('Tenure (months)',    min_value=1,   max_value=72,     value=12)
        with c2:
            monthly        = st.slider('Monthly Charges ($)',min_value=18,  max_value=119,    value=65)
        with c3:
            total          = st.number_input('Total Charges ($)', min_value=18.0, max_value=8700.0,
                                             value=float(tenure * monthly), step=1.0)

        submitted = st.form_submit_button('🔍 Predict Churn', use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────────
    if submitted:
        input_data = pd.DataFrame([{
            'gender':           gender,
            'SeniorCitizen':    senior,
            'Partner':          partner,
            'Dependents':       dependents,
            'tenure':           tenure,
            'PhoneService':     phone_service,
            'MultipleLines':    multi_lines,
            'InternetService':  internet,
            'OnlineSecurity':   online_sec,
            'OnlineBackup':     online_bkp,
            'DeviceProtection': device_prot,
            'TechSupport':      tech_support,
            'StreamingTV':      streaming_tv,
            'StreamingMovies':  streaming_mv,
            'Contract':         contract,
            'PaperlessBilling': paperless,
            'PaymentMethod':    payment,
            'MonthlyCharges':   monthly,
            'TotalCharges':     total,
        }])

        prediction = model.predict(input_data)[0]

        st.markdown('---')
        _, result_col, _ = st.columns([1, 2, 1])
        with result_col:
            if prediction == 1:
                st.markdown("""
                <div class="predict-box-churn">
                    <div class="predict-title">⚠️ High Churn Risk</div>
                    <div class="predict-subtitle">This customer is <b>likely to churn</b>.<br>Consider a retention offer or proactive outreach.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="predict-box-no-churn">
                    <div class="predict-title">✅ Low Churn Risk</div>
                    <div class="predict-subtitle">This customer is <b>likely to stay</b>.<br>No immediate action required.</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Input summary table ─────────────────────────────────────────────
        st.markdown('---')
        st.markdown("#### 📋 Input Summary")
        summary = input_data.T.rename(columns={0: 'Value'})
        st.dataframe(summary, use_container_width=True)
