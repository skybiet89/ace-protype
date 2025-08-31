import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ACE Marketing Dashboard — 50 Buyers Prototype", layout="wide")

# ----------------------
# Helpers: Synthetic 50-buyer generator
# ----------------------
def generate_50_buyers(seed=42):
    np.random.seed(seed)
    n = 50
    ids = [f"B{1000+i}" for i in range(n)]
    names = [f"Buyer_{i}" for i in range(n)]
    emails = [f"buyer{i}@example.com" for i in range(n)]

    age = np.random.choice(range(18, 66), size=n)
    income_bracket = np.random.choice(["Low","Middle","High","VeryHigh"], size=n, p=[0.25,0.45,0.2,0.1])
    city_tier = np.random.choice([1,2,3], size=n, p=[0.3,0.5,0.2])
    culture = np.random.choice(["North","South","East","West","GlobalExposed"], size=n)
    education = np.random.choice(["HighSchool","Graduate","PostGrad","Professional"], size=n, p=[0.25,0.45,0.2,0.1])

    engagement = np.random.randint(10,101,size=n)  # 0-100
    freq_12m = np.random.poisson(lam=2, size=n)
    avg_order_value = np.round(np.random.normal(loc=70, scale=40, size=n).clip(5),2)
    recency_days = np.random.randint(0,120,size=n)
    preferred_channel = np.random.choice(["Email","Social","App","Website","Chatbot"], size=n, p=[0.5,0.15,0.15,0.15,0.05])
    price_sensitivity = np.random.choice(["Low","Medium","High"], size=n, p=[0.4,0.4,0.2])

    # Rule-based labeling for actions (used as ground truth to train a simple model)
    actions = []
    for i in range(n):
        score = (engagement[i]/100)*0.4 + (min(freq_12m[i],5)/5)*0.3 + (1 - recency_days[i]/120)*0.3
        if score > 0.65 and preferred_channel[i]=="Email":
            actions.append("SendOffer")
        elif 0.45 < score <= 0.65 and preferred_channel[i] in ["Chatbot","App"]:
            actions.append("RetentionChat")
        elif price_sensitivity[i]=="High" and income_bracket[i]=="Low":
            actions.append("DiscountIncentive")
        elif income_bracket[i] in ["High","VeryHigh"] and engagement[i] > 60:
            actions.append("UpsellCrossSell")
        else:
            actions.append("Awareness")

    df = pd.DataFrame({
        "buyer_id": ids,
        "name": names,
        "email": emails,
        "age": age,
        "income_bracket": income_bracket,
        "city_tier": city_tier,
        "culture": culture,
        "education": education,
        "engagement": engagement,
        "freq_12m": freq_12m,
        "avg_order_value": avg_order_value,
        "recency_days": recency_days,
        "preferred_channel": preferred_channel,
        "price_sensitivity": price_sensitivity,
        "action_label": actions
    })
    return df


# ----------------------
# Preprocess & Model
# ----------------------

def preprocess_and_train(df):
    dfc = df.copy()
    # Simple encodings
    le_income = LabelEncoder().fit(dfc['income_bracket'])
    dfc['income_code'] = le_income.transform(dfc['income_bracket'])
    le_culture = LabelEncoder().fit(dfc['culture'])
    dfc['culture_code'] = le_culture.transform(dfc['culture'])
    le_edu = LabelEncoder().fit(dfc['education'])
    dfc['edu_code'] = le_edu.transform(dfc['education'])
    le_channel = LabelEncoder().fit(dfc['preferred_channel'])
    dfc['channel_code'] = le_channel.transform(dfc['preferred_channel'])
    le_price = LabelEncoder().fit(dfc['price_sensitivity'])
    dfc['price_code'] = le_price.transform(dfc['price_sensitivity'])

    feature_cols = ["age","income_code","city_tier","culture_code","edu_code",
                    "engagement","freq_12m","avg_order_value","recency_days","channel_code","price_code"]
    X = dfc[feature_cols]
    y = dfc['action_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc, feature_cols


# ----------------------
# Session: generate data & train
# ----------------------
if 'buyers_df' not in st.session_state:
    st.session_state['buyers_df'] = generate_50_buyers(seed=123)
    st.session_state['action_log'] = []

model, test_acc, feat_cols = preprocess_and_train(st.session_state['buyers_df'])

# Predict probabilities and top action
proba = model.predict_proba(st.session_state['buyers_df'][feat_cols])
classes = model.classes_
max_prob = proba.max(axis=1)
pred = model.predict(st.session_state['buyers_df'][feat_cols])
st.session_state['buyers_df']['pred_action'] = pred
st.session_state['buyers_df']['pred_prob'] = np.round(max_prob,3)

# ----------------------
# Top KPI bar (top of page)
# ----------------------
st.title("ACE Marketing — 50 Buyer Prototype")
st.markdown("**Top-level monitoring (live)** — model-backed action recommendations and execution log")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Buyers", len(st.session_state['buyers_df']))
with k2:
    st.metric("Model Test Accuracy", f"{test_acc*100:.1f}%")
with k3:
    st.metric("Avg Engagement", f"{st.session_state['buyers_df']['engagement'].mean():.1f}")
with k4:
    st.metric("% High Income (High+VeryHigh)", f"{(st.session_state['buyers_df']['income_bracket'].isin(['High','VeryHigh']).mean()*100):.1f}%")
with k5:
    st.metric("Avg Recency (days)", f"{st.session_state['buyers_df']['recency_days'].mean():.1f}")

st.markdown("---")

# ----------------------
# Detailed Analysis & Controls (below top)
# ----------------------
left, right = st.columns([2,1])

with left:
    st.header("Buyer Table & Predictions")
    st.markdown("Search, filter, and select a buyer to view recommended action and execute it.")

    # Filters
    f_income = st.multiselect("Filter income bracket", options=st.session_state['buyers_df']['income_bracket'].unique(), default=None)
    f_city = st.multiselect("Filter city tier", options=sorted(st.session_state['buyers_df']['city_tier'].unique()), default=None)
    if f_income:
        df_view = st.session_state['buyers_df'][st.session_state['buyers_df']['income_bracket'].isin(f_income)]
    else:
        df_view = st.session_state['buyers_df']
    if f_city:
        df_view = df_view[df_view['city_tier'].isin(f_city)]

    # show table
    st.dataframe(df_view[['buyer_id','email','age','income_bracket','city_tier','engagement','freq_12m','avg_order_value','recency_days','pred_action','pred_prob']], height=350)

    st.markdown("**Select buyer to see details & run action**")
    sel = st.selectbox("Select buyer", options=df_view.index, format_func=lambda i: f"{st.session_state['buyers_df'].loc[i,'buyer_id']} — {st.session_state['buyers_df'].loc[i,'email']}")
    buyer = st.session_state['buyers_df'].loc[sel]

    st.subheader("Buyer Details")
    st.json(buyer.to_dict())

    st.subheader("Recommended Action")
    st.info(f"Action: {buyer['pred_action']} (Confidence: {buyer['pred_prob']})")

    # Action buttons
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Execute Action for Selected Buyer"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state['action_log'].append({
                'timestamp': timestamp,
                'buyer_id': buyer['buyer_id'],
                'email': buyer['email'],
                'action': buyer['pred_action'],
                'confidence': buyer['pred_prob']
            })
            st.success(f"Executed action '{buyer['pred_action']}' for {buyer['buyer_id']}")
    with col_b:
        if st.button("Simulate Buyer Conversion"):
            # mark as converted in df
            st.session_state['buyers_df'].at[sel,'converted'] = True
            st.success("Simulated conversion recorded")
    with col_c:
        if st.button("Add to Retargeting Audience"):
            st.session_state['action_log'].append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'buyer_id': buyer['buyer_id'],
                'email': buyer['email'],
                'action': 'RetargetingAudience',
                'confidence': buyer['pred_prob']
            })
            st.success("Added to retargeting audience")

with right:
    st.header("Analytics & Charts")
    st.subheader("Action Distribution (Predicted)")
    action_counts = st.session_state['buyers_df']['pred_action'].value_counts()
    st.bar_chart(action_counts)

    st.subheader("Income vs Action")
    pivot = pd.crosstab(st.session_state['buyers_df']['income_bracket'], st.session_state['buyers_df']['pred_action'])
    st.dataframe(pivot)

    st.subheader("Top Feature Insights")
    # crude feature importance using permutation-like importance from tree
    try:
        importances = model.feature_importances_
        imp = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
        st.table(imp.head(10))
    except Exception as e:
        st.write("Feature importance not available")

# ----------------------
# Execution Log & KPIs (bottom)
# ----------------------
st.markdown("---")
st.header("Execution Log & KPIs")

log_df = pd.DataFrame(st.session_state['action_log'])
if not log_df.empty:
    st.subheader("Recent Actions")
    st.dataframe(log_df.sort_values('timestamp', ascending=False).head(50))

    executed = log_df['action'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Actions Executed", len(log_df))
    col2.metric("Unique Buyers Targeted", log_df['buyer_id'].nunique())
    # conversions simulated
    conv_count = int(st.session_state['buyers_df'].get('converted', False).sum()) if 'converted' in st.session_state['buyers_df'].columns else 0
    col3.metric("Simulated Conversions", conv_count)
else:
    st.write("No actions executed yet. Use the buttons above to run actions for buyers.")

st.markdown("---")
st.caption("Prototype: Synthetic data, rule-derived labels and RandomForest model. For real deployment, replace with real CDP data, continuous model training, consent management, and channel connectors.")
