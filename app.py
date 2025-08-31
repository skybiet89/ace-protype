import streamlit as st
import pandas as pd
import random
from datetime import datetime

# ----------------------
# INITIAL SETUP
# ----------------------
st.set_page_config(page_title="ACE Platform Prototype", layout="wide")

# Simulated Customer Database (in-memory)
if "customers" not in st.session_state:
    st.session_state.customers = pd.DataFrame(columns=[
        "user_id", "name", "email", "last_event", "cart_items",
        "recency_days", "frequency", "monetary", "purchase_score",
        "decision", "last_message", "converted"
    ])

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def simulate_event(event_type, name, email, product, price):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = email.split("@")[0]

    # If customer exists, update profile
    df = st.session_state.customers
    if email in df["email"].values:
        idx = df[df["email"] == email].index[0]
        df.loc[idx, "last_event"] = event_type
        if event_type == "purchase":
            df.loc[idx, "frequency"] += 1
            df.loc[idx, "monetary"] = (df.loc[idx, "monetary"] + price) / 2
            df.loc[idx, "recency_days"] = 0
            df.loc[idx, "converted"] = "Yes"
        elif event_type == "cart_abandon":
            df.loc[idx, "cart_items"] = product
            df.loc[idx, "recency_days"] += 1
    else:
        new_row = pd.DataFrame({
            "user_id": [user_id],
            "name": [name],
            "email": [email],
            "last_event": [event_type],
            "cart_items": [product if event_type == "cart_abandon" else None],
            "recency_days": [1],
            "frequency": [1 if event_type == "purchase" else 0],
            "monetary": [price if event_type == "purchase" else 0],
            "purchase_score": [0],
            "decision": ["Pending"],
            "last_message": ["None"],
            "converted": ["No"]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    st.session_state.customers = df


def compute_score():
    df = st.session_state.customers
    for idx, row in df.iterrows():
        recency_score = max(0, 1 - (row["recency_days"] / 30))
        freq_score = min(1, row["frequency"] / 10)
        monetary_score = min(1, row["monetary"] / 100)
        score = round((0.5 * recency_score + 0.3 * freq_score + 0.2 * monetary_score) * 100, 2)
        df.loc[idx, "purchase_score"] = score
    st.session_state.customers = df


def decide_action():
    df = st.session_state.customers
    for idx, row in df.iterrows():
        if row["purchase_score"] >= 70:
            df.loc[idx, "decision"] = "Send Email Reminder"
            df.loc[idx, "last_message"] = f"Email: Complete your purchase of {row['cart_items']}!"
        elif 40 <= row["purchase_score"] < 70:
            df.loc[idx, "decision"] = "Trigger Chat Message"
            df.loc[idx, "last_message"] = f"Chat: Hey {row['name']}, still thinking about {row['cart_items']}?"
        else:
            df.loc[idx, "decision"] = "Do Nothing"
            df.loc[idx, "last_message"] = "-"
    st.session_state.customers = df

# ----------------------
# UI LAYOUT
# ----------------------
st.title("✨ ACE Platform Prototype ✨")
st.markdown("This demo shows how customer events flow through the ACE platform layers.")

col1, col2, col3 = st.columns([1,2,2])

# ----------------------
# LEFT COLUMN: Event Simulation
# ----------------------
with col1:
    st.header("1️⃣ Simulate Event")
    name = st.text_input("Customer Name", "Alice")
    email = st.text_input("Email", "alice@example.com")
    product = st.text_input("Product", "Running Shoes")
    price = st.number_input("Price", min_value=10, max_value=500, value=60)

    if st.button("🛒 Add to Cart (Abandon)"):
        simulate_event("cart_abandon", name, email, product, price)
        st.success("Cart Abandon Event Captured!")

    if st.button("✅ Purchase"):
        simulate_event("purchase", name, email, product, price)
        st.success("Purchase Event Captured!")

    if st.button("📊 Run Processing + Decision"):
        compute_score()
        decide_action()
        st.info("Profile updated with score and decision!")

# ----------------------
# MIDDLE COLUMN: Internal Platform View
# ----------------------
with col2:
    st.header("2️⃣ Platform Processing")
    st.markdown("### Customer Profiles")
    st.dataframe(st.session_state.customers, use_container_width=True)

# ----------------------
# RIGHT COLUMN: Customer View (Messages)
# ----------------------
with col3:
    st.header("3️⃣ Customer Experience")
    for idx, row in st.session_state.customers.iterrows():
        with st.container():
            st.subheader(f"👤 {row['name']} ({row['email']})")
            st.write(f"**Decision:** {row['decision']}")
            st.info(row["last_message"])
            if row["converted"] == "Yes":
                st.success("✅ Customer Converted!")

# ----------------------
# MONITORING SECTION
# ----------------------
st.markdown("---")
st.header("📈 Monitoring & KPIs")

df = st.session_state.customers
if not df.empty:
    total_customers = len(df)
    converted = len(df[df["converted"] == "Yes"])
    email_sends = len(df[df["decision"] == "Send Email Reminder"])
    chat_sends = len(df[df["decision"] == "Trigger Chat Message"])

    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Total Customers", total_customers)
    colm2.metric("Conversions", converted)
    colm3.metric("Emails Sent", email_sends)
    colm4.metric("Chats Triggered", chat_sends)

    st.line_chart(df[["purchase_score"]])
else:
    st.write("No customer data yet. Simulate events to see monitoring.")
