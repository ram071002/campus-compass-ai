import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Campus Compass AI+", page_icon="🏡", layout="wide")

# ---------- CSS for gradient theme + glow cards ----------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 10% 20%, #0f172a 0%, #1e1b4b 90%);
    color: #E2E8F0;
    font-family: 'Poppins', sans-serif;
}
.header {
    text-align: center;
    padding: 50px 20px 40px;
    border-radius: 25px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    color: white;
    box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    margin-bottom: 40px;
}
.card {
    position: relative;
    background: rgba(30,41,59,0.85);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    color: #E2E8F0;
    padding: 25px;
    margin-bottom: 15px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.card:hover {transform: scale(1.05); box-shadow: 0 0 25px #60A5FA;}
.card .tooltip {
    visibility: hidden;
    position: absolute;
    top: -50px;
    left: 10px;
    background: rgba(17,24,39,0.9);
    color: #E0E0E0;
    padding: 10px 12px;
    border-radius: 10px;
    font-size: 14px;
    white-space: nowrap;
}
.card:hover .tooltip {visibility: visible;}
.user-bubble {
    background: #3B82F6;
    color: white;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    text-align: right;
}
.ai-bubble {
    background: #6D28D9;
    color: white;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    text-align: left;
}
.send-btn {
    background-color: #3B82F6;
    color: white;
    padding: 6px 14px;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    cursor: pointer;
}
.send-btn:hover {background-color: #2563EB;}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="header">
    <h1>🏡 Campus Compass AI+</h1>
    <p>Your Intelligent Roommate & Housing Recommendation Assistant</p>
</div>
""", unsafe_allow_html=True)

# ---------- Input Section ----------
st.markdown("### 🎯 Personalize Your Lifestyle Profile")

col1, col2, col3 = st.columns(3)
with col1:
    cleanliness = st.slider("🧼 Cleanliness", 1, 5, 3)
    noise = st.slider("🔉 Noise Tolerance", 1, 5, 3)
with col2:
    sleep = st.slider("🌙 Sleep Schedule (1=Early, 5=Late)", 1, 5, 3)
    food = st.selectbox("🍴 Food Preference", ["Veg", "Non-Veg"])
with col3:
    budget = st.number_input("💰 Monthly Rent Budget ($)", 600, 1500, 900)

# ---------- Roommate + Housing Data ----------
roommates = pd.DataFrame([
    {"Name": "Aarav", "Cleanliness": 4, "NoiseTolerance": 2, "SleepSchedule": 3, "FoodPreference": "Veg", "Budget": 900, "Note": "Prefers clean shared spaces & early mornings."},
    {"Name": "Neel", "Cleanliness": 5, "NoiseTolerance": 1, "SleepSchedule": 2, "FoodPreference": "Veg", "Budget": 1000, "Note": "Quiet roommate with similar routines."},
    {"Name": "Dheeraj", "Cleanliness": 3, "NoiseTolerance": 4, "SleepSchedule": 5, "FoodPreference": "Non-Veg", "Budget": 800, "Note": "Night owl and music lover."},
    {"Name": "Manohar", "Cleanliness": 2, "NoiseTolerance": 3, "SleepSchedule": 4, "FoodPreference": "Non-Veg", "Budget": 750, "Note": "Relaxed and friendly personality."},
    {"Name": "Gayu", "Cleanliness": 5, "NoiseTolerance": 1, "SleepSchedule": 2, "FoodPreference": "Veg", "Budget": 950, "Note": "Organized and enjoys cooking healthy meals."},
])
le = LabelEncoder()
roommates["FoodPreference"] = le.fit_transform(roommates["FoodPreference"])
user_food = le.transform([food])[0]

weights = np.array([1.2, 1.0, 1.0, 0.8, 1.0])
user_vec = np.array([[cleanliness, noise, sleep, user_food, budget]])
X = roommates[["Cleanliness", "NoiseTolerance", "SleepSchedule", "FoodPreference", "Budget"]].to_numpy()
X_weighted = X * weights
user_vec_weighted = user_vec * weights
similarities = cosine_similarity(user_vec_weighted, X_weighted)[0]
roommates["SimilarityScore"] = similarities
matches = roommates.sort_values("SimilarityScore", ascending=False).head(3)

housing = pd.DataFrame([
    {"Name": "Parkside Apartments", "Rent": 950, "Type": "Shared", "Distance (mi)": 1.0, "Note": "Affordable shared units, 10 mins walk to campus."},
    {"Name": "Fairlane Meadows", "Rent": 1100, "Type": "Private", "Distance (mi)": 0.8, "Note": "Upscale apartments near Fairlane Mall."},
    {"Name": "Village Green", "Rent": 800, "Type": "Shared", "Distance (mi)": 1.2, "Note": "Budget friendly and pet-friendly housing."},
    {"Name": "Union at Dearborn", "Rent": 1200, "Type": "Private", "Distance (mi)": 0.5, "Note": "Luxury private rooms near university center."},
    {"Name": "Dearborn View", "Rent": 1000, "Type": "Shared", "Distance (mi)": 0.9, "Note": "Quiet neighborhood with community spaces."},
])
housing["MatchScore"] = 1 - abs(housing["Rent"] - budget) / budget
housing["CompositeScore"] = 0.7 * housing["MatchScore"] + 0.3 * (1 / housing["Distance (mi)"])
recs = housing.sort_values("CompositeScore", ascending=False).head(3)

# ---------- Generate Button ----------
generate = st.button("✨ Generate AI Recommendations")
if generate:
    st.markdown("### 👥 Top Roommate Matches")
    for _, row in matches.iterrows():
        st.markdown(f"""
        <div class="card">
            <div class="tooltip">{row['Note']}</div>
            <h3>👤 {row['Name']}</h3>
            <p><b>Compatibility Score:</b> {row['SimilarityScore']:.2f}</p>
            <p>💡 Shares similar lifestyle preferences and budget.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏠 Housing Recommendations")
    for _, row in recs.iterrows():
        st.markdown(f"""
        <div class="card">
            <div class="tooltip">{row['Note']}</div>
            <h3>🏢 {row['Name']}</h3>
            <p><b>Rent:</b> ${row['Rent']} | <b>Type:</b> {row['Type']}</p>
            <p>📍 Distance: {row['Distance (mi)']} mi | <b>Score:</b> {row['CompositeScore']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- CHAT: Compass AI Assistant ----------------
st.markdown("### 🤖 Compass AI Assistant")

# 1) Keep chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2) Helper that processes a message and refreshes the UI
def handle_message():
    text = st.session_state.get("chat_input", "").strip()
    if not text:
        return

    # Append user message
    st.session_state.chat_history.append(("🧍 You", text))
    lower = text.lower()

    # --- Generate reply (uses your existing matches/recs computed above) ---
    if "hi" in lower or "hello" in lower:
        reply = "Hi 👋! I’m Compass AI — ready to show your roommate or housing matches."
    elif "roommate" in lower:
        best = matches.iloc[0]
        reply = f"Your top roommate match is **{best['Name']}** with a compatibility score of {best['SimilarityScore']:.2f}. 🎯"
    elif "housing" in lower or "apartment" in lower:
        home = recs.iloc[0]
        reply = f"I recommend **{home['Name']}** — ${home['Rent']} rent, {home['Type']} type, {home['Distance (mi)']} mi away. 🏢"
    elif "recommendation" in lower or "show" in lower:
        reply = "Tap ✨ *Generate AI Recommendations* above to see your personalized matches."
    else:
        reply = "I can help with roommates and housing. Try: ‘Who’s my best roommate?’ or ‘Show housing near campus.’ 💡"

    # Append AI reply, clear input, and rerun cleanly
    st.session_state.chat_history.append(("🤖 Compass AI", reply))
    st.session_state.chat_input = ""     # clears the text box after send
    st.rerun()                           # Streamlit ≥ 1.30 (no experimental_ call)

# 3) Render past messages
for speaker, msg in st.session_state.chat_history:
    bubble = "user-bubble" if speaker == "🧍 You" else "ai-bubble"
    st.markdown(f"<div class='{bubble}'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)

# 4) Input + Send (Enter triggers on_change; button triggers on_click)
c1, c2 = st.columns([6, 1])
with c1:
    st.text_input("💬 Message Compass AI...", key="chat_input",
                  placeholder="Type: hi, who's my roommate?, show housing",
                  on_change=handle_message)
with c2:
    st.button("Send", on_click=handle_message)



st.caption("✨ Designed by Ramkumaar E.T | Campus Compass Project | Intelligent Student Experience 🌙")
