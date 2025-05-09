import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="CrewNPS Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dummy_crew_nps_data.csv", parse_dates=["nps_valid_from", "sentiment_valid_from"])
    df['engage_sentiment'] = df['engage_sentiment'].str.lower().str.strip()
    return df

df = load_data()
st.title("✈️ CrewNPS Intelligence Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filters")
ranks = df['rank'].dropna().unique().tolist()
managers = df['manager'].dropna().unique().tolist()
flight_types = df['flight_type'].dropna().unique().tolist()

selected_rank = st.sidebar.multiselect("Select Rank", ranks, default=ranks)
selected_manager = st.sidebar.multiselect("Select Manager", managers, default=managers)
selected_flight = st.sidebar.multiselect("Flight Type", flight_types, default=flight_types)

date_range = st.sidebar.slider(
    "NPS Date Range",
    min_value=df['nps_valid_from'].min().date(),
    max_value=df['nps_valid_from'].max().date(),
    value=(df['nps_valid_from'].min().date(), df['nps_valid_from'].max().date())
)

# Filter data
filtered_df = df[
    (df['rank'].isin(selected_rank)) &
    (df['manager'].isin(selected_manager)) &
    (df['flight_type'].isin(selected_flight)) &
    (df['nps_valid_from'].dt.date.between(*date_range))
]

# Show filtered data
st.markdown(f"### 🎯 {len(filtered_df):,} Records Selected")
st.dataframe(filtered_df.sample(10), use_container_width=True)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("📈 Avg CrewNPS", f"{filtered_df['crewnps'].mean():.2f}")
col2.metric("😊 % Favorable Engagement", f"{(filtered_df['engage_sentiment'] == 'favorable').mean() * 100:.1f}%")
col3.metric("🧍 Avg Tenure", f"{filtered_df['tenure_years'].mean():.1f} yrs")

# Relationship plots
st.subheader("📊 Feature Relationships")
tabs = st.tabs(["Engagement Score", "Absences", "Tenure"])

with tabs[0]:
    fig = px.scatter(filtered_df, x="engage_sentiment_score", y="crewnps", color="rank", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    fig = px.scatter(filtered_df, x="absence_days_past_6_months", y="crewnps", color="performance_flag", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    fig = px.scatter(filtered_df, x="tenure_years", y="crewnps", color="flight_type", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

# Driver Importance
st.subheader("🔬 Feature Importance (Random Forest)")

# Preprocess for model
model_df = filtered_df.dropna(subset=['crewnps', 'engage_sentiment_score', 'absence_days_past_6_months', 'tenure_years'])
features = ['engage_sentiment_score', 'absence_days_past_6_months', 'tenure_years']
X = model_df[features]
y = model_df['crewnps']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=True)

fig = px.bar(importances, x="Importance", y="Feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)

# PDPs
st.subheader("📈 Partial Dependence Plots (Feature Impact on NPS)")

fig, ax = plt.subplots(figsize=(10, 4))
display = PartialDependenceDisplay.from_estimator(rf, X, features, ax=ax)
st.pyplot(fig)

# Summary
st.subheader("🧠 Key Insight Summary")
high_nps = filtered_df[filtered_df['crewnps'] > 80]
low_engagement = filtered_df[filtered_df['engage_sentiment'] == 'unfavorable']

st.markdown(f"""
- ✅ **{len(high_nps)} employees** scored above 80 in CrewNPS — potential role models.
- ⚠️ **{len(low_engagement)} employees** marked as 'unfavorable' in engagement — flag for attention.
- 📌 Top influencing feature (RF): **{importances.iloc[-1]['Feature']}**
""")

st.success("✅ Dashboard Ready — Client-Grade Insights Delivered.")