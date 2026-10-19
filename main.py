import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Streamlit Power Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Streamlit Power Dashboard")
st.markdown(
    "A tour of what Streamlit can do — native charts, Plotly, Matplotlib/Seaborn, "
    "Altair, widgets, layouts, metrics and more."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
n_points = st.sidebar.slider("Number of data points", 50, 1000, 300, step=50)
noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.2, step=0.05)
chart_theme = st.sidebar.selectbox("Color theme", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])
show_raw = st.sidebar.checkbox("Show raw data table", value=False)

rng = np.random.default_rng(42)
x = np.linspace(0, 10, n_points)
trend = np.sin(x) + 0.3 * x
noise = rng.normal(0, noise_level, size=n_points)
y = trend + noise

df = pd.DataFrame(
    {
        "x": x,
        "y": y,
        "trend": trend,
        "category": rng.choice(["A", "B", "C", "D"], size=n_points),
        "value": rng.uniform(0, 100, size=n_points),
    }
)

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
st.subheader("🔑 Quick Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Points", f"{n_points:,}", delta=f"+{n_points - 50}")
m2.metric("Mean Y", f"{y.mean():.3f}", delta=f"{y.mean():.3f}")
m3.metric("Std Y", f"{y.std():.3f}")
m4.metric("Max Y", f"{y.max():.3f}", delta=f"{y.max() - y.min():.3f}", delta_color="off")

st.divider()

# ---------------------------------------------------------------------------
# Native Streamlit charts
# ---------------------------------------------------------------------------
st.subheader("📈 Native Streamlit Charts")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Line chart**")
    st.line_chart(df.set_index("x")[["y", "trend"]], height=300)
with c2:
    st.markdown("**Area chart**")
    st.area_chart(df.set_index("x")[["y"]], height=300)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Bar chart**")
    bar_df = df.groupby("category")["value"].mean().reset_index()
    st.bar_chart(bar_df.set_index("category"), height=300)
with c4:
    st.markdown("**Scatter chart**")
    st.scatter_chart(df[["x", "y"]], height=300)

st.divider()

# ---------------------------------------------------------------------------
# Plotly interactive charts
# ---------------------------------------------------------------------------
st.subheader("🌀 Plotly Interactive Charts")
p1, p2 = st.columns(2)
with p1:
    fig_line = px.line(
        df, x="x", y=["y", "trend"], title="Sine + Linear Trend with Noise",
        color_discrete_sequence=px.colors.sequential.Viridis[:2][::-1],
    )
    st.plotly_chart(fig_line, use_container_width=True)
with p2:
    fig_scatter = px.scatter(
        df, x="x", y="y", color="category", size="value",
        title="Scatter by Category (size = value)", opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

p3, p4 = st.columns(2)
with p3:
    hist_df = pd.DataFrame({"y": y})
    fig_hist = px.histogram(hist_df, x="y", nbins=40, title="Distribution of Y",
                            color_discrete_sequence=[px.colors.sequential.Plasma[4]])
    st.plotly_chart(fig_hist, use_container_width=True)
with p4:
    box_df = df[["category", "value"]]
    fig_box = px.box(box_df, x="category", y="value", title="Value by Category",
                     color="category", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_box, use_container_width=True)

# 3D surface
st.markdown("**3D Surface (Plotly)**")
u = np.linspace(-2, 2, 40)
v = np.linspace(-2, 2, 40)
U, V = np.meshgrid(u, v)
Z = np.sin(U) * np.cos(V) * np.exp(-(U**2 + V**2) / 4)
fig_surf = go.Figure(data=[go.Surface(z=Z, x=u, y=v, colorscale=chart_theme.lower())])
fig_surf.update_layout(title="3D Surface: sin(u)·cos(v)·exp(-(u²+v²)/4)",
                        scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="z"),
                        margin=dict(l=0, r=0, b=0, t=40))
st.plotly_chart(fig_surf, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Matplotlib / Seaborn static charts
# ---------------------------------------------------------------------------
st.subheader("🎨 Matplotlib & Seaborn Static Charts")
s1, s2 = st.columns(2)
with s1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, label="noisy", alpha=0.6)
    ax.plot(x, trend, label="trend", linewidth=2)
    ax.set_title("Matplotlib: Trend vs Noisy")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
    st.pyplot(fig)
    plt.close(fig)
with s2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(y, bins=40, kde=True, ax=ax2, color=sns.color_palette("flare")[3])
    ax2.set_title("Seaborn: Histogram + KDE")
    st.pyplot(fig2)
    plt.close(fig2)

s3, s4 = st.columns(2)
with s3:
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="category", y="value", ax=ax3, palette="Set2")
    ax3.set_title("Seaborn: Boxplot by Category")
    st.pyplot(fig3)
    plt.close(fig3)
with s4:
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    corr = df[["x", "y", "trend", "value"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax4)
    ax4.set_title("Seaborn: Correlation Heatmap")
    st.pyplot(fig4)
    plt.close(fig4)

st.divider()

# ---------------------------------------------------------------------------
# Altair charts
# ---------------------------------------------------------------------------
st.subheader("🔷 Altair Charts")
a1, a2 = st.columns(2)
with a1:
    line_alt = (
        alt.Chart(df).mark_line().encode(
            x="x:Q", y="trend:Q", color=alt.value("#2a9d8f")
        ).properties(title="Altair: Trend Line", height=300)
    )
    st.altair_chart(line_alt, use_container_width=True)
with a2:
    scatter_alt = (
        alt.Chart(df).mark_circle(opacity=0.6).encode(
            x="x:Q", y="y:Q", color="category:N", size="value:Q"
        ).properties(title="Altair: Scatter", height=300)
    )
    st.altair_chart(scatter_alt, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Tabs + progress + dataframe
# ---------------------------------------------------------------------------
st.subheader("🗂️ Tabs, Progress & Data")
tab1, tab2, tab3 = st.tabs(["Summary", "Raw Data", "Animation"])

with tab1:
    st.dataframe(df.describe(), use_container_width=True)

with tab2:
    if show_raw:
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("Enable **Show raw data table** in the sidebar to view rows.")

with tab3:
    st.markdown("**Animated bar race (Plotly)**")
    days = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    anim_df = pd.DataFrame({
        "day": days * 4,
        "category": sum(([c] * 30 for c in ["A", "B", "C", "D"]), []),
        "value": rng.uniform(10, 100, size=120),
    })
    fig_anim = px.bar(
        anim_df, x="category", y="value", color="category",
        animation_frame="day", range_y=[0, 110],
        title="Daily Value by Category", color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_anim, use_container_width=True)

st.divider()
st.caption(f"Generated at {datetime.now():%Y-%m-%d %H:%M:%S} · Streamlit + Plotly + Seaborn + Altair")
