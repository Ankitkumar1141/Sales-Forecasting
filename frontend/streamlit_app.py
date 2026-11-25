import streamlit as st
import requests
import pandas as pd
import altair as alt

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

PRODUCT_FAMILIES = sorted([
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "EGGS",
    "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
    "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR/WINE/BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
    "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
    "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES",
    "SEAFOOD", "SEASONAL",
])

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.4rem; font-weight: 700; color: #1a73e8; }
    .sub-title   { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card { background: #f0f4ff; border-radius: 10px;
                   padding: 1rem; text-align: center; }
    .status-ok   { color: #0a9e4f; font-weight: 600; }
    .status-err  { color: #c62828; font-weight: 600; }
    .section-hdr { font-size: 1.1rem; font-weight: 600;
                   color: #1a1a1a; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper: check API health
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else {"status": "error", "detail": r.text}
    except Exception as exc:
        return {"status": "unreachable", "detail": str(exc)}


@st.cache_data(ttl=120)
def get_model_info() -> dict:
    try:
        r = requests.get(f"{API_BASE}/info", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shop.png", width=70)
    st.markdown("## ⚙️ Forecast Parameters")

    store_nbr = st.slider(
        "🏪 Store Number", min_value=1, max_value=54, value=1, step=1
    )
    family = st.selectbox("📦 Product Family", PRODUCT_FAMILIES, index=0)
    horizon = st.slider(
        "📅 Forecast Horizon (days)", min_value=1, max_value=30, value=7, step=1
    )

    st.markdown("---")
    st.markdown("### 📋 Day-level Covariates")
    st.caption(
        "Set `On Promotion` and `Is Holiday` for each forecast day. "
        "0 = No, 1 = Yes."
    )

    covariate_df = pd.DataFrame(
        {
            "Day": [f"Day {i+1}" for i in range(horizon)],
            "On Promotion": [0] * horizon,
            "Is Holiday": [0] * horizon,
        }
    )

    edited_df = st.data_editor(
        covariate_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Day": st.column_config.TextColumn("Day", disabled=True),
            "On Promotion": st.column_config.SelectboxColumn(
                "On Promotion", options=[0, 1], required=True
            ),
            "Is Holiday": st.column_config.SelectboxColumn(
                "Is Holiday", options=[0, 1], required=True
            ),
        },
        key="covariate_editor",
    )

    predict_btn = st.button(
        "🚀 Generate Forecast", type="primary", use_container_width=True
    )

# ─────────────────────────────────────────────
# Main area — header + status
# ─────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🛒 Sales Forecasting Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Powered by LightGBM + Darts · FastAPI backend</p>',
    unsafe_allow_html=True,
)

# Health status banner
health = check_health()
col_h1, col_h2, col_h3 = st.columns([2, 2, 4])
with col_h1:
    if health.get("status") in ("ok", "healthy"):
        st.success(f"✅ API Online — {API_BASE}")
    else:
        st.error(f"❌ API Unreachable — {health.get('detail', '')}")
        st.info("▶ Start the backend: `uvicorn app:app --reload`")

with col_h2:
    info = get_model_info()
    if info:
        st.info(f"🤖 Model: {info.get('model', 'LightGBM')}")

st.markdown("---")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if predict_btn:
    onpromotion = edited_df["On Promotion"].astype(int).tolist()
    is_holiday  = edited_df["Is Holiday"].astype(int).tolist()

    payload = {
        "store_nbr":   store_nbr,
        "family":      family,
        "horizon":     horizon,
        "onpromotion": onpromotion,
        "is_holiday":  is_holiday,
    }

    with st.spinner("⏳ Calling forecast API …"):
        try:
            response = requests.post(
                f"{API_BASE}/predict", json=payload, timeout=30
            )
            if response.ok:
                result = response.json()
                forecasts = result.get("forecasts", [])

                # ── Summary metrics ──────────────────────────────
                st.subheader("📊 Forecast Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Store", f"#{result['store_nbr']}")
                m2.metric("Family", result["family"])
                m3.metric("Days Ahead", result["horizon"])
                m4.metric("Peak Forecast", f"{max(forecasts):.2f}")

                # ── Forecast table ───────────────────────────────
                forecast_df = pd.DataFrame(
                    {
                        "Day":      [f"Day {i+1}" for i in range(len(forecasts))],
                        "Date":     pd.date_range(
                            start=pd.Timestamp.today().normalize(), periods=len(forecasts)
                        ).strftime("%Y-%m-%d"),
                        "Forecast (Sales)": [round(v, 4) for v in forecasts],
                        "On Promotion":     onpromotion,
                        "Is Holiday":       is_holiday,
                    }
                )

                col_chart, col_table = st.columns([3, 2])

                with col_chart:
                    st.markdown("#### 📈 Sales Forecast Chart")
                    chart = (
                        alt.Chart(forecast_df)
                        .mark_line(point=True, color="#1a73e8", strokeWidth=2.5)
                        .encode(
                            x=alt.X("Day:O", title="Forecast Day"),
                            y=alt.Y(
                                "Forecast (Sales):Q",
                                title="Predicted Sales",
                                scale=alt.Scale(zero=False),
                            ),
                            tooltip=["Day", "Date", "Forecast (Sales)",
                                     "On Promotion", "Is Holiday"],
                        )
                        .properties(height=320)
                        .interactive()
                    )
                    # Overlay promotion / holiday markers
                    promo_df = forecast_df[forecast_df["On Promotion"] == 1]
                    if not promo_df.empty:
                        promo_layer = (
                            alt.Chart(promo_df)
                            .mark_point(shape="triangle", color="#f4a300",
                                        size=120, filled=True)
                            .encode(
                                x="Day:O",
                                y="Forecast (Sales):Q",
                                tooltip=["Day", "Date", "Forecast (Sales)"],
                            )
                        )
                        chart = chart + promo_layer

                    holiday_df = forecast_df[forecast_df["Is Holiday"] == 1]
                    if not holiday_df.empty:
                        holiday_layer = (
                            alt.Chart(holiday_df)
                            .mark_point(shape="diamond", color="#e53935",
                                        size=120, filled=True)
                            .encode(
                                x="Day:O",
                                y="Forecast (Sales):Q",
                                tooltip=["Day", "Date", "Forecast (Sales)"],
                            )
                        )
                        chart = chart + holiday_layer

                    st.altair_chart(chart, use_container_width=True)

                    if promo_df is not None and not promo_df.empty:
                        st.caption("🔶 Triangle = Promotion day  ·  🔴 Diamond = Holiday")

                with col_table:
                    st.markdown("#### 🗂️ Forecast Table")
                    st.dataframe(
                        forecast_df[["Day", "Date", "Forecast (Sales)",
                                     "On Promotion", "Is Holiday"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    # Download button
                    csv_data = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_data,
                        file_name=f"forecast_store{store_nbr}_{family.replace('/', '-')}_h{horizon}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                # ── Summary stats ────────────────────────────────
                st.markdown("#### 📉 Forecast Statistics")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Mean",    f"{pd.Series(forecasts).mean():.4f}")
                s2.metric("Std Dev", f"{pd.Series(forecasts).std():.4f}")
                s3.metric("Min",     f"{min(forecasts):.4f}")
                s4.metric("Max",     f"{max(forecasts):.4f}")

            else:
                # API returned an error
                try:
                    err = response.json()
                    detail = err.get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"❌ API Error {response.status_code}: {detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the FastAPI backend.\n\n"
                "**Start it first:** `uvicorn app:app --reload`"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The model may still be loading — try again.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

else:
    # Placeholder when no forecast has been run yet
    st.info(
        "👈 Set your parameters in the **sidebar**, then click **Generate Forecast**."
    )
    with st.expander("ℹ️ How it works", expanded=True):
        st.markdown("""
| Step | What happens |
|---|---|
| 1️⃣ Select store & product | Choose store number (1–54) and one of 33 product families |
| 2️⃣ Set horizon | Pick how many days ahead to forecast (1–30) |
| 3️⃣ Set covariates | Mark each day as On Promotion or Is Holiday |
| 4️⃣ Generate | Frontend POSTs to FastAPI → LightGBM model → returns predictions |
| 5️⃣ Visualise | Interactive chart + downloadable CSV |

**Backend endpoint:** `POST http://127.0.0.1:8000/predict`
""")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🛒 Sales Forecasting · LightGBM + Darts time-series pipeline · "
    "FastAPI backend · Streamlit frontend"
)
