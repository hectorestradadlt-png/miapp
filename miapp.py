# miapp_enhanced.py
# VERSION LIMPIA + TRADUCCIÓN NOTICIAS A ESPAÑOL + FIX GEMINI

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta
import requests

# -------------------------------- CONFIG --------------------------------
st.set_page_config(page_title="PA' EL TASTYTRADE", layout="wide", page_icon="🤑")

st.markdown("""
<style>
body { background-color: #7DBCFF; }
.main { background-color: #7DBCFF; }
.stSidebar { background-color: #DB4E35; }
h1 { color: #F8C537; text-align: center; font-size: 55px; font-weight: bold; text-shadow: 2px 2px 5px #FF5F1F; }
h2, h3 { color: #FF8C00; }
</style>
""", unsafe_allow_html=True)

# -------------------------------- HEADER --------------------------------
IMG_PATH = "Tom.jpg"
col_title, col_logo = st.columns([3,2])
with col_title:
    st.markdown("<h1 style='text-align:left;'>💥 PA' EL TASTYTRADE 💥</h1>", unsafe_allow_html=True)
with col_logo:
    if os.path.exists(IMG_PATH): st.image(IMG_PATH, width=520)

st.markdown("---")

# -------------------------------- API KEYS --------------------------------
GENAI_API_KEY = "AIzaSyBvUKkfxwE5zcvF4iPb6qHDnAkR8uwB8y0"  # Pega tu key
NEWS_API_KEY = "8af8a9f4caff44379a9b0794ea717e4f"   # Pega tu key

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# -------------------------------- SIDEBAR --------------------------------
st.sidebar.header("Parámetros")
stonk = st.sidebar.text_input("Símbolo", "AAPL")
years = st.sidebar.selectbox("Periodo (años)", [1,3,5,7,10], index=2)
end_date = datetime.today()
start_date = end_date - timedelta(days=years*365)

multi_symbols = st.sidebar.text_area("Comparativa (coma, max 5):", "MSFT, GOOGL, AMZN").strip()
multi_symbols = [s.strip().upper() for s in multi_symbols.split(",") if s.strip()][:5]

mc_sim = st.sidebar.number_input("Simulaciones Monte Carlo", 100, 5000, 1000, 100)
mc_horizon_days = st.sidebar.number_input("Horizonte (días)", 30, 252*5, 252, 30)

# -------------------------------- DOWNLOAD DATA --------------------------------
data = yf.download(stonk, start=start_date, end=end_date)
spy = yf.download("SPY", start=start_date, end=end_date)

if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)

if data.empty:
    st.error("No hay datos del símbolo.")
    st.stop()

# -------------------------------- DESCRIPCION --------------------------------
ticker_obj = yf.Ticker(stonk)
info = ticker_obj.info if hasattr(ticker_obj, "info") else {}
nombre = info.get("longName", stonk)
descripcion = info.get("longBusinessSummary", "Sin descripción disponible.")

st.subheader(f"📘 {nombre}")
st.write(f"**Descripción (Inglés):** {descripcion[:800]}...")

# Gemini translate

def translate_with_gemini(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        resp = model.generate_content(f"Traduce al español financiero formal y resume a 100 palabras: {text}")
        return resp.text
    except:
        return None

trand = None
if GENAI_API_KEY:
    trand = translate_with_gemini(descripcion)

if not trand:
    try:
        trand = GoogleTranslator(source="en", target="es").translate(descripcion)
    except:
        trand = "No fue posible traducir."

st.write("**Descripción (Español):**", trand)

# -------------------------------- TABS --------------------------------
tabs = st.tabs([
    "📈 Histórico",
    "📉 Técnicos",
    "📊 SPY",
    "🔁 Monte Carlo",
    "🧪 Backtest EMA",
    "📈 Comparativa",
    "📰 Noticias",
    "🤖 Insights IA"
])

# -------------------------------- TAB 1 HIST --------------------------------
with tabs[0]:
    st.subheader("Histórico Close & Open")
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(x=data.index, y=data["Close"], ax=ax, label="Close")
    sns.lineplot(x=data.index, y=data["Open"], ax=ax, label="Open")
    st.pyplot(fig)
    st.dataframe(data.tail(20))

# -------------------------------- TAB 2 TÉCNICOS --------------------------------
with tabs[1]:
    st.subheader("Indicadores técnicos")

    data["EMA_12"] = data["Close"].ewm(span=12).mean()
    data["EMA_26"] = data["Close"].ewm(span=26).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9).mean()
    data["RSI"] = 100 - (100/(1 + data["Close"].pct_change().apply(lambda x: max(x,0)).rolling(14).mean() /
                                     data["Close"].pct_change().apply(lambda x: max(-x,0)).rolling(14).mean()))

    fig1, ax1 = plt.subplots(figsize=(12,4))
    ax1.plot(data.index, data["Close"], label="Close")
    ax1.plot(data.index, data["EMA_12"], label="EMA12")
    ax1.plot(data.index, data["EMA_26"], label="EMA26")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12,3))
    ax2.plot(data.index, data["MACD"], label="MACD")
    ax2.plot(data.index, data["MACD_SIGNAL"], label="Signal")
    ax2.legend()
    st.pyplot(fig2)

    st.dataframe(data[["Close","EMA_12","EMA_26","MACD","MACD_SIGNAL","RSI"]].tail(10))

# -------------------------------- TAB 3 SPY --------------------------------
with tabs[2]:
    st.subheader("Comparativa SPY")
    combined = pd.DataFrame({
        stonk: data["Close"] / data["Close"].iloc[0],
        "SPY": spy["Close"] / spy["Close"].iloc[0]
    })
    st.line_chart(combined)

# -------------------------------- TAB 4 MONTE CARLO --------------------------------
with tabs[3]:
    st.subheader("Simulación Monte Carlo")

    log_returns = np.log(data["Close"] / data["Close"].shift()).dropna()
    mu = log_returns.mean()*252
    sigma = log_returns.std()*np.sqrt(252)

    S0 = data["Close"].iloc[-1]
    dt = 1/252
    T = mc_horizon_days/252

    sims = np.zeros((mc_sim, mc_horizon_days+1))
    sims[:,0] = S0

    for i in range(mc_sim):
        rand = np.random.normal(0,1, mc_horizon_days)
        for t in range(1, mc_horizon_days+1):
            sims[i,t] = sims[i,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand[t-1])

    fig_mc, ax = plt.subplots(figsize=(10,5))
    for i in range(min(20,mc_sim)):
        ax.plot(sims[i], alpha=0.5)
    st.pyplot(fig_mc)

# -------------------------------- TAB 5 BACKTEST EMA --------------------------------
with tabs[4]:
    st.subheader("Estrategia EMA20/EMA50")

    df = data.copy().dropna()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["signal"] = (df["EMA20"] > df["EMA50"]).astype(int)
    df["position"] = df["signal"].shift().fillna(0)
    df["ret"] = df["Close"].pct_change().fillna(0)
    df["strat"] = df["position"] * df["ret"]

    st.line_chart(pd.DataFrame({
        "Estrategia": (1+df["strat"]).cumprod(),
        "Buy&Hold": (1+df["ret"]).cumprod()
    }))

# -------------------------------- TAB 6 MULTI-COMPARATIVA --------------------------------
with tabs[5]:
    st.subheader("Comparador de acciones")

    tickers = [stonk] + multi_symbols
    resultados = {}

    for t in tickers:
        d = yf.download(t, start=start_date, end=end_date)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        resultados[t] = d["Close"]

    df_multi = pd.DataFrame(resultados).dropna()
    st.line_chart(df_multi/df_multi.iloc[0])

# -------------------------------- TAB 7 NOTICIAS (CON TRADUCCIÓN) --------------------------------
with tabs[6]:
    st.subheader(f"Noticias de {stonk}")

    if not NEWS_API_KEY:
        st.warning("Agrega tu NEWS_API_KEY arriba.")
    else:
        url = f"https://newsapi.org/v2/everything?q={stonk}&language=en&pageSize=5&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        r = requests.get(url)

        if r.status_code == 200:
            for a in r.json().get("articles", []):
                titulo = a.get("title","")
                desc = a.get("description","")
                link = a.get("url","")
                img = a.get("urlToImage")

                # traducir
                try:
                    titulo_es = GoogleTranslator(source="en", target="es").translate(titulo)
                    desc_es = GoogleTranslator(source="en", target="es").translate(desc)
                except:
                    titulo_es, desc_es = titulo, desc

                st.markdown(f"### [{titulo_es}]({link})")
                st.write(desc_es)
                if img:
                    st.image(img, use_column_width=True)
                st.markdown("---")
        else:
            st.error("Error cargando noticias")

# -------------------------------- TAB 8 INSIGHTS IA --------------------------------
with tabs[7]:
    st.subheader("Insights de IA")

    latest_price = data["Close"].iloc[-1]
    last_ret = data["Close"].pct_change().iloc[-1]
    vol = data["Close"].pct_change().std()*np.sqrt(252)
    rsi = data["RSI"].iloc[-1]
    macd = data["MACD"].iloc[-1]

    prompt = (
        f"Genera un insight financiero profesional en español sobre la acción {stonk}. "
        f"Precio actual {latest_price:.2f}, retorno diario {last_ret:.2%}, volatilidad anual {vol:.2%}. "
        f"RSI {rsi:.1f}, MACD {macd:.4f}. Sin dar recomendaciones de inversión."
    )

    if GENAI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            resp = model.generate_content(prompt)
            st.write(resp.text)
        except:
            st.error("Error con Gemini. Mostrando insight básico.")
            st.write("Análisis automático: volatilidad moderada, indicadores mixtos, requiere seguimiento.")
    else:
        st.write("Gemini no configurado. Insight simple:")
        st.write("Indicadores muestran señales mixtas y volatilidad estable. Observación recomendada.")

# -------------------------------- FOOTER --------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>© 2025 Héctor Estrada — PA' EL TASTYTRADE</p>", unsafe_allow_html=True)
