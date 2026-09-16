from __future__ import annotations

import io
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.dummy import DummyRegressor
except Exception:  # pragma: no cover
    RandomForestRegressor = None
    mean_absolute_error = None
    r2_score = None
    train_test_split = None

st.set_page_config(page_title="EcoRadar | Inteligência Hídrica e Atmosférica", page_icon="🌊", layout="wide")

def obter_css_tema(modo_visual: str) -> str:
    if modo_visual == "Noturno":
        return """
        <style>
        :root {
            --bg-main: #030916;
            --bg-secondary: #071427;
            --panel: rgba(7, 21, 39, 0.92);
            --panel-soft: rgba(7, 21, 39, 0.78);
            --panel-strong: rgba(5, 18, 35, 0.96);
            --line: rgba(100, 181, 246, 0.18);
            --line-strong: rgba(148, 197, 255, 0.30);
            --text-main: #eef6ff;
            --text-soft: #9fc1e3;
            --text-muted: #6f93b6;
            --accent: #4cc9f0;
            --accent-2: #2f80ed;
            --shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
            --radius-xl: 28px;
            --radius-lg: 22px;
            --radius-md: 18px;
        }
        html, body, [class*="css"]  {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(44, 130, 201, 0.16), transparent 24%),
                radial-gradient(circle at 100% 0%, rgba(65, 182, 230, 0.18), transparent 20%),
                linear-gradient(180deg, #020611 0%, #05111f 50%, #020814 100%);
            color: var(--text-main);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2.4rem;
            max-width: 1380px;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #031020 0%, #07182b 100%);
            border-right: 1px solid rgba(120, 180, 255, 0.12);
            min-width: 340px !important;
            max-width: 340px !important;
        }
        [data-testid="stSidebar"] * {
            color: #ebf5ff;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stDateInput input,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] [data-baseweb="base-input"] {
            background: rgba(255,255,255,0.04) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
        }
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            gap: 0.4rem;
        }
        [data-testid="stSidebar"] .stRadio label {
            background: rgba(255,255,255,0.04);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
        }
        .hero-box {
            background:
                radial-gradient(circle at 90% 10%, rgba(118, 211, 255, 0.18), transparent 26%),
                linear-gradient(100deg, rgba(3, 14, 33, 0.98) 0%, rgba(7, 32, 58, 0.98) 48%, rgba(59, 156, 197, 0.92) 100%);
            color: white;
            border-radius: var(--radius-xl);
            padding: 28px 28px 22px 28px;
            margin-bottom: 1rem;
            box-shadow: 0 24px 60px rgba(0, 6, 18, 0.42);
            border: 1px solid rgba(152, 219, 255, 0.14);
            position: relative;
            overflow: hidden;
        }
        .hero-box::after {
            content: "";
            position: absolute;
            right: -40px;
            top: -40px;
            width: 300px;
            height: 300px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(155, 223, 255, 0.16), transparent 60%);
            pointer-events: none;
        }
        .hero-copy {
            position: relative;
            z-index: 1;
        }
        .hero-box h1 {
            margin: 0;
            font-size: 2.05rem;
            line-height: 1.08;
            letter-spacing: -0.04em;
            color: #ffffff;
        }
        .hero-subtitle {
            margin-top: 0.35rem;
            font-size: 0.95rem;
            color: rgba(235, 245, 255, 0.88);
        }
        .hero-badges {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .hero-badge {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.72);
            border-radius: 999px;
            padding: 0.38rem 0.82rem;
            font-size: 0.84rem;
            font-weight: 600;
            color: #f8fbff;
            backdrop-filter: blur(6px);
        }
        .top-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }
        .top-kpi-card {
            background: linear-gradient(180deg, rgba(4, 16, 31, 0.94) 0%, rgba(6, 22, 42, 0.88) 100%);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-height: 124px;
            position: relative;
            overflow: hidden;
        }
        .top-kpi-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, rgba(87, 170, 255, 0.0), rgba(118, 211, 255, 0.75), rgba(87, 170, 255, 0.0));
            opacity: 0.5;
        }
        .kpi-label {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--text-soft);
            margin-bottom: 0.7rem;
        }
        .kpi-value {
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -0.04em;
            color: var(--text-main);
            margin-bottom: 0.38rem;
        }
        .kpi-sub {
            color: var(--text-muted);
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .mini-note {
            background: rgba(8, 20, 37, 0.86);
            color: #d7e9fb;
            border: 1px solid rgba(120, 180, 255, 0.14);
            padding: 0.95rem 1rem;
            border-radius: 18px;
            margin-bottom: 0.85rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
        }
        .section-card {
            background: linear-gradient(180deg, rgba(7, 20, 37, 0.92) 0%, rgba(7, 18, 33, 0.86) 100%);
            border: 1px solid rgba(120, 180, 255, 0.12);
            border-radius: 22px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            margin-bottom: 0.8rem;
            backdrop-filter: blur(8px);
        }
        .section-card h3 {
            margin: 0 0 0.35rem 0;
            font-size: 1.05rem;
            color: #eef6ff;
            letter-spacing: -0.02em;
        }
        .section-card p {
            margin: 0;
            color: #9fc1e3;
            font-size: 0.94rem;
            line-height: 1.55;
        }
        .section-card .card-footnote {
            margin-top: 10px;
            color: #6f93b6;
            font-size: 0.84rem;
        }
        .status-ok, .status-warn, .status-neutral {
            display: inline-block;
            padding: 0.24rem 0.58rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.76rem;
            border: 1px solid transparent;
        }
        .status-ok {
            background: rgba(34, 197, 94, 0.12);
            color: #b8f7d2;
            border-color: rgba(34, 197, 94, 0.18);
        }
        .status-warn {
            background: rgba(245, 158, 11, 0.12);
            color: #fde68a;
            border-color: rgba(245, 158, 11, 0.18);
        }
        .status-neutral {
            background: rgba(148, 163, 184, 0.10);
            color: #dbeafe;
            border-color: rgba(148, 163, 184, 0.16);
        }
        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, rgba(5, 18, 35, 0.96) 0%, rgba(7, 20, 37, 0.90) 100%);
            border: 1px solid rgba(120, 180, 255, 0.14);
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.20);
        }
        div[data-testid="metric-container"] * {
            color: #eef6ff !important;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.55rem;
            margin-bottom: 0.8rem;
            flex-wrap: wrap;
        }
        div[data-baseweb="tab"] {
            min-height: 42px;
            border-radius: 999px;
            padding: 0 16px;
            background: rgba(7, 20, 37, 0.92);
            color: #d9ebff;
            border: 1px solid rgba(255,255,255,0.20);
            box-shadow: none;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, rgba(9, 31, 55, 1) 0%, rgba(11, 42, 74, 1) 100%);
            border: 1px solid rgba(145, 216, 255, 0.38);
            color: #ffffff;
        }
        [data-testid="stDataFrame"], .stPlotlyChart {
            background: transparent;
        }
        .stExpander {
            border: 1px solid rgba(120, 180, 255, 0.12) !important;
            background: rgba(7, 20, 37, 0.76) !important;
            border-radius: 18px !important;
        }
        .stAlert {
            border-radius: 18px;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        @media (max-width: 980px) {
            .top-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .top-kpi-grid {
                grid-template-columns: 1fr;
            }
            .hero-box h1 {
                font-size: 1.7rem;
            }
        }
        </style>
        """
    return """
    <style>
    :root {
        --bg-main: #f4f9ff;
        --bg-secondary: #eaf3ff;
        --panel: rgba(255,255,255,0.96);
        --panel-soft: rgba(255,255,255,0.90);
        --line: rgba(61, 129, 195, 0.14);
        --line-strong: rgba(61, 129, 195, 0.24);
        --text-main: #0d2744;
        --text-soft: #3b6288;
        --text-muted: #6d89a6;
        --accent: #2563eb;
        --accent-2: #06b6d4;
        --shadow: 0 18px 40px rgba(30, 64, 105, 0.08);
    }
    html, body, [class*="css"]  {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(114, 182, 255, 0.18), transparent 22%),
            linear-gradient(180deg, #f7fbff 0%, #edf6ff 52%, #f8fbff 100%);
        color: var(--text-main);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2.4rem;
        max-width: 1380px;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e2844 0%, #153f67 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        min-width: 340px !important;
        max-width: 340px !important;
    }
    [data-testid="stSidebar"] * {
        color: #f8fbff;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stDateInput input,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    .hero-box {
        background:
            radial-gradient(circle at 90% 10%, rgba(255,255,255,0.20), transparent 24%),
            linear-gradient(100deg, #0d2744 0%, #14416e 48%, #50add1 100%);
        color: white;
        border-radius: 28px;
        padding: 28px 28px 22px 28px;
        margin-bottom: 1rem;
        box-shadow: 0 24px 60px rgba(27, 68, 112, 0.14);
        border: 1px solid rgba(255,255,255,0.20);
        position: relative;
        overflow: hidden;
    }
    .hero-box h1 {
        margin: 0;
        font-size: 2.05rem;
        line-height: 1.08;
        letter-spacing: -0.04em;
        color: #ffffff;
    }
    .hero-subtitle {
        margin-top: 0.35rem;
        font-size: 0.95rem;
        color: rgba(240, 248, 255, 0.94);
    }
    .hero-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .hero-badge {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.72);
        border-radius: 999px;
        padding: 0.38rem 0.82rem;
        font-size: 0.84rem;
        font-weight: 600;
        color: #ffffff;
    }
    .top-kpi-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(249,252,255,0.94) 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
        min-height: 124px;
    }
    .kpi-label {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: var(--text-soft);
        margin-bottom: 0.7rem;
    }
    .kpi-value {
        font-size: 2rem;
        line-height: 1;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: var(--text-main);
        margin-bottom: 0.38rem;
    }
    .kpi-sub {
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .mini-note {
        background: rgba(255,255,255,0.94);
        color: #21496f;
        border: 1px solid rgba(61, 129, 195, 0.14);
        padding: 0.95rem 1rem;
        border-radius: 18px;
        margin-bottom: 0.85rem;
        box-shadow: 0 12px 28px rgba(30, 64, 105, 0.06);
    }
    .section-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(61, 129, 195, 0.12);
        border-radius: 22px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 14px 28px rgba(30, 64, 105, 0.05);
        margin-bottom: 0.8rem;
    }
    .section-card h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1.05rem;
        color: #0d2744;
    }
    .section-card p {
        margin: 0;
        color: #456989;
        font-size: 0.94rem;
        line-height: 1.55;
    }
    .section-card .card-footnote {
        margin-top: 10px;
        color: #6d89a6;
        font-size: 0.84rem;
    }
    .stTabs [data-baseweb="tab-list"],
    div[data-baseweb="tab-list"] {
        gap: 0.55rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
        background: transparent !important;
    }
    .stTabs button[role="tab"],
    div[data-baseweb="tab"] {
        min-height: 42px;
        border-radius: 999px !important;
        padding: 0 16px !important;
        background: rgba(255,255,255,0.86) !important;
        color: #18456e !important;
        border: 1px solid rgba(61, 129, 195, 0.20) !important;
        box-shadow: none !important;
    }
    .stTabs button[role="tab"] p {
        color: #18456e !important;
        font-weight: 600 !important;
    }
    .stTabs button[role="tab"][aria-selected="true"],
    div[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f8ff 100%) !important;
        color: #0d2744 !important;
        border: 1px solid rgba(59, 130, 246, 0.40) !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08) !important;
    }
    .stTabs button[role="tab"][aria-selected="true"] p {
        color: #0d2744 !important;
    }
    .stTabs [data-testid="stMarkdownContainer"] p {
        color: inherit;
    }
    .stPlotlyChart {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(61, 129, 195, 0.12);
        border-radius: 22px;
        padding: 0.35rem;
        box-shadow: 0 10px 24px rgba(30, 64, 105, 0.05);
    }
    [data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.72);
        border-radius: 20px;
        border: 1px solid rgba(61, 129, 195, 0.10);
    }
    .stExpander {
        border: 1px solid rgba(61, 129, 195, 0.12) !important;
        background: rgba(255,255,255,0.78) !important;
        border-radius: 18px !important;
    }
    .stAlert {
        border-radius: 18px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    @media (max-width: 980px) {
        .hero-box h1 {
            font-size: 1.8rem;
        }
    }
    @media (max-width: 640px) {
        .hero-box h1 {
            font-size: 1.7rem;
        }
    }
    </style>
    """


def aplicar_tema_interface(modo_visual: str) -> None:
    st.markdown(obter_css_tema(modo_visual), unsafe_allow_html=True)
    px.defaults.template = "plotly_dark" if modo_visual == "Noturno" else "plotly_white"
    st.session_state["modo_visual_ecoradar"] = modo_visual

def estilizar_figura(fig: go.Figure) -> go.Figure:
    modo_visual = st.session_state.get("modo_visual_ecoradar", "Noturno")
    if modo_visual == "Noturno":
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2f3ff"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#12385c"),
            title_font=dict(color="#0d2744"),
            legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color="#12385c")),
            xaxis=dict(
                title_font=dict(color="#33597d"),
                tickfont=dict(color="#55789a"),
                gridcolor="rgba(62, 109, 156, 0.12)",
                zerolinecolor="rgba(62, 109, 156, 0.16)",
            ),
            yaxis=dict(
                title_font=dict(color="#33597d"),
                tickfont=dict(color="#55789a"),
                gridcolor="rgba(62, 109, 156, 0.12)",
                zerolinecolor="rgba(62, 109, 156, 0.16)",
            ),
        )
        fig.update_annotations(font=dict(color="#33597d"))
    return fig

def renderizar_grafico(fig: go.Figure, **kwargs: Any) -> None:
    st.plotly_chart(estilizar_figura(fig), **kwargs)

REQUEST_TIMEOUT = 20
REQUEST_TIMEOUT = 20
DEFAULT_PM25_DAYS = 15
DEFAULT_CLIMATE_DAYS = 15

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CIDADES = {
    "Aparecida, SP": {"lat": -22.8469, "lon": -45.2297, "ibge": 3502507, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Bananal, SP": {"lat": -22.6819, "lon": -44.3234, "ibge": 3504909, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Caçapava, SP": {"lat": -23.1002, "lon": -45.7064, "ibge": 3508504, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Campos do Jordão, SP": {"lat": -22.7394, "lon": -45.5914, "ibge": 3509700, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Canas, SP": {"lat": -22.7007, "lon": -45.0530, "ibge": 3509957, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Cachoeira Paulista, SP": {"lat": -22.6666, "lon": -45.0094, "ibge": 3508603, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Cruzeiro, SP": {"lat": -22.5765, "lon": -44.9656, "ibge": 3513405, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Guaratinguetá, SP": {"lat": -22.8166, "lon": -45.1927, "ibge": 3518404, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Jacareí, SP": {"lat": -23.3053, "lon": -45.9658, "ibge": 3524402, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Lavrinhas, SP": {"lat": -22.5704, "lon": -44.9026, "ibge": 3526605, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Lorena, SP": {"lat": -22.7333, "lon": -45.1236, "ibge": 3527207, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Pindamonhangaba, SP": {"lat": -22.9246, "lon": -45.4613, "ibge": 3538006, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Piquete, SP": {"lat": -22.6136, "lon": -45.1830, "ibge": 3538501, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Potim, SP": {"lat": -22.8343, "lon": -45.2554, "ibge": 3540754, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Queluz, SP": {"lat": -22.5312, "lon": -44.7782, "ibge": 3539905, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Roseira, SP": {"lat": -22.8940, "lon": -45.3058, "ibge": 3544301, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Santa Branca, SP": {"lat": -23.3963, "lon": -45.8832, "ibge": 3546009, "uf": "SP", "regiao": "Vale do Paraíba"},
    "São Bento do Sapucaí, SP": {"lat": -22.6837, "lon": -45.7312, "ibge": 3548609, "uf": "SP", "regiao": "Vale do Paraíba"},
    "São José dos Campos, SP": {"lat": -23.1791, "lon": -45.8872, "ibge": 3549904, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Taubaté, SP": {"lat": -23.0273, "lon": -45.5553, "ibge": 3554102, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Tremembé, SP": {"lat": -22.9584, "lon": -45.5494, "ibge": 3554805, "uf": "SP", "regiao": "Vale do Paraíba"},
    "Caraguatatuba, SP": {"lat": -23.6197, "lon": -45.4128, "ibge": 3510500, "uf": "SP", "regiao": "Litoral Norte"},
    "Ubatuba, SP": {"lat": -23.4332, "lon": -45.0834, "ibge": 3555406, "uf": "SP", "regiao": "Litoral Norte"},
    "São Sebastião, SP": {"lat": -23.7602, "lon": -45.4042, "ibge": 3550704, "uf": "SP", "regiao": "Litoral Norte"},
    "Ilhabela, SP": {"lat": -23.7781, "lon": -45.3581, "ibge": 3520400, "uf": "SP", "regiao": "Litoral Norte"},
}

REGIOES = {
    "Área completa do estudo": list(CIDADES.keys()),
    "Vale do Paraíba": [cidade for cidade, info in CIDADES.items() if info["regiao"] == "Vale do Paraíba"],
    "Litoral Norte": [cidade for cidade, info in CIDADES.items() if info["regiao"] == "Litoral Norte"],
}

CETESB_BALNEABILIDADE_URL = (
    "https://arcgis.cetesb.sp.gov.br/server/rest/services/Hosted/Praias/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&returnGeometry=true&f=geojson"
)

MANUAL_SOURCE_CONFIG = {
    "CETESB_BALNEABILIDADE": ["cetesb.csv", "cetesb.xlsx", "cetesb.xls", "balneabilidade.csv", "balneabilidade.xlsx", "balneabilidade.xls"],
    "SIMQUA": ["simqua.csv", "simqua.xlsx", "simqua.xls", "agua_cetesb.csv", "agua_cetesb.xlsx"],
}

GUIDES_FONTES = {
    "CETESB_BALNEABILIDADE": "A CETESB tenta carregar balneabilidade automaticamente por API REST. Se falhar, use um CSV/XLSX local.",
    "SIMQUA": "Use um CSV/XLSX exportado do SIMQUA com colunas de data, pH, turbidez, precipitação, nível ou vazão.",
    "OPENMETEO_AR": "Os poluentes do ar são carregados automaticamente pela API Open-Meteo: PM2.5, PM10, NO2, O3 e SO2.",
}


POPULACAO_FALLBACK_IBGE = {
    3502507: 32569,  # Aparecida (SP) — Censo 2022
    3504909: 9969,   # Bananal (SP) — Censo 2022
    3555406: 92981,  # Ubatuba (SP) — Censo 2022
}



def exibir_logo_sidebar() -> None:
    candidatos = [
        BASE_DIR / "ecoradar_logo.png",
        BASE_DIR / "logo_ecoradar.png",
        BASE_DIR / "a_digital_graphic_design_logo_for_ecoradar_a_wate.png",
    ]
    caminho_logo = next((c for c in candidatos if c.exists()), None)
    if caminho_logo is None:
        return

    st.markdown("<div style='padding-top:0.2rem'></div>", unsafe_allow_html=True)
    st.image(str(caminho_logo), use_container_width=True)
    st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)

def normalizar_texto(texto: Any) -> str:
    if texto is None:
        return ""
    texto = str(texto).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))

def detectar_coluna(df: pd.DataFrame, candidatos: list[str]) -> Optional[str]:
    if df.empty:
        return None
    mapa = {normalizar_texto(col): col for col in df.columns}
    for candidato in candidatos:
        if normalizar_texto(candidato) in mapa:
            return mapa[normalizar_texto(candidato)]
    for coluna in df.columns:
        nome = normalizar_texto(coluna)
        for candidato in candidatos:
            cand = normalizar_texto(candidato)
            if cand == nome or cand in nome:
                return coluna
    return None

def formatar_inteiro_br(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")

def formatar_decimal_br(valor: float, casas: int = 1, sufixo: str = "") -> str:
    if pd.isna(valor):
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",") + sufixo

def classificar_pm25(valor: float) -> tuple[str, str, str]:
    if pd.isna(valor):
        return "Sem leitura", "status-neutral", "Não há leitura suficiente para classificar o material particulado fino."
    if valor <= 15:
        return "Melhor cenário", "status-ok", "A média está próxima da referência de 15 μg/m³ usada como linha de interpretação no painel."
    if valor <= 25:
        return "Atenção", "status-warn", "O material particulado está acima da referência do painel e merece discussão ambiental."
    return "Pressão elevada", "status-warn", "O valor indica pressão ambiental mais alta e reforça a leitura crítica da qualidade do ar."


def formatar_card_valor(valor: Any, casas: int = 1, unidade: str = "", vazio: str = "—") -> str:
    if valor is None or pd.isna(valor):
        return vazio
    numero = float(valor)
    if float(numero).is_integer() and casas == 0:
        base = f"{int(numero)}"
    else:
        base = f"{numero:.{casas}f}".replace(".", ",")
        if casas == 0:
            base = base.split(",")[0]
    return f"{base}{unidade}"


def resumir_nome_area(nome_area: str) -> str:
    if normalizar_texto(nome_area) == normalizar_texto("Área completa do estudo"):
        return "Área completa"
    return nome_area



def montar_metricas_topo_resumo(df_atual: pd.DataFrame, nome_area: str) -> list[dict[str, str]]:
    base = df_atual.copy()
    for coluna in ["População", "PM2.5", "Temperatura"]:
        if coluna in base.columns:
            base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    municipios = int(base["Cidade"].nunique()) if not base.empty and "Cidade" in base.columns else 0
    populacao_total = int(base["População"].fillna(0).sum()) if "População" in base.columns else 0
    pm25_medio = float(base["PM2.5"].dropna().mean()) if "PM2.5" in base.columns and base["PM2.5"].notna().any() else np.nan
    temperatura_media = float(base["Temperatura"].dropna().mean()) if "Temperatura" in base.columns and base["Temperatura"].notna().any() else np.nan

    return [
        {
            "icone": "⌘",
            "titulo": "Recorte",
            "valor": resumir_nome_area(nome_area),
            "sub": "Área selecionada no painel",
        },
        {
            "icone": "◫",
            "titulo": "Municípios",
            "valor": formatar_inteiro_br(municipios),
            "sub": "Cidades monitoradas",
        },
        {
            "icone": "◎",
            "titulo": "População total",
            "valor": formatar_inteiro_br(populacao_total),
            "sub": "Estimativa IBGE agregada",
        },
        {
            "icone": "◌",
            "titulo": "PM2.5 médio",
            "valor": formatar_card_valor(pm25_medio, 1, " μg/m³"),
            "sub": "Indicador do ar com maior destaque",
        },
        {
            "icone": "⌁",
            "titulo": "Temperatura média",
            "valor": formatar_card_valor(temperatura_media, 1, " °C"),
            "sub": "Média das estações",
        },
    ]


def montar_metricas_topo_detalhe(df_atual: pd.DataFrame, df_simqua: pd.DataFrame) -> list[dict[str, str]]:
    temperatura = pd.to_numeric(df_atual.get("Temperatura"), errors="coerce").mean() if not df_atual.empty and "Temperatura" in df_atual.columns else np.nan
    umidade = pd.to_numeric(df_atual.get("Umidade"), errors="coerce").mean() if not df_atual.empty and "Umidade" in df_atual.columns else np.nan
    ph = pd.to_numeric(df_simqua.get("ph"), errors="coerce").mean() if not df_simqua.empty and "ph" in df_simqua.columns else np.nan
    turbidez = pd.to_numeric(df_simqua.get("turbidez"), errors="coerce").mean() if not df_simqua.empty and "turbidez" in df_simqua.columns else np.nan

    return [
        {
            "icone": "⌁",
            "titulo": "Temperatura",
            "valor": formatar_card_valor(temperatura, 1, "°C"),
            "sub": "Média das estações",
        },
        {
            "icone": "◌",
            "titulo": "pH da água",
            "valor": formatar_card_valor(ph, 1, ""),
            "sub": "Índice médio observado",
        },
        {
            "icone": "◔",
            "titulo": "Turbidez",
            "valor": formatar_card_valor(turbidez, 0, " NTU"),
            "sub": "Nível estimado na base",
        },
        {
            "icone": "≋",
            "titulo": "Umidade",
            "valor": formatar_card_valor(umidade, 0, "%"),
            "sub": "Relativa do ar",
        },
    ]


def exibir_topo_dashboard(df_atual: pd.DataFrame, df_simqua: pd.DataFrame, nome_area: str, nomes_area: list[str]) -> None:
    municipios_texto = ", ".join(nomes_area)
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-copy">
                <h1>EcoRadar</h1>
                <div class="hero-subtitle">Inteligência Hídrica e Atmosférica</div>
                <div class="hero-badges">
                    <span class="hero-badge">◌ Qualidade da Água</span>
                    <span class="hero-badge">≋ Dados Atmosféricos</span>
                    <span class="hero-badge">⌁ Predição ML</span>
                    <span class="hero-badge">↯ Tempo Real</span>
                </div>
                <div style="margin-top:1rem;padding:0.95rem 1rem;border-radius:18px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.18);color:#eef6ff;line-height:1.55;">
                    <strong>Municípios considerados:</strong> {municipios_texto}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards_resumo = montar_metricas_topo_resumo(df_atual, nome_area)
    colunas_resumo = st.columns(len(cards_resumo))
    for coluna, card in zip(colunas_resumo, cards_resumo):
        with coluna:
            st.markdown(
                f"""
                <div class='top-kpi-card'>
                    <div class='kpi-label'><span>{card['icone']}</span><span>{card['titulo']}</span></div>
                    <div class='kpi-value'>{card['valor']}</div>
                    <div class='kpi-sub'>{card['sub']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    cards_detalhe = montar_metricas_topo_detalhe(df_atual, df_simqua)
    colunas_detalhe = st.columns(len(cards_detalhe))
    for coluna, card in zip(colunas_detalhe, cards_detalhe):
        with coluna:
            st.markdown(
                f"""
                <div class='top-kpi-card'>
                    <div class='kpi-label'><span>{card['icone']}</span><span>{card['titulo']}</span></div>
                    <div class='kpi-value'>{card['valor']}</div>
                    <div class='kpi-sub'>{card['sub']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def nomes_municipios(cidades: list[str]) -> list[str]:

    return [cidade.split(",")[0].strip() for cidade in cidades]

def obter_cidades_area(area: str) -> list[str]:
    return REGIOES.get(area, [area])

def area_tem_litoral(cidades: list[str]) -> bool:
    return any(CIDADES[cidade]["regiao"] == "Litoral Norte" for cidade in cidades)

def converter_texto_para_numero(valor: Any) -> float | int | None:
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return valor
    texto = str(valor).strip()
    if not texto or texto.lower() in {"nan", "none", "null", "n/a", "na", "-", "--"}:
        return np.nan
    texto = texto.replace(" ", " ").replace(" ", "")
    texto = re.sub(r"[^0-9,\.\-]", "", texto)
    if not texto or texto in {"-", ".", ","}:
        return np.nan
    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif texto.count(",") > 1:
        partes = texto.split(",")
        texto = "".join(partes[:-1]) + "." + partes[-1]
    elif texto.count(".") > 1:
        partes = texto.split(".")
        texto = "".join(partes[:-1]) + "." + partes[-1]
    elif "," in texto:
        texto = texto.replace(",", ".")
    return pd.to_numeric(texto, errors="coerce")

def sanitizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().replace('"', "") for c in df.columns]
    possiveis_numericas = [
        "latitude", "lat", "longitude", "lon", "x", "y", "coord", "utm",
        "temperatura", "umidade", "vento", "velocidade", "precipit", "chuva",
        "ph", "turbidez", "vazao", "vazão", "nivel", "nível", "cota", "condutividade",
        "pm25", "pm2.5", "pm10", "no2", "o3", "so2", "co", "aqi", "valor"
    ]
    for col in df.columns:
        col_norm = normalizar_texto(col)
        if any(p in col_norm for p in possiveis_numericas):
            df[col] = df[col].apply(converter_texto_para_numero)
    return df
def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "EcoRadar-OpenMeteo/1.0"})
    return session

def fetch_json(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[dict[str, Any]]:
    try:
        response = get_session().get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=86400)
def carregar_cetesb_balneabilidade_auto() -> pd.DataFrame:
    data = fetch_json(CETESB_BALNEABILIDADE_URL)
    if not data or "features" not in data:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry", {}) or {}
        coords = geom.get("coordinates") or []
        row = dict(props)
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            row["Longitude"] = coords[0]
            row["Latitude"] = coords[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "classificacao_texto" in df.columns and "Classificação" not in df.columns:
        df["Classificação"] = df["classificacao_texto"]
    if "data_atual" in df.columns:
        try:
            df["data_atual"] = pd.to_datetime(df["data_atual"], errors="coerce", unit="ms")
        except Exception:
            df["data_atual"] = pd.to_datetime(df["data_atual"], errors="coerce")
    return sanitizar_dataframe(df)

def ler_bytes_tabulares(conteudo: bytes, nome_arquivo: str) -> pd.DataFrame:
    nome = nome_arquivo.lower()
    tentativas = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]
    if nome.endswith((".xlsx", ".xls")):
        try:
            return sanitizar_dataframe(pd.read_excel(io.BytesIO(conteudo)))
        except Exception:
            pass
    for params in tentativas:
        try:
            df = pd.read_csv(io.BytesIO(conteudo), low_memory=False, **params)
            if not df.empty and len(df.columns) > 1:
                return sanitizar_dataframe(df)
        except Exception:
            continue
    return pd.DataFrame()

def carregar_arquivo_tabular(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    if isinstance(uploaded_file, (list, tuple)):
        bases = []
        for arquivo in uploaded_file:
            if arquivo is None:
                continue
            df_item = ler_bytes_tabulares(arquivo.getvalue(), arquivo.name)
            if not df_item.empty:
                df_item = df_item.copy()
                df_item["arquivo_origem"] = arquivo.name
                bases.append(df_item)
        return pd.concat(bases, ignore_index=True) if bases else pd.DataFrame()

    return ler_bytes_tabulares(uploaded_file.getvalue(), uploaded_file.name)

def listar_arquivos_data() -> list[str]:
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_file()])

def carregar_fonte_local(nome_fonte: str) -> tuple[pd.DataFrame, str]:
    for nome_arquivo in MANUAL_SOURCE_CONFIG.get(nome_fonte, []):
        caminho = DATA_DIR / nome_arquivo
        if caminho.exists():
            try:
                df = ler_bytes_tabulares(caminho.read_bytes(), caminho.name)
                if not df.empty:
                    return df, f"arquivo local: {caminho.name}"
            except Exception:
                continue
    return pd.DataFrame(), "sem arquivo local"

def carregar_fonte_hibrida(nome_fonte: str, uploaded_file=None) -> tuple[pd.DataFrame, str]:
    if nome_fonte == "CETESB_BALNEABILIDADE":
        auto_df = carregar_cetesb_balneabilidade_auto()
        if not auto_df.empty:
            return auto_df, "API/REST oficial"
    if uploaded_file is not None:
        up_df = carregar_arquivo_tabular(uploaded_file)
        if not up_df.empty:
            if isinstance(uploaded_file, (list, tuple)):
                nomes = [arquivo.name for arquivo in uploaded_file if arquivo is not None]
                if len(nomes) == 1:
                    return up_df, f"upload: {nomes[0]}"
                return up_df, f"upload múltiplo: {len(nomes)} arquivos"
            return up_df, f"upload: {uploaded_file.name}"
    return carregar_fonte_local(nome_fonte)

def aplicar_filtro_uf(df: pd.DataFrame, uf: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    col_uf = detectar_coluna(df, ["uf", "sigla_uf", "estado", "unidade_federativa"])
    if not col_uf:
        return df.copy()
    return df[df[col_uf].astype(str).str.upper().str.strip() == uf.upper()].copy()

def filtrar_dataset_por_municipios(df: pd.DataFrame, cidades: list[str], filtrar_uf_sp: bool = True) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    base = aplicar_filtro_uf(df.copy(), "SP") if filtrar_uf_sp else df.copy()
    municipios = [normalizar_texto(n) for n in nomes_municipios(cidades)]
    col_municipio = detectar_coluna(base, ["municipio", "município", "cidade", "localidade", "estacao", "estação", "nome_municipio", "nome"])
    if col_municipio:
        serie = base[col_municipio].astype(str).map(normalizar_texto)
        mascara = serie.apply(lambda valor: any(m in valor for m in municipios))
        filtrado = base[mascara].copy()
        return filtrado if not filtrado.empty else base
    return base

def aplicar_filtro_textual(df: pd.DataFrame, termo: str) -> pd.DataFrame:
    termo = normalizar_texto(termo)
    if not termo or df.empty:
        return df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_cols:
        return df.copy()
    mascara = pd.Series(False, index=df.index)
    for coluna in text_cols:
        mascara = mascara | df[coluna].astype(str).map(normalizar_texto).str.contains(termo, na=False)
    filtrado = df[mascara].copy()
    return filtrado if not filtrado.empty else df.copy()

def preparar_fonte(df: pd.DataFrame, cidades: list[str], busca_textual: str = "", aplicar_filtro_regional: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    diag = {
        "linhas_originais": len(df),
        "linhas_pos_filtro_regional": 0,
        "linhas_pos_busca": 0,
        "fallback_sem_filtro": False,
        "filtro_regional_aplicado": aplicar_filtro_regional,
    }
    if df.empty:
        return df, diag
    resultado = df.copy()
    if aplicar_filtro_regional:
        filtrado = filtrar_dataset_por_municipios(resultado, cidades, filtrar_uf_sp=True)
        diag["linhas_pos_filtro_regional"] = len(filtrado)
        if filtrado.empty and len(resultado) > 0:
            diag["fallback_sem_filtro"] = True
        else:
            resultado = filtrado
    else:
        diag["linhas_pos_filtro_regional"] = len(resultado)
    if busca_textual.strip():
        buscado = aplicar_filtro_textual(resultado, busca_textual.strip())
        diag["linhas_pos_busca"] = len(buscado)
        resultado = buscado
    else:
        diag["linhas_pos_busca"] = len(resultado)
    return resultado, diag

@st.cache_data(ttl=1800)
def buscar_clima_atual(lat: float, lon: float) -> Optional[dict[str, Any]]:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,uv_index"
        "&timezone=America%2FSao_Paulo"
    )
    data = fetch_json(url)
    return data.get("current") if data else None

@st.cache_data(ttl=1800)
def buscar_qualidade_ar_atual(lat: float, lon: float) -> Optional[dict[str, Any]]:
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        "&timezone=America%2FSao_Paulo"
    )
    data = fetch_json(url)
    return data.get("current") if data else None

@st.cache_data(ttl=3600)
def buscar_historico_pm25_real(lat: float, lon: float, dias: int = DEFAULT_PM25_DAYS) -> pd.DataFrame:
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide"
        f"&past_days={dias}&forecast_days=0"
        "&timezone=America%2FSao_Paulo"
    )
    data = fetch_json(url)
    if not data or "hourly" not in data:
        return pd.DataFrame(columns=["Data", "PM2.5 médio", "PM10 médio", "NO2 médio", "O3 médio", "SO2 médio", "Tendência (MM 3d)"])
    hourly = data["hourly"]
    df = pd.DataFrame({
        "DataHora": pd.to_datetime(hourly.get("time", []), errors="coerce"),
        "PM2.5": hourly.get("pm2_5", []),
        "PM10": hourly.get("pm10", []),
        "NO2": hourly.get("nitrogen_dioxide", []),
        "O3": hourly.get("ozone", []),
        "SO2": hourly.get("sulphur_dioxide", []),
    })
    if df.empty:
        return pd.DataFrame(columns=["Data", "PM2.5 médio", "PM10 médio", "NO2 médio", "O3 médio", "SO2 médio", "Tendência (MM 3d)"])
    df["Data"] = df["DataHora"].dt.date
    diario = (
        df.groupby("Data", as_index=False)[["PM2.5", "PM10", "NO2", "O3", "SO2"]]
        .mean(numeric_only=True)
        .rename(columns={
            "PM2.5": "PM2.5 médio",
            "PM10": "PM10 médio",
            "NO2": "NO2 médio",
            "O3": "O3 médio",
            "SO2": "SO2 médio",
        })
    )
    diario["Tendência (MM 3d)"] = diario["PM2.5 médio"].rolling(window=3).mean().round(1)
    return diario.copy()

@st.cache_data(ttl=3600)
def buscar_historico_clima_real(lat: float, lon: float, dias: int = DEFAULT_CLIMATE_DAYS) -> pd.DataFrame:
    data_final = date.today()
    data_inicial = data_final - timedelta(days=dias - 1)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={data_inicial.isoformat()}"
        f"&end_date={data_final.isoformat()}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
        "&timezone=America%2FSao_Paulo"
    )
    data = fetch_json(url)
    if not data or "daily" not in data:
        return pd.DataFrame(columns=["Data", "Temp Máx", "Temp Mín", "Chuva", "Vento Máx"])
    return pd.DataFrame({
        "Data": pd.to_datetime(data["daily"]["time"]),
        "Temp Máx": data["daily"]["temperature_2m_max"],
        "Temp Mín": data["daily"]["temperature_2m_min"],
        "Chuva": data["daily"]["precipitation_sum"],
        "Vento Máx": data["daily"]["wind_speed_10m_max"],
    })

@st.cache_data(ttl=86400)
def puxar_populacao_ibge_valor(cod_ibge: int) -> Optional[int]:
    url = (
        "https://servicodados.ibge.gov.br/api/v3/agregados/9514/periodos/2022/variaveis/93"
        f"?localidades=N6[{cod_ibge}]"
    )
    data = fetch_json(url, timeout=15)
    if not data:
        return POPULACAO_FALLBACK_IBGE.get(cod_ibge)
    try:
        valor = int(data[0]["resultados"][0]["series"][0]["serie"]["2022"])
        if valor > 0:
            return valor
    except Exception:
        pass
    return POPULACAO_FALLBACK_IBGE.get(cod_ibge)

@st.cache_data(ttl=1800)
def montar_painel_regional(cidades: tuple[str, ...], dias_pm25: int, dias_clima: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registros, h_ar, h_clima = [], [], []
    for cidade in cidades:
        info = CIDADES[cidade]
        clima = buscar_clima_atual(info["lat"], info["lon"])
        ar = buscar_qualidade_ar_atual(info["lat"], info["lon"])
        pop = puxar_populacao_ibge_valor(info["ibge"])
        hist_ar = buscar_historico_pm25_real(info["lat"], info["lon"], dias=dias_pm25)
        hist_cl = buscar_historico_clima_real(info["lat"], info["lon"], dias=dias_clima)
        if clima and ar:
            registros.append({
                "Cidade": cidade.split(",")[0].strip(),
                "Região": info["regiao"],
                "Latitude": info["lat"],
                "Longitude": info["lon"],
                "População": pop,
                "Temperatura": float(clima.get("temperature_2m", 0)),
                "Umidade": float(clima.get("relative_humidity_2m", 0)),
                "Vento": float(clima.get("wind_speed_10m", 0)),
                "Chuva": float(clima.get("precipitation", 0)),
                "PM2.5": float(ar.get("pm2_5", 0)),
                "PM10": float(ar.get("pm10", 0)),
                "NO2": float(ar.get("nitrogen_dioxide", 0)),
                "SO2": float(ar.get("sulphur_dioxide", 0)),
                "O3": float(ar.get("ozone", 0)),
                "Hora atualização": ar.get("time"),
            })
        if not hist_ar.empty:
            tmp = hist_ar.copy()
            tmp["Cidade"] = cidade.split(",")[0].strip()
            tmp["Região"] = info["regiao"]
            h_ar.append(tmp)
        if not hist_cl.empty:
            tmp = hist_cl.copy()
            tmp["Cidade"] = cidade.split(",")[0].strip()
            tmp["Região"] = info["regiao"]
            h_clima.append(tmp)
    return (
        pd.DataFrame(registros),
        pd.concat(h_ar, ignore_index=True) if h_ar else pd.DataFrame(),
        pd.concat(h_clima, ignore_index=True) if h_clima else pd.DataFrame(),
    )

def montar_resumo_executivo(df_atual: pd.DataFrame) -> list[dict[str, str]]:
    if df_atual.empty:
        return []
    pm25_medio = float(df_atual["PM2.5"].mean()) if "PM2.5" in df_atual.columns else np.nan
    cidade_pm25 = df_atual.sort_values(by="PM2.5", ascending=False).iloc[0]["Cidade"] if "PM2.5" in df_atual.columns else "—"
    cidade_chuva = df_atual.sort_values(by="Chuva", ascending=False).iloc[0]["Cidade"] if "Chuva" in df_atual.columns else "—"
    cidade_temp = df_atual.sort_values(by="Temperatura", ascending=False).iloc[0]["Cidade"] if "Temperatura" in df_atual.columns else "—"
    status_pm25, status_css, detalhe = classificar_pm25(pm25_medio)
    return [
        {
            "titulo": "Leitura regional do ar",
            "conteudo": f"PM2.5 médio de {formatar_decimal_br(pm25_medio, 1, ' μg/m³')} com destaque atual para {cidade_pm25}.",
            "badge": f"<span class='{status_css}'>{status_pm25}</span>",
            "rodape": detalhe,
        },
        {
            "titulo": "Município com maior chuva",
            "conteudo": f"No recorte atual, {cidade_chuva} aparece com o maior volume de precipitação entre os municípios carregados.",
            "badge": "<span class='status-neutral'>Clima</span>",
            "rodape": "Esse indicador ajuda a discutir drenagem, arraste e contraste com a poluição atmosférica.",
        },
        {
            "titulo": "Ponto mais quente",
            "conteudo": f"{cidade_temp} lidera a temperatura na leitura atual do painel.",
            "badge": "<span class='status-neutral'>Temperatura</span>",
            "rodape": "A variação térmica ajuda a contextualizar a dispersão de poluentes e o comportamento meteorológico.",
        },
    ]

def exibir_cards_resumo(df_atual: pd.DataFrame) -> None:
    cards = montar_resumo_executivo(df_atual)
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(
            f"""
            <div class='section-card'>
                <div style='display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px;'>
                    <h3>{card['titulo']}</h3>
                    {card['badge']}
                </div>
                <p>{card['conteudo']}</p>
                <div class='card-footnote'>{card['rodape']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def exibir_contexto_projeto(nome_area: str, nomes_area: list[str]) -> None:
    st.markdown(
        """
        <div class='section-card'>
            <h3>Integração principal do painel</h3>
            <p>O painel reúne dados de água, atmosfera e meteorologia em uma leitura regional única, usando Open-Meteo para clima e qualidade do ar, com apoio de mapas, séries temporais, comparações entre municípios e integração entre fontes ambientais.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def exibir_guia_local() -> None:
    with st.expander("Como alimentar as fontes complementares", expanded=False):
        st.markdown("O app segue esta ordem: **API/endpoint oficial → arquivo local em `data/` → upload manual**.")
        exemplos = [{"Fonte": fonte, "Exemplos de arquivo local": ", ".join(nomes[:3])} for fonte, nomes in MANUAL_SOURCE_CONFIG.items()]
        st.dataframe(pd.DataFrame(exemplos), use_container_width=True, hide_index=True)
        arquivos = listar_arquivos_data()
        if arquivos:
            st.dataframe(pd.DataFrame({"Arquivo": arquivos}), use_container_width=True, hide_index=True)
        else:
            st.warning(f"Nenhum arquivo encontrado na pasta `data/`. Caminho procurado: {DATA_DIR}")

def exibir_status_fontes(automaticas: dict[str, bool], origens: dict[str, str], dfs: dict[str, pd.DataFrame], diags: dict[str, dict[str, Any]]) -> None:
    st.markdown("### Integridade das fontes")

    cols = st.columns(max(1, len(automaticas)))
    for col, (nome, ok) in zip(cols, automaticas.items()):
        status = "status-ok" if ok else "status-warn"
        rotulo = "Ativa" if ok else "Falha"
        col.markdown(
            f"""
            <div class='section-card'>
                <h3 style='margin-bottom:10px'>{nome}</h3>
                <div><span class='{status}'>{rotulo}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if origens:
        st.markdown("### Fontes complementares")
        cols2 = st.columns(max(1, len(origens)))
        for col, nome in zip(cols2, origens.keys()):
            origem = origens[nome]
            df = dfs[nome]
            diag = diags[nome]
            detalhes = []
            if origem:
                detalhes.append(origem)
            if diag.get("linhas_originais", 0):
                detalhes.append(f"orig.: {diag.get('linhas_originais', 0)}")
            if diag.get("filtro_regional_aplicado"):
                detalhes.append(f"reg.: {diag.get('linhas_pos_filtro_regional', 0)}")
            if diag.get("fallback_sem_filtro"):
                detalhes.append("sem filtro regional")
            resumo = f"{len(df)} linhas prontas para uso" if not df.empty else "Sem linhas úteis detectadas"
            col.markdown(
                f"""
                <div class='section-card'>
                    <h3 style='margin-bottom:10px'>{nome}</h3>
                    <p style='margin-bottom:8px'>{resumo}</p>
                    <div style='color:#64748b;font-size:0.84rem'>{' • '.join(detalhes) if detalhes else 'Sem detalhes adicionais'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
def exibir_mapa_municipios(df_atual: pd.DataFrame, area: str) -> None:
    if df_atual.empty:
        st.info("Sem dados atuais para o mapa dos municípios.")
        return
    fig = px.scatter_map(
        df_atual,
        lat="Latitude",
        lon="Longitude",
        size="PM2.5",
        color="PM2.5",
        hover_name="Cidade",
        hover_data={"Região": True, "Temperatura": ":.1f", "Chuva": ":.1f", "Umidade": ":.0f", "PM2.5": ":.1f", "Latitude": False, "Longitude": False},
        zoom=7.2,
        center={"lat": df_atual["Latitude"].mean(), "lon": df_atual["Longitude"].mean()},
        title=f"Mapa dos municípios monitorados — {area}",
    )
    renderizar_grafico(fig, use_container_width=True, key=f"mapa_municipios_{normalizar_texto(area)}")

def _inferir_cidade_costeira(valor: Any) -> str:
    texto = normalizar_texto(valor)
    if not texto:
        return ""
    for cidade in ["Ubatuba", "Caraguatatuba", "São Sebastião", "Ilhabela"]:
        if normalizar_texto(cidade) in texto:
            return cidade
    return ""

def _extrair_lat_lon_de_texto(valor: Any) -> tuple[float | None, float | None]:
    if valor is None or pd.isna(valor):
        return None, None
    texto = str(valor)
    numeros = re.findall(r"-?\d+[\.,]?\d*", texto)
    if len(numeros) < 2:
        return None, None
    nums = []
    for n in numeros[:4]:
        try:
            nums.append(float(str(n).replace(",", ".")))
        except Exception:
            continue
    if len(nums) < 2:
        return None, None
    pares = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1)]
    for a, b in pares:
        lat, lon = (a, b)
        if -30 <= lat <= -20 and -50 <= lon <= -40:
            return lat, lon
        lat, lon = (b, a)
        if -30 <= lat <= -20 and -50 <= lon <= -40:
            return lat, lon
    return None, None

def preparar_mapa_qualidade_agua_cetesb(df: pd.DataFrame, cidades: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base = df.copy()
    col_lat = detectar_coluna(base, ["Latitude", "latitude", "lat", "y", "coord_y", "coordenada_y"])
    col_lon = detectar_coluna(base, ["Longitude", "longitude", "lon", "x", "coord_x", "coordenada_x"])
    col_geo = detectar_coluna(base, ["geometry", "geom", "shape", "wkt", "coordenadas", "localizacao", "localização"])
    col_municipio = detectar_coluna(base, ["municipio", "município", "cidade", "localidade", "munic"])
    col_praia = detectar_coluna(base, ["praia", "nome da praia", "nome_praia", "nome"])
    col_status = detectar_coluna(base, ["Classificação", "classificacao_texto", "classificacao", "classificação", "status", "situacao", "situação"])
    col_data = detectar_coluna(base, ["data_atual", "data", "data coleta", "data_medicao", "semana"])

    base["latitude_mapa"] = base[col_lat].apply(converter_texto_para_numero) if col_lat else np.nan
    base["longitude_mapa"] = base[col_lon].apply(converter_texto_para_numero) if col_lon else np.nan

    if col_geo:
        extraidos = base[col_geo].apply(_extrair_lat_lon_de_texto)
        base["latitude_geo"] = extraidos.apply(lambda x: x[0])
        base["longitude_geo"] = extraidos.apply(lambda x: x[1])
        base["latitude_mapa"] = base["latitude_mapa"].fillna(base["latitude_geo"])
        base["longitude_mapa"] = base["longitude_mapa"].fillna(base["longitude_geo"])

    if col_municipio:
        base["Municipio_mapa"] = base[col_municipio].astype(str).str.strip()
    elif col_praia:
        base["Municipio_mapa"] = base[col_praia].apply(_inferir_cidade_costeira)
    else:
        base["Municipio_mapa"] = ""

    municipios_validos = {cidade.split(",")[0].strip() for cidade in cidades if CIDADES[cidade]["regiao"] == "Litoral Norte"}
    if municipios_validos:
        base["Municipio_norm"] = base["Municipio_mapa"].map(normalizar_texto)
        alvo_norm = {normalizar_texto(c) for c in municipios_validos}
        filtrado = base[base["Municipio_norm"].isin(alvo_norm)].copy()
        if not filtrado.empty:
            base = filtrado

    base["Praia_mapa"] = base[col_praia].astype(str).str.strip() if col_praia else "Ponto CETESB"
    base["Classificacao_mapa"] = base[col_status].astype(str).str.strip() if col_status else "Sem classificação"
    base["Data_mapa"] = pd.to_datetime(base[col_data], errors="coerce", dayfirst=True) if col_data else pd.NaT

    latitudes, longitudes, origens = [], [], []
    for _, row in base.iterrows():
        lat = row.get("latitude_mapa")
        lon = row.get("longitude_mapa")
        municipio = str(row.get("Municipio_mapa", "")).strip()
        cidade_ref = None
        for chave, info in CIDADES.items():
            nome = chave.split(",")[0].strip()
            if normalizar_texto(nome) == normalizar_texto(municipio):
                cidade_ref = info
                break
        if pd.notna(lat) and pd.notna(lon) and -30 <= float(lat) <= -20 and -50 <= float(lon) <= -40:
            latitudes.append(float(lat))
            longitudes.append(float(lon))
            origens.append("CETESB")
        elif cidade_ref:
            latitudes.append(cidade_ref["lat"])
            longitudes.append(cidade_ref["lon"])
            origens.append("Município (fallback)")
        else:
            latitudes.append(np.nan)
            longitudes.append(np.nan)
            origens.append("Sem coordenada")

    base["latitude_mapa"] = latitudes
    base["longitude_mapa"] = longitudes
    base["Origem coordenada"] = origens
    base = base.dropna(subset=["latitude_mapa", "longitude_mapa"]).copy()
    return base

def exibir_mapa_costeiro_cetesb(df: pd.DataFrame, cidades: list[str], area: str) -> None:
    if df.empty:
        st.info("Sem dados da CETESB para o mapa costeiro.")
        return
    base = preparar_mapa_qualidade_agua_cetesb(df, cidades)
    if base.empty:
        st.info("Não foi possível montar o mapa da CETESB com as colunas disponíveis. Tente um arquivo com município, praia ou coordenadas.")
        return
    fig = px.scatter_map(
        base,
        lat="latitude_mapa",
        lon="longitude_mapa",
        color="Classificacao_mapa",
        hover_name="Praia_mapa",
        hover_data={
            "Municipio_mapa": True,
            "Origem coordenada": True,
            "Data_mapa": True,
            "latitude_mapa": False,
            "longitude_mapa": False,
        },
        zoom=8,
        center={"lat": base["latitude_mapa"].mean(), "lon": base["longitude_mapa"].mean()},
        title=f"Mapa de qualidade da água costeira — {area}",
    )
    renderizar_grafico(fig, use_container_width=True, key=f"mapa_costeiro_{normalizar_texto(area)}")
    if (base["Origem coordenada"] == "Município (fallback)").any():
        st.caption("Alguns pontos foram posicionados pelo centro do município porque a base da CETESB não trouxe coordenadas válidas em todas as linhas.")

def exibir_panorama_regional(df_atual: pd.DataFrame, df_pm25: pd.DataFrame, df_clima: pd.DataFrame, area: str) -> None:
    if df_atual.empty:
        st.error("Não foi possível montar o panorama atual da área selecionada.")
        return

    base = df_atual.copy()
    for coluna in ["População", "PM2.5", "Chuva", "Temperatura"]:
        if coluna in base.columns:
            base[coluna] = base[coluna].apply(converter_texto_para_numero)

    populacao_valida = pd.to_numeric(base.get("População"), errors="coerce") if "População" in base.columns else pd.Series(dtype=float)
    if "População" not in base.columns:
        base["População"] = np.nan
        base["População_plot"] = 20.0
    else:
        mediana_pop = populacao_valida.dropna().median()
        if pd.isna(mediana_pop) or mediana_pop <= 0:
            mediana_pop = 1.0
        base["População"] = populacao_valida
        base["População_plot"] = base["População"].fillna(mediana_pop).clip(lower=1)

    st.markdown(
        """
        <div class='section-card'>
            <h3>Síntese do panorama</h3>
            <p>Esta visão combina leituras automáticas de clima, poluentes atmosféricos e população para apresentar rapidamente o comportamento ambiental do recorte selecionado.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        base_bar = base.dropna(subset=["PM2.5", "Cidade"]).sort_values(by="PM2.5", ascending=False)
        if not base_bar.empty:
            fig = px.bar(base_bar, x="PM2.5", y="Cidade", orientation="h", color="PM2.5", text_auto=True, title="PM2.5 atual por município")
            renderizar_grafico(fig, use_container_width=True)
        else:
            st.info("Sem dados válidos de PM2.5 para o gráfico por município.")
    with col2:
        base_disp = base.dropna(subset=["PM2.5", "Chuva"]).copy()
        if not base_disp.empty:
            fig = px.scatter(
                base_disp,
                x="PM2.5",
                y="Chuva",
                size="População_plot",
                color="Região",
                hover_name="Cidade",
                hover_data={"População": ":,.0f", "População_plot": False},
                title="PM2.5 × chuva × população",
            )
            renderizar_grafico(fig, use_container_width=True)
        else:
            st.info("Sem dados válidos para montar a dispersão entre PM2.5 e chuva.")
    exibir_mapa_municipios(base, area)

    c3, c4 = st.columns(2)
    with c3:
        if not df_pm25.empty:
            fig = px.line(df_pm25, x="Data", y="PM2.5 médio", color="Cidade", markers=True, title=f"Histórico real de PM2.5 — {area}")
            fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Referência OMS")
            renderizar_grafico(fig, use_container_width=True)
        else:
            st.info("Sem histórico real de PM2.5.")
    with c4:
        if not df_clima.empty:
            fig = px.line(df_clima, x="Data", y="Chuva", color="Cidade", markers=True, title=f"Precipitação diária real — {area}")
            renderizar_grafico(fig, use_container_width=True)
        else:
            st.info("Sem histórico climático real.")

def exibir_aba_litoral(df_cetesb: pd.DataFrame, cidades: list[str], area: str) -> None:
    if not area_tem_litoral(cidades):
        st.warning("A área selecionada não inclui o Litoral Norte. Prefira Litoral Norte ou Área completa do estudo para esta aba.")
    st.markdown("### Balneabilidade e qualidade da água costeira")
    st.markdown(
        """
        <div class='section-card'>
            <h3>Leitura costeira</h3>
            <p>Esta aba resume a classificação das praias e posiciona os pontos da CETESB no mapa. Quando a base vier sem coordenadas válidas, o painel usa o município como fallback para não perder a leitura espacial.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if df_cetesb.empty:
        st.info(GUIDES_FONTES["CETESB_BALNEABILIDADE"])
        return
    col_status = detectar_coluna(df_cetesb, ["Classificação", "classificacao_texto", "classificacao", "classificação", "status", "situacao", "situação"])
    col_municipio = detectar_coluna(df_cetesb, ["municipio", "município", "cidade", "localidade"])
    col_praia = detectar_coluna(df_cetesb, ["praia", "nome da praia"])
    if col_status and col_municipio:
        resumo = df_cetesb.groupby([col_municipio, col_status]).size().reset_index(name="Quantidade")
        fig = px.bar(resumo, x=col_municipio, y="Quantidade", color=col_status, barmode="group", title="Balneabilidade por município")
        renderizar_grafico(fig, use_container_width=True)
    else:
        cols_show = [c for c in [col_municipio, col_praia, col_status] if c] or df_cetesb.columns.tolist()[:8]
        st.dataframe(df_cetesb[cols_show].head(100), use_container_width=True, hide_index=True)
    exibir_mapa_costeiro_cetesb(df_cetesb, cidades, area)
def _mapear_municipio_estudo(valor: Any, cidades: list[str]) -> str:
    texto = str(valor).strip()
    texto_norm = normalizar_texto(texto)
    for cidade in cidades:
        nome = cidade.split(",")[0].strip()
        if normalizar_texto(nome) in texto_norm:
            return nome
    return texto.strip()

def converter_datas_simqua(serie: pd.Series) -> pd.Series:
    """Converte formatos ISO e brasileiros sem trocar mês e dia."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")
    texto = serie.astype("string").str.strip()
    saida = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    iso = texto.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    for formato in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        pendente = iso & saida.isna()
        saida.loc[pendente] = pd.to_datetime(texto.loc[pendente], format=formato, errors="coerce")
    for formato in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        pendente = ~iso & saida.isna()
        saida.loc[pendente] = pd.to_datetime(texto.loc[pendente], format=formato, errors="coerce")
    return saida

def _preparar_base_fonte(
    df: pd.DataFrame,
    cidades: list[str],
    date_candidates: list[str],
    city_candidates: list[str],
    metric_map: dict[str, list[str]],
    default_city_label: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base = df.copy()
    col_data = detectar_coluna(base, date_candidates)
    if not col_data:
        col_data = detectar_coluna(base, ["ano", "mes", "mês"])
    col_cidade = detectar_coluna(base, city_candidates)
    if detectar_coluna(base, ["ano"]) and detectar_coluna(base, ["mes", "mês"]) and col_data in {"ano", "mes", "mês", None}:
        ano_col = detectar_coluna(base, ["ano"])
        mes_col = detectar_coluna(base, ["mes", "mês"])
        ano = pd.to_numeric(base[ano_col], errors="coerce")
        mes = pd.to_numeric(base[mes_col], errors="coerce")
        base["data_ref"] = pd.to_datetime(dict(year=ano, month=mes, day=1), errors="coerce")
    elif col_data:
        base["data_ref"] = converter_datas_simqua(base[col_data])
    else:
        return pd.DataFrame()
    if col_cidade:
        base["municipio"] = base[col_cidade].apply(lambda x: _mapear_municipio_estudo(x, cidades))
    else:
        base["municipio"] = default_city_label
    base["municipio_norm"] = base["municipio"].map(normalizar_texto)
    for destino, candidatos in metric_map.items():
        col = detectar_coluna(base, candidatos)
        if col:
            base[destino] = pd.to_numeric(base[col], errors="coerce")
    metricas = [m for m in metric_map.keys() if m in base.columns]
    if not metricas:
        return pd.DataFrame()
    base = base.dropna(subset=["data_ref"]).copy()
    if base.empty:
        return pd.DataFrame()
    base["data"] = pd.to_datetime(base["data_ref"].dt.date)
    # Inconsistências físicas são excluídas antes do cálculo das médias.
    if "turbidez" in base:
        base.loc[base["turbidez"] < 0, "turbidez"] = np.nan
    if "ph" in base:
        base.loc[~base["ph"].between(0, 14), "ph"] = np.nan
    chaves = ["municipio", "municipio_norm", "data"]
    grupos = base.groupby(chaves, sort=True)
    medias = grupos[metricas].mean(numeric_only=True)
    contagens = grupos[metricas].count()
    limites = {}
    for municipio, g in base.groupby("municipio"):
        passos = g["data_ref"].drop_duplicates().sort_values().diff().dropna()
        passo = passos[passos > pd.Timedelta(0)].median()
        esperados = max(1, int(round(pd.Timedelta(days=1) / passo))) if pd.notna(passo) else 1
        limites[municipio] = max(1, int(np.ceil(0.75 * esperados)))
    minimo = pd.Series([limites[m] for m in medias.index.get_level_values("municipio")], index=medias.index)
    medias = medias.where(contagens.ge(minimo, axis=0))
    # A precipitação do CSV é média do campo original, não total em mm/dia.
    return medias.join(contagens.add_suffix("_n")).reset_index()

def preparar_base_simqua(df: pd.DataFrame, cidades: list[str]) -> pd.DataFrame:
    return _preparar_base_fonte(
        df,
        cidades,
        ["data", "datahora", "data_hora", "datetime", "timestamp"],
        ["municipio", "município", "cidade", "estacao", "estação", "ponto", "nome"],
        {
            "ph": ["ph", "potencial hidrogenionico", "potencial hidrogeniônico"],
            "turbidez": ["turbidez"],
            "precipitacao_agua": ["precipitacao", "precipitação", "chuva"],
            "nivel": ["nivel", "nível", "cota"],
            "vazao": ["vazao", "vazão"],
            "condutividade": ["condutividade", "condutividade eletrica"],
            "temperatura_agua": ["temperatura", "temperatura da agua", "temperatura da água"],
        },
        default_city_label="Área de estudo",
    )

def preparar_base_cetesb_ar(df_pm25: pd.DataFrame, cidades: list[str]) -> pd.DataFrame:
    if df_pm25.empty or "Data" not in df_pm25.columns:
        return pd.DataFrame()
    base = df_pm25.copy()
    base["data"] = pd.to_datetime(base["Data"], errors="coerce")
    if "Cidade" in base.columns:
        base["municipio"] = base["Cidade"].apply(lambda x: _mapear_municipio_estudo(x, cidades))
    else:
        base["municipio"] = "Área de estudo"
    base["municipio_norm"] = base["municipio"].map(normalizar_texto)

    mapa = {
        "pm25": "PM2.5 médio",
        "pm10": "PM10 médio",
        "no2": "NO2 médio",
        "o3": "O3 médio",
        "so2": "SO2 médio",
        "pm25_tendencia": "Tendência (MM 3d)",
    }
    for destino, origem in mapa.items():
        if origem in base.columns:
            base[destino] = pd.to_numeric(base[origem], errors="coerce")

    metricas = [c for c in ["pm25", "pm10", "no2", "o3", "so2", "pm25_tendencia"] if c in base.columns]
    base = base.dropna(subset=["data"]).copy()
    if "pm25" in base.columns:
        base = base[base["pm25"].notna()].copy()
    if base.empty or not metricas:
        return pd.DataFrame()
    return base.groupby(["municipio", "municipio_norm", "data"], as_index=False)[metricas].mean(numeric_only=True)

def preparar_base_clima_openmeteo(df_clima: pd.DataFrame, cidades: list[str]) -> pd.DataFrame:
    if df_clima.empty or "Data" not in df_clima.columns:
        return pd.DataFrame()
    base = df_clima.copy()
    base["data"] = pd.to_datetime(base["Data"], errors="coerce")
    if "Cidade" in base.columns:
        base["municipio"] = base["Cidade"].apply(lambda x: _mapear_municipio_estudo(x, cidades))
    else:
        base["municipio"] = "Área de estudo"
    base["municipio_norm"] = base["municipio"].map(normalizar_texto)

    mapa = {
        "temp_max_openmeteo": "Temp Máx",
        "temp_min_openmeteo": "Temp Mín",
        "chuva_openmeteo": "Chuva",
        "vento_max_openmeteo": "Vento Máx",
    }
    for destino, origem in mapa.items():
        if origem in base.columns:
            base[destino] = pd.to_numeric(base[origem], errors="coerce")

    metricas = [c for c in ["temp_max_openmeteo", "temp_min_openmeteo", "chuva_openmeteo", "vento_max_openmeteo"] if c in base.columns]
    base = base.dropna(subset=["data"]).copy()
    if base.empty or not metricas:
        return pd.DataFrame()
    return base.groupby(["municipio", "municipio_norm", "data"], as_index=False)[metricas].mean(numeric_only=True)

def montar_base_integrada(
    df_ar: pd.DataFrame,
    df_agua: pd.DataFrame,
    df_clima_openmeteo: pd.DataFrame,
) -> pd.DataFrame:
    if df_ar.empty or df_agua.empty:
        return pd.DataFrame()
    base = df_ar.merge(df_agua, on=["municipio", "municipio_norm", "data"], how="inner")
    if base.empty:
        ar_area = df_ar.groupby("data", as_index=False).mean(numeric_only=True)
        agua_area = df_agua.groupby("data", as_index=False).mean(numeric_only=True)
        base = ar_area.merge(agua_area, on="data", how="inner")
        if base.empty:
            return pd.DataFrame()
        base["municipio"] = "Área integrada"
        base["municipio_norm"] = "area integrada"

    if not df_clima_openmeteo.empty:
        if {"municipio", "municipio_norm"}.issubset(df_clima_openmeteo.columns):
            base = base.merge(df_clima_openmeteo, on=["municipio", "municipio_norm", "data"], how="left")
        else:
            base = base.merge(df_clima_openmeteo.groupby("data", as_index=False).mean(numeric_only=True), on="data", how="left")
    return base.sort_values(["municipio", "data"]).copy()

def _adicionar_linha_tendencia(fig: go.Figure, x: pd.Series, y: pd.Series, nome: str = "Tendência") -> go.Figure:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mascara = x_num.notna() & y_num.notna()
    if mascara.sum() < 2:
        return fig
    coef = np.polyfit(x_num[mascara], y_num[mascara], 1)
    x_line = np.linspace(x_num[mascara].min(), x_num[mascara].max(), 50)
    y_line = coef[0] * x_line + coef[1]
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=nome))
    return fig

def nome_amigavel_variavel(nome: str) -> str:
    mapa = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO₂",
        "o3": "O₃",
        "so2": "SO₂",
        "pm25_tendencia": "PM2.5 tendência (MM 3d)",
        "ph": "pH",
        "turbidez": "Turbidez",
        "precipitacao_agua": "Precipitação do CSV (média do campo)",
        "chuva_openmeteo": "Chuva Open-Meteo",
        "nivel": "Nível",
        "vazao": "Vazão",
        "condutividade": "Condutividade",
        "temperatura_agua": "Temperatura da água",
        "temp_max_openmeteo": "Temp. máx Open-Meteo",
        "temp_min_openmeteo": "Temp. mín Open-Meteo",
        "vento_max_openmeteo": "Vento máx Open-Meteo",
    }
    return mapa.get(nome, nome.replace("_", " ").title())

def obter_variavel_precipitacao_integrada(base: pd.DataFrame) -> Optional[str]:
    for coluna in ["chuva_openmeteo", "precipitacao_agua"]:
        if coluna in base.columns and base[coluna].notna().sum() >= 3:
            return coluna
    return None

def preparar_mapa_intensidade_integrada(base: pd.DataFrame, variaveis: list[str]) -> pd.DataFrame:
    if base.empty or "municipio" not in base.columns:
        return pd.DataFrame()
    cols_validas = [col for col in variaveis if col in base.columns]
    if not cols_validas:
        return pd.DataFrame()
    agregado = base.groupby("municipio", as_index=False)[cols_validas].mean(numeric_only=True)
    latitudes, longitudes = [], []
    for municipio in agregado["municipio"]:
        lat, lon = np.nan, np.nan
        for chave, info in CIDADES.items():
            nome = chave.split(",")[0].strip()
            if normalizar_texto(nome) == normalizar_texto(municipio):
                lat, lon = info["lat"], info["lon"]
                break
        latitudes.append(lat)
        longitudes.append(lon)
    agregado["Latitude"] = latitudes
    agregado["Longitude"] = longitudes
    return agregado.dropna(subset=["Latitude", "Longitude"]).copy()

def obter_variaveis_clima_ml(base: pd.DataFrame) -> list[str]:
    candidatos = [
        "temp_max_openmeteo", "temp_min_openmeteo", "vento_max_openmeteo", "chuva_openmeteo",
    ]
    return [c for c in candidatos if c in base.columns and base[c].notna().sum() >= 5]

def ajustar_rf_avaliado(X_train, X_test, y_train, y_test):
    """Pré-processamento aprendido apenas no treino e referência explícita."""
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    usadas = [c for c in X_train if X_train[c].notna().any()]
    if not usadas or len(y_test) < 2:
        return None
    modelo = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("floresta", RandomForestRegressor(n_estimators=300, random_state=42)),
    ])
    modelo.fit(X_train[usadas], y_train)
    previsto = modelo.predict(X_test[usadas])
    referencia = DummyRegressor(strategy="mean")
    referencia.fit(X_train[usadas], y_train)
    baseline = referencia.predict(X_test[usadas])
    metricas = {
        "MAE": float(mean_absolute_error(y_test, previsto)),
        "R2": float(r2_score(y_test, previsto)),
        "MAE referência": float(mean_absolute_error(y_test, baseline)),
        "R2 referência": float(r2_score(y_test, baseline)),
        "N treino": len(y_train), "N teste": len(y_test),
    }
    return modelo, previsto, metricas, usadas

def rodar_modelagem_turbidez(base: pd.DataFrame) -> Optional[dict[str, Any]]:
    if base.empty or RandomForestRegressor is None or not {"data", "turbidez", "no2", "o3"}.issubset(base):
        return None
    trabalho = base.copy().replace([np.inf, -np.inf], np.nan)
    trabalho["data"] = pd.to_datetime(trabalho["data"], errors="coerce")
    trabalho["turbidez"] = pd.to_numeric(trabalho["turbidez"], errors="coerce")
    trabalho = trabalho.loc[trabalho["data"].notna() & trabalho["turbidez"].ge(0)].sort_values("data").reset_index(drop=True)
    datas = sorted(trabalho["data"].dt.normalize().unique())
    if len(trabalho) < 16 or len(datas) < 4:
        return None
    corte = max(1, min(len(datas)-1, int(np.floor(0.7*len(datas)))))
    treino = trabalho["data"].dt.normalize().isin(datas[:corte])
    features = ["no2", "o3"] + obter_variaveis_clima_ml(trabalho.loc[treino])
    X = trabalho[features].apply(pd.to_numeric, errors="coerce")
    if not all(X.loc[treino,c].notna().any() for c in ["no2","o3"]):
        return None
    y = trabalho["turbidez"]
    payload = ajustar_rf_avaliado(X.loc[treino], X.loc[~treino], y.loc[treino], y.loc[~treino])
    if payload is None:
        return None
    modelo, previsto, metricas, usadas = payload
    trabalho.loc[~treino,"turbidez_prevista"] = previsto
    importancias = pd.DataFrame({"Variável":[nome_amigavel_variavel(c) for c in usadas],"Importância":modelo.named_steps["floresta"].feature_importances_}).sort_values("Importância",ascending=False)
    return {"base":trabalho,"metricas":metricas,"importancias":importancias,"features":[nome_amigavel_variavel(c) for c in usadas],"avaliacao":"Separação cronológica por datas completas"}

def exibir_aba_integracao(
    df_simqua: pd.DataFrame,
    df_openmeteo_ar: pd.DataFrame,
    df_clima_openmeteo: pd.DataFrame,
    area: str,
) -> None:
    st.markdown("### Integração SIMQUA + Open-Meteo Ar + clima")
    st.markdown(
        """
        Este eixo do app foi adaptado para o recorte recomendado do seu trabalho:
        - **CETESB SIMQUA** para pH, turbidez, precipitação, nível e vazão
        - **Open-Meteo Ar** para **PM2.5, PM10, NO₂, O₃ e SO₂**
        - **Open-Meteo clima** para chuva, temperatura e vento
        """
    )

    if df_simqua.empty:
        st.info(GUIDES_FONTES["SIMQUA"])
    if df_openmeteo_ar.empty:
        st.info(GUIDES_FONTES["OPENMETEO_AR"])

    if df_simqua.empty or df_openmeteo_ar.empty:
        st.warning("Para os gráficos integrados, o app precisa do SIMQUA e do histórico automático do Open-Meteo Ar.")
        return

    base = montar_base_integrada(df_openmeteo_ar, df_simqua, df_clima_openmeteo)
    if base.empty:
        st.warning("As bases foram carregadas, mas não houve cruzamento suficiente por data e município.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas integradas", len(base))
    c2.metric("Municípios cruzados", base["municipio"].nunique())
    c3.metric("Período", f"{base['data'].min().strftime('%d/%m/%Y')} a {base['data'].max().strftime('%d/%m/%Y')}")

    abas = st.tabs([
        "🎯 Relações alvo",
        "🕒 Série temporal",
        "🔥 Correlação",
        "📆 Médias semanais/mensais",
        "⏭️ Atraso temporal",
        "🗺️ Mapa intensidade",
        "🧩 Matriz de dispersão",
        "🤖 ML turbidez",
        "🧾 Base integrada",
    ])

    with abas[0]:
        col1, col2 = st.columns(2)

        with col1:
            if {"pm25", "turbidez"}.issubset(base.columns) and base[["pm25", "turbidez"]].dropna().shape[0] >= 3:
                tmp = base.dropna(subset=["pm25", "turbidez"]).copy()
                fig = px.scatter(tmp, x="pm25", y="turbidez", color="municipio", hover_name="municipio", title="PM2.5 × turbidez")
                fig = _adicionar_linha_tendencia(fig, tmp["pm25"], tmp["turbidez"])
                renderizar_grafico(fig, use_container_width=True)
            else:
                st.info("Sem dados suficientes para PM2.5 × turbidez.")

            if {"o3", "ph"}.issubset(base.columns) and base[["o3", "ph"]].dropna().shape[0] >= 3:
                tmp = base.dropna(subset=["o3", "ph"]).copy()
                fig = px.scatter(tmp, x="o3", y="ph", color="municipio", hover_name="municipio", title="O₃ × pH")
                fig = _adicionar_linha_tendencia(fig, tmp["o3"], tmp["ph"])
                renderizar_grafico(fig, use_container_width=True)
            else:
                st.info("Sem dados suficientes para O₃ × pH.")

        with col2:
            if {"no2", "turbidez"}.issubset(base.columns) and base[["no2", "turbidez"]].dropna().shape[0] >= 3:
                tmp = base.dropna(subset=["no2", "turbidez"]).copy()
                fig = px.scatter(tmp, x="no2", y="turbidez", color="municipio", hover_name="municipio", title="NO₂ × turbidez")
                fig = _adicionar_linha_tendencia(fig, tmp["no2"], tmp["turbidez"])
                renderizar_grafico(fig, use_container_width=True)
            else:
                st.info("Sem dados suficientes para NO₂ × turbidez.")

            if {"pm10", "turbidez"}.issubset(base.columns) and base[["pm10", "turbidez"]].dropna().shape[0] >= 3:
                tmp = base.dropna(subset=["pm10", "turbidez"]).copy()
                fig = px.scatter(tmp, x="pm10", y="turbidez", color="municipio", hover_name="municipio", title="PM10 × turbidez")
                fig = _adicionar_linha_tendencia(fig, tmp["pm10"], tmp["turbidez"])
                renderizar_grafico(fig, use_container_width=True)
            else:
                st.info("Sem dados suficientes para PM10 × turbidez.")

    with abas[1]:
        serie_cols = [c for c in ["pm25", "pm10", "no2", "o3", "turbidez", "ph"] if c in base.columns]
        serie = base.groupby("data", as_index=False)[serie_cols].mean(numeric_only=True) if serie_cols else pd.DataFrame()
        if serie.empty:
            st.info("Sem dados suficientes para a série temporal.")
        else:
            unidades = {"pm25": "PM₂,₅ (µg/m³)", "turbidez": "Turbidez (NTU)", "no2": "NO₂ (µg/m³)", "o3": "O₃ (µg/m³)"}
            for coluna in [c for c in unidades if c in serie]:
                fig = px.line(serie, x="data", y=coluna, markers=True, title=unidades[coluna], labels={"data":"Data",coluna:unidades[coluna]})
                fig.update_traces(connectgaps=False)
                renderizar_grafico(fig, use_container_width=True)

    with abas[2]:
        heat_cols = [c for c in [
            "pm25", "pm10", "no2", "o3", "so2",
            "ph", "turbidez", "precipitacao_agua", "nivel", "vazao", "condutividade", "temperatura_agua",
                "temp_max_openmeteo", "temp_min_openmeteo", "vento_max_openmeteo", "chuva_openmeteo",
        ] if c in base.columns and base[c].notna().sum() >= 3]
        if len(heat_cols) >= 3:
            corr = base[heat_cols].corr(numeric_only=True)
            fig = px.imshow(corr.round(2), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", title="Correlações exploratórias entre ar, água e clima")
            validos = base[heat_cols].notna().astype(int)
            st.caption("Número de pares válidos por correlação")
            st.dataframe(validos.T.dot(validos), use_container_width=True)
            renderizar_grafico(fig, use_container_width=True)
        else:
            st.info("Sem variáveis suficientes para o heatmap.")

    with abas[3]:
        chuva_col = obter_variavel_precipitacao_integrada(base)
        media_cols = [c for c in ["pm25", "turbidez"] if c in base.columns]
        if chuva_col:
            media_cols.append(chuva_col)
        serie = base.groupby("data", as_index=False)[media_cols].mean(numeric_only=True) if media_cols else pd.DataFrame()
        if len(media_cols) < 2 or serie.empty:
            st.info("Sem dados suficientes para o gráfico de médias agregadas.")
        else:
            periodicidade = st.radio("Agregação temporal", ["Semanal", "Mensal"], horizontal=True, key=f"periodicidade_{normalizar_texto(area)}")
            regra = "W" if periodicidade == "Semanal" else "MS"
            agregado = serie.set_index("data")[media_cols].resample(regra).mean().reset_index()
            agregado_melt = agregado.melt(id_vars="data", var_name="variavel", value_name="valor")
            agregado_melt["variavel"] = agregado_melt["variavel"].map(nome_amigavel_variavel)
            fig = px.line(agregado_melt, x="data", y="valor", color="variavel", markers=True, title=f"Médias {periodicidade.lower()} de PM2.5, turbidez e precipitação")
            renderizar_grafico(fig, use_container_width=True)
            st.dataframe(agregado.rename(columns={c: nome_amigavel_variavel(c) for c in media_cols}), use_container_width=True, hide_index=True)

    with abas[4]:
        if "turbidez" not in base.columns or base["turbidez"].notna().sum() < 3:
            st.info("Sem turbidez suficiente para calcular atraso temporal.")
        else:
            base_lag = base.sort_values(["municipio", "data"]).copy()
            base_lag["turbidez_amanha"] = base_lag.groupby("municipio")["turbidez"].shift(-1)
            proxima_data = base_lag.groupby("municipio")["data"].shift(-1)
            base_lag.loc[(proxima_data-base_lag["data"]).dt.days.ne(1), "turbidez_amanha"] = np.nan
            col_lag_1, col_lag_2 = st.columns(2)
            with col_lag_1:
                if "pm25" in base_lag.columns and base_lag[["pm25", "turbidez_amanha"]].dropna().shape[0] >= 3:
                    tmp = base_lag.dropna(subset=["pm25", "turbidez_amanha"]).copy()
                    fig = px.scatter(tmp, x="pm25", y="turbidez_amanha", color="municipio", hover_name="municipio", title="PM2.5 de hoje × turbidez de amanhã")
                    fig = _adicionar_linha_tendencia(fig, tmp["pm25"], tmp["turbidez_amanha"])
                    renderizar_grafico(fig, use_container_width=True)
                else:
                    st.info("Sem dados suficientes para PM2.5 de hoje × turbidez de amanhã.")
            with col_lag_2:
                chuva_col = obter_variavel_precipitacao_integrada(base_lag)
                if chuva_col and base_lag[[chuva_col, "turbidez_amanha"]].dropna().shape[0] >= 3:
                    tmp = base_lag.dropna(subset=[chuva_col, "turbidez_amanha"]).copy()
                    fig = px.scatter(tmp, x=chuva_col, y="turbidez_amanha", color="municipio", hover_name="municipio", title=f"{nome_amigavel_variavel(chuva_col)} de hoje × turbidez de amanhã")
                    fig = _adicionar_linha_tendencia(fig, tmp[chuva_col], tmp["turbidez_amanha"])
                    renderizar_grafico(fig, use_container_width=True)
                else:
                    st.info("Sem dados suficientes para chuva de hoje × turbidez de amanhã.")

    with abas[5]:
        variaveis_mapa = [c for c in ["pm25", "pm10", "no2", "o3", "turbidez"] if c in base.columns and base[c].notna().sum() >= 3]
        mapa_base = preparar_mapa_intensidade_integrada(base, variaveis_mapa)
        if mapa_base.empty or mapa_base["municipio"].nunique() < 2:
            st.info("O mapa de intensidade fica melhor quando há mais de um município ou estação no recorte integrado.")
        else:
            escolha = st.selectbox("Variável do mapa", variaveis_mapa, format_func=nome_amigavel_variavel, key=f"mapa_var_{normalizar_texto(area)}")
            fig = px.scatter_map(
                mapa_base,
                lat="Latitude",
                lon="Longitude",
                size=escolha,
                color=escolha,
                hover_name="municipio",
                zoom=7.2,
                center={"lat": mapa_base["Latitude"].mean(), "lon": mapa_base["Longitude"].mean()},
                title=f"Mapa de intensidade — {nome_amigavel_variavel(escolha)}",
            )
            renderizar_grafico(fig, use_container_width=True, key=f"mapa_int_{normalizar_texto(area)}_{escolha}")

    with abas[6]:
        vars_matriz = [c for c in ["pm25", "pm10", "no2", "o3", "turbidez", "ph", "precipitacao_agua", "condutividade"] if c in base.columns and base[c].notna().sum() >= 3]
        if len(vars_matriz) < 3:
            st.info("Sem variáveis suficientes para a matriz de dispersão.")
        else:
            matriz = base.dropna(subset=vars_matriz).copy()
            if len(matriz) > 500:
                matriz = matriz.sample(500, random_state=42)
            labels = {col: nome_amigavel_variavel(col) for col in vars_matriz}
            fig = px.scatter_matrix(matriz, dimensions=vars_matriz, color="municipio", title="Matriz de dispersão entre poluentes do ar, água e clima", labels=labels)
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            renderizar_grafico(fig, use_container_width=True)

    with abas[7]:
        payload = rodar_modelagem_turbidez(base)
        if not payload:
            st.info("Regressão de turbidez indisponível: são necessários NO₂ e O₃ utilizáveis no treino, pelo menos 16 registros de turbidez, quatro datas distintas e duas observações de teste. Esses mínimos são operacionais.")
        else:
            if payload["metricas"]:
                st.caption("Desempenho do modelo: " + " | ".join(f"{k}={v:.3f}" for k, v in payload["metricas"].items()))
            st.caption("Variáveis usadas: " + ", ".join(payload["features"]))
            base_ml = payload["base"].dropna(subset=["turbidez_prevista"]).copy()
            if not base_ml.empty:
                fig = px.scatter(base_ml, x="turbidez", y="turbidez_prevista", color="municipio", hover_name="municipio", title="NO₂ + O₃ + clima → previsão de turbidez com ML")
                minimo = float(base_ml[["turbidez", "turbidez_prevista"]].min().min())
                maximo = float(base_ml[["turbidez", "turbidez_prevista"]].max().max())
                fig.add_shape(type="line", x0=minimo, y0=minimo, x1=maximo, y1=maximo)
                renderizar_grafico(fig, use_container_width=True)
            fig_imp = px.bar(payload["importancias"], x="Importância", y="Variável", orientation="h", title="Importância das variáveis na previsão de turbidez")
            renderizar_grafico(fig_imp, use_container_width=True)

    with abas[8]:
        st.dataframe(base.sort_values(["municipio", "data"]), use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Baixar base integrada (CSV)",
            data=base.to_csv(index=False).encode("utf-8"),
            file_name=f"base_integrada_openmeteo_ar_agua_{normalizar_texto(area)}.csv",
            mime="text/csv",
        )

def rodar_modelagem_qualidade_ar(df_atual: pd.DataFrame) -> dict[str, dict[str, Any]]:
    resultados = {}
    if df_atual.empty or RandomForestRegressor is None:
        return resultados
    features = [c for c in ["Temperatura", "Umidade", "Vento", "Chuva", "População"] if c in df_atual]
    if len(features) < 2:
        return resultados
    for alvo in ["PM2.5", "NO2"]:
        if alvo not in df_atual:
            continue
        base_ok = df_atual.copy().reset_index(drop=True)
        base_ok[alvo] = pd.to_numeric(base_ok[alvo], errors="coerce")
        base_ok = base_ok.loc[base_ok[alvo].notna() & np.isfinite(base_ok[alvo])].copy()
        if len(base_ok) < 10:
            continue
        idx_train, idx_test = train_test_split(base_ok.index, test_size=0.3, random_state=42)
        X = base_ok[features].apply(pd.to_numeric, errors="coerce")
        payload = ajustar_rf_avaliado(X.loc[idx_train],X.loc[idx_test],base_ok.loc[idx_train,alvo],base_ok.loc[idx_test,alvo])
        if payload is None:
            continue
        modelo, previsto, metricas, usadas = payload
        base_ok.loc[idx_test,"previsto_rf"] = previsto
        importancias = pd.DataFrame({"Variavel":usadas,"Importancia":modelo.named_steps["floresta"].feature_importances_}).sort_values("Importancia",ascending=False)
        resultados[alvo] = {"base":base_ok,"metricas":metricas,"importancias":importancias,"avaliacao":"Comparação exploratória entre municípios na consulta atual"}
    return resultados

def exibir_bloco_predicao_qualidade_ar(df_atual: pd.DataFrame, area: str) -> None:
    st.markdown("### Predição de qualidade do ar")
    st.markdown("Estimativa exploratória entre municípios na consulta atual. Os valores atmosféricos são fornecidos por modelos CAMS via Open-Meteo. Esta avaliação não representa previsão temporal nem validação com estações independentes.")
    resultados = rodar_modelagem_qualidade_ar(df_atual)
    if not resultados:
        st.info("São necessárias observações municipais com temperatura, umidade, vento, chuva e poluentes para estimar PM2.5/NO2.")
        return
    abas = st.tabs([f"🌫️ {alvo}" for alvo in resultados.keys()])
    for aba, alvo in zip(abas, resultados.keys()):
        payload = resultados[alvo]
        with aba:
            if payload["metricas"]:
                st.caption("Desempenho do Random Forest: " + " | ".join(f"{k}={v:.3f}" for k, v in payload["metricas"].items()))
            base = payload["base"]
            fig = px.scatter(base.dropna(subset=["previsto_rf"]), x=alvo, y="previsto_rf", hover_name="Cidade", title=f"Observado × previsto — {alvo}")
            minimo = float(base[[alvo, "previsto_rf"]].min().min())
            maximo = float(base[[alvo, "previsto_rf"]].max().max())
            fig.add_shape(type="line", x0=minimo, y0=minimo, x1=maximo, y1=maximo)
            renderizar_grafico(fig, use_container_width=True)
            fig_imp = px.bar(payload["importancias"], x="Importancia", y="Variavel", orientation="h", title=f"Importância das variáveis meteorológicas — {alvo}")
            renderizar_grafico(fig_imp, use_container_width=True)

def exibir_aba_modelagem(df_atual: pd.DataFrame, area: str) -> None:
    st.markdown("### Modelagem ambiental e aprendizado de máquina")
    st.markdown(
        """
        Este eixo do projeto ficou focado em:
        - **CETESB SIMQUA + Open-Meteo** como bases principais do TCC
        - **Open-Meteo Ar, Open-Meteo clima e IBGE** como apoio automático ao panorama regional
        - **aprendizado de máquina aplicado à qualidade do ar**
        """
    )
    exibir_bloco_predicao_qualidade_ar(df_atual, area)

with st.sidebar:
    exibir_logo_sidebar()
    st.title("EcoRadar")
    st.markdown("**Painel regional com foco em água, atmosfera, clima e modelagem ambiental.**")
    modo_visual = st.radio("Tema visual", ["Noturno", "Claro"], index=0, horizontal=True)
    st.markdown("Use a barra lateral para trocar o recorte, ajustar janelas históricas e conectar bases complementares.")
    st.markdown("---")
    opcoes_area = list(REGIOES.keys()) + list(CIDADES.keys())
    area_escolhida = st.selectbox("Selecione a área de análise:", opcoes_area)
    dias_pm25 = st.slider("Janela do histórico de PM2.5 (dias)", 7, 30, DEFAULT_PM25_DAYS)
    dias_clima = st.slider("Janela do histórico climático (dias)", 7, 30, DEFAULT_CLIMATE_DAYS)
    st.markdown("---")
    busca_textual_geral = st.text_input("Filtro textual opcional nas fontes complementares", placeholder="Ex.: pindamonhangaba, simqua, guaratinguetá")
    aplicar_filtro_regional_complementares = st.checkbox("Aplicar filtro regional nas fontes complementares", value=True)
    st.markdown("---")
    with st.expander("Uploads opcionais", expanded=False):
        up_simqua = st.file_uploader("SIMQUA — CSV/XLSX", type=["csv", "xlsx", "xls"], key="up_simqua", accept_multiple_files=True)
        up_cetesb = st.file_uploader("CETESB Balneabilidade/Qualidade da água — CSV/XLSX", type=["csv", "xlsx", "xls"], key="up_cetesb")

def main() -> None:
    aplicar_tema_interface(modo_visual)
    cidades_selecionadas = obter_cidades_area(area_escolhida)
    nome_area = area_escolhida
    nomes_area = nomes_municipios(cidades_selecionadas)

    with st.spinner("Carregando APIs automáticas, arquivos locais e uploads..."):
        df_atual, df_pm25, df_clima = montar_painel_regional(tuple(cidades_selecionadas), dias_pm25, dias_clima)

        raw_cetesb, origem_cetesb = carregar_fonte_hibrida("CETESB_BALNEABILIDADE", up_cetesb)
        raw_simqua, origem_simqua = carregar_fonte_hibrida("SIMQUA", up_simqua)

        df_cetesb, diag_cetesb = preparar_fonte(raw_cetesb, cidades_selecionadas, busca_textual_geral, aplicar_filtro_regional_complementares)
        df_simqua_bruto, diag_simqua = preparar_fonte(raw_simqua, cidades_selecionadas, busca_textual_geral, aplicar_filtro_regional_complementares)

        df_simqua = preparar_base_simqua(df_simqua_bruto, cidades_selecionadas)
        df_cetesb_ar = preparar_base_cetesb_ar(df_pm25, cidades_selecionadas)
        df_clima_openmeteo = preparar_base_clima_openmeteo(df_clima, cidades_selecionadas)

    exibir_topo_dashboard(df_atual, df_simqua, nome_area, nomes_area)
    exibir_contexto_projeto(nome_area, nomes_area)
    exibir_guia_local()


    if df_atual.empty:
        st.error("Não foi possível montar o panorama principal com as APIs automáticas.")
        st.stop()

    serie_hora = pd.to_datetime(df_atual["Hora atualização"], errors="coerce")
    if serie_hora.notna().any():
        st.caption(f"🕒 Última atualização capturada nas APIs automáticas: {serie_hora.max().strftime('%d/%m/%Y às %H:%M')} (horário local)")

    automaticas = {
        "Open-Meteo clima": not df_clima.empty,
        "Open-Meteo Ar": not df_pm25.empty,
        "IBGE população": df_atual["População"].notna().any(),
        "CETESB REST": not raw_cetesb.empty,
    }
    origens = {
        "Balneabilidade CETESB": origem_cetesb,
        "SIMQUA": origem_simqua,
        "Open-Meteo Ar": "API automática",
        "Open-Meteo clima": "API automática",
    }
    dfs = {
        "Balneabilidade CETESB": df_cetesb,
        "SIMQUA": df_simqua_bruto,
        "Open-Meteo Ar": df_pm25,
        "Open-Meteo clima": df_clima,
    }
    diags = {
        "Balneabilidade CETESB": diag_cetesb,
        "SIMQUA": diag_simqua,
        "Open-Meteo Ar": {"linhas_originais": len(df_pm25), "linhas_pos_filtro_regional": len(df_pm25), "linhas_pos_busca": len(df_pm25), "fallback_sem_filtro": False, "filtro_regional_aplicado": False},
        "Open-Meteo clima": {"linhas_originais": len(df_clima), "linhas_pos_filtro_regional": len(df_clima), "linhas_pos_busca": len(df_clima), "fallback_sem_filtro": False, "filtro_regional_aplicado": False},
    }
    exibir_status_fontes(automaticas, origens, dfs, diags)
    exibir_cards_resumo(df_atual)

    abas = st.tabs([
        "🌎 Panorama regional",
        "🏖️ Litoral Norte e balneabilidade",
        "🔬 Ar, água e meteorologia",
        "🤖 Modelagem ambiental e ML",
    ])
    with abas[0]:
        exibir_panorama_regional(df_atual, df_pm25, df_clima, nome_area)
    with abas[1]:
        exibir_aba_litoral(df_cetesb, cidades_selecionadas, nome_area)
    with abas[2]:
        exibir_aba_integracao(df_simqua, df_cetesb_ar, df_clima_openmeteo, nome_area)
    with abas[3]:
        exibir_aba_modelagem(df_atual, nome_area)

    st.divider()
    st.markdown(
        '''
        ### Arquivos locais esperados em `data/`
        - `simqua.csv` ou `simqua.xlsx`
        - `cetesb.csv` ou `balneabilidade.csv`

        O app continua funcionando com panorama automático, e a aba integrada usa Open-Meteo Ar + Open-Meteo clima junto com o SIMQUA.
        '''
    )

if __name__ == "__main__":
    main()
