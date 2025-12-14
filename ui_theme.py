import streamlit as st

def _accent_palette(name: str):
  presets = {
    "Purple/Teal": ("#21f1d2", "#06b6d4"),
    "Pink/Orange": ("#ec4899", "#f59e0b"),
    "Blue/Lime": ("#f4b429", "#84cc16"),
    "Red/Gold": ("#ef4444", "#fbbf24"),
  }
  return presets.get(name, presets["Purple/Teal"])

def apply_theme():
  accent_name = st.session_state.get("ACCENT_THEME", "Purple/Teal")
  a1, a2 = _accent_palette(accent_name)
  style_name = st.session_state.get("THEME_STYLE", "Neon")
  # Base tokens; panel styles vary by theme style
  if style_name == "Glass":
    panel_bg = "rgba(16,22,34,0.45)"
    panel_border = "1px solid rgba(255,255,255,0.12)"
    panel_shadow = "0 20px 40px rgba(0,0,0,0.35)"
    panel_backdrop = "backdrop-filter: blur(12px);"
    app_bg = "linear-gradient(180deg, #0b0f19 0%, #0e1726 100%)"
  else:
    panel_bg = "radial-gradient( circle at 10% 10%, #111827 0%, #101622 60%)"
    panel_border = "1px solid #1f2937"
    panel_shadow = "0 10px 30px rgba(0,0,0,0.35)"
    panel_backdrop = ""
    app_bg = "var(--bg)"
  css = f"""
    <style>
    :root {{
      --bg: #0b0f19;
      --panel: #101622;
      --accent: {a1};
      --accent-2: {a2};
      --text: #e5e7eb;
      --muted: #94a3b8;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
    }}
    html, body, [data-testid="stAppViewContainer"] {{
      background: {app_bg} !important;
      color: var(--text) !important;
    }}
    [data-testid="stHeader"]{{background: transparent}}
    [data-testid="stSidebar"] > div:first-child {{
      background: linear-gradient(180deg, #0b0f19 0%, #0e1526 100%);
      border-right: 1px solid #13203b;
    }}
    .themed-panel {{
      background: {panel_bg};
      border: {panel_border};
      border-radius: 18px;
      box-shadow: {panel_shadow};
      padding: 18px;
      margin-bottom: 18px;
      {panel_backdrop}
    }}
    .glow-title {{
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--text);
      text-shadow: 0 0 10px rgba(124,58,237,.35), 0 0 30px rgba(6,182,212,.25);
    }}
    .rainbow-hr {{
      height: 2px;
      border: none;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      box-shadow: 0 0 8px rgba(124,58,237,.4);
      margin: 16px 0;
    }}
    /* Streamlit button styling */
    button[kind="primary"], div.stButton button {{
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
      color: #ffffff !important;
      border: 0 !important;
      border-radius: 14px !important;
      box-shadow: 0 12px 24px rgba(124,58,237,.30);
      transition: transform .08s ease-out, box-shadow .2s ease;
    }}
    div.stButton button:hover {{
      transform: translateY(-1px) scale(1.02);
      box-shadow: 0 18px 32px rgba(6,182,212,.35);
    }}
    /* Inputs and selects */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {{
      background: #0b1220 !important;
      color: var(--text) !important;
      border: 1px solid #1f2937 !important;
    }}
    .stSlider > div > div[data-baseweb="slider"] > div {{
      background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    }}
    .stSlider > div > div[data-baseweb="slider"] > div > div {{
      background: var(--accent-2) !important;
    }}
    /* Radio and checkbox */
    .stRadio div[role="radiogroup"] > label, .stCheckbox {{
      color: var(--text) !important;
    }}
    /* Chat messages bubble hint */
    [data-testid="stChatMessage"] > div {{
      background: rgba(17,24,39,.6);
      border: 1px solid #1f2937;
      border-radius: 14px;
      padding: 8px 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(124,58,237,.18);
      border: 1px solid rgba(124,58,237,.4);
      color: var(--text);
      font-size: 12px;
      margin-left: 6px;
    }}
    code, pre {{background: #0b1220 !important; color: #d1d5db !important;}}
    .chat-bubble-user {{background: rgba(124,58,237,.15); border: 1px solid rgba(124,58,237,.4);}}
    .chat-bubble-assistant {{background: rgba(6,182,212,.12); border: 1px solid rgba(6,182,212,.4);}}
    </style>
    """
  st.markdown(css, unsafe_allow_html=True)

def hero(title: str, subtitle: str = ""):
  html = f"""
  <div style="
    margin: 10px 0 22px 0;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    border-radius: 20px;
    padding: 26px 28px;
    box-shadow: 0 25px 60px rgba(6,182,212,.15);
    position: relative;
    overflow: hidden;">
    <div style="font-size: 28px; font-weight: 800; color: #0e1726;">
      {title}
    </div>
    <div style="font-size: 14px; color: #0e1726; opacity: .85; margin-top: 6px;">
      {subtitle}
    </div>
  </div>
  """
  st.markdown(html, unsafe_allow_html=True)


def themed_panel_start():
  st.markdown('<div class="themed-panel">', unsafe_allow_html=True)


def themed_panel_end():
  st.markdown('</div>', unsafe_allow_html=True)
