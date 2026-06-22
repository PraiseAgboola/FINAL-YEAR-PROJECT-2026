import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import os

# ==========================================
# PAGE CONFIG & GLOBAL STYLES
# ==========================================
st.set_page_config(
    page_title="COGNIFLOW · Cognitive Impairment Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A1628 !important;
    border-right: 0.5px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #00B4D8 !important;
}
[data-testid="stSidebar"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.4) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Main area ── */
.main .block-container {
    max-width: 100% !important;
    padding: 0 !important;
}

/* ── Custom components ── */
.ns-topbar {
    background: #ffffff;
    border-bottom: 0.5px solid #e5e7eb;
    padding: 0 28px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.ns-breadcrumb {
    font-size: 12px;
    color: #9ca3af;
    letter-spacing: 0.01em;
}
.ns-breadcrumb b { color: #111827; font-weight: 500; }
.ns-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #065f46;
    background: #d1fae5;
    padding: 4px 12px;
    border-radius: 99px;
    font-weight: 500;
}
.ns-dot { width:6px; height:6px; border-radius:50%; background:#10b981; display:inline-block; }

.ns-metric {
    background: #ffffff;
    border: 0.5px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
}
.ns-metric-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 6px;
}
.ns-metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #111827;
    line-height: 1;
}
.ns-metric-sub { font-size: 10px; color: #9ca3af; margin-top: 4px; }

.ns-panel {
    background: #ffffff;
    border: 0.5px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
}
.ns-panel-header {
    padding: 10px 16px;
    border-bottom: 0.5px solid #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.ns-panel-title {
    font-size: 12px;
    font-weight: 500;
    color: #374151;
    letter-spacing: 0.01em;
}
.ns-panel-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #9ca3af;
}
.ns-panel-body { padding: 14px 16px; }

.ns-detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 0.5px solid #f3f4f6;
    font-size: 12px;
}
.ns-detail-row:last-child { border-bottom: none; }
.ns-detail-key { color: #6b7280; }
.ns-detail-val {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    color: #111827;
}

.ns-risk-high {
    background: #fef2f2;
    border: 0.5px solid #fecaca;
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 14px 16px;
}
.ns-risk-mod {
    background: #fffbeb;
    border: 0.5px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 16px;
}
.ns-risk-low {
    background: #f0fdf4;
    border: 0.5px solid #bbf7d0;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 14px 16px;
}
.ns-risk-title {
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 2px;
}
.ns-risk-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 36px;
    font-weight: 500;
    line-height: 1;
    margin: 8px 0 4px;
}
.ns-risk-label { font-size: 11px; color: #6b7280; }

.ns-conf-bar-track {
    height: 5px;
    background: #f3f4f6;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 8px;
}

.ns-upload-zone {
    border: 0.5px dashed #d1d5db;
    border-radius: 10px;
    padding: 24px 16px;
    text-align: center;
    background: #f9fafb;
    cursor: pointer;
}
.ns-upload-icon { font-size: 28px; margin-bottom: 8px; }
.ns-upload-text { font-size: 13px; color: #374151; margin-bottom: 4px; }
.ns-upload-hint { font-size: 11px; font-family: 'IBM Plex Mono', monospace; color: #9ca3af; }

.ns-channel-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 11px;
    color: rgba(255,255,255,0.5);
    border-bottom: 0.5px solid rgba(255,255,255,0.05);
}
.ns-channel-row:last-child { border-bottom: none; }
.ns-channel-val {
    font-family: 'IBM Plex Mono', monospace;
    color: rgba(255,255,255,0.85);
    font-size: 11px;
}
.ns-dot-ok { width:5px; height:5px; border-radius:50%; background:#52B788; display:inline-block; margin-right:6px; }
.ns-dot-warn { width:5px; height:5px; border-radius:50%; background:#F4A261; display:inline-block; margin-right:6px; }

.ns-patient-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 0;
    border-top: 0.5px solid rgba(255,255,255,0.07);
}
.ns-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: rgba(0,180,216,0.2);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 500; color: #00B4D8 !important;
}
.ns-patient-name { font-size: 12px; color: rgba(255,255,255,0.8) !important; }
.ns-patient-id { font-size: 10px; font-family: 'IBM Plex Mono', monospace; color: rgba(255,255,255,0.3) !important; }

.ns-section-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.3);
    margin: 0 0 10px;
}

/* Streamlit button override */
.stButton > button {
    background: #0A1628 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #152236 !important;
}

/* Page background */
.stApp { background: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneck   = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_small   = nn.Conv1d(out_channels, out_channels, kernel_size=3,  padding=1,  bias=False)
        self.conv_medium  = nn.Conv1d(out_channels, out_channels, kernel_size=7,  padding=3,  bias=False)
        self.conv_large   = nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5,  bias=False)
        self.maxpool      = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool    = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn           = nn.BatchNorm1d(out_channels * 4)
        self.activation   = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([self.conv_small(b), self.conv_medium(b), self.conv_large(b),
                         self.conv_pool(self.maxpool(x))], dim=1)
        return self.activation(self.bn(out))


class MultiModalCognitiveClassifier(nn.Module):
    def __init__(self, signal_channels=2, num_metadata_features=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            InceptionBlock1D(signal_channels, 16),
            nn.MaxPool1d(4),
            InceptionBlock1D(64, 32),
            nn.MaxPool1d(4),
            InceptionBlock1D(128, 64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 16, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2),
        )

    def forward(self, signals, metadata):
        if signals.dim() == 2:  signals  = signals.unsqueeze(0)
        if metadata.dim() == 1: metadata = metadata.unsqueeze(0)
        sf = self.feature_extractor(signals).squeeze(-1)
        mf = self.metadata_mlp(metadata)
        return self.classifier(torch.cat([sf, mf], dim=1))


@st.cache_resource
def load_model():
    model = MultiModalCognitiveClassifier()
    model.load_state_dict(torch.load('best_model_fold_3.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 4px 14px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span style="font-size:18px;">🧠</span>
            <span style="font-size:14px;font-weight:500;color:#ffffff;">NeuroScreen</span>
        </div>
        <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:rgba(255,255,255,0.3);margin-left:26px;">
            Cognitive Risk Platform
        </div>
    </div>
    <hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.07);margin:0 0 16px;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="ns-section-label">Patient demographics</div>', unsafe_allow_html=True)
    age       = st.slider("Age", 20, 100, 65)
    sex_input = st.radio("Sex", ["Male", "Female"], horizontal=True)
    bmi       = st.slider("BMI", 10.0, 50.0, 25.0, step=0.5)

    st.markdown('<hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.07);margin:16px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="ns-section-label">Channel status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ns-channel-row"><span><span class="ns-dot-ok"></span>EEG signal</span><span class="ns-channel-val">100 Hz</span></div>
    <div class="ns-channel-row"><span><span class="ns-dot-ok"></span>ECG signal</span><span class="ns-channel-val">100 Hz</span></div>
    <div class="ns-channel-row"><span><span class="ns-dot-ok"></span>Samples</span><span class="ns-channel-val">120,000</span></div>
    <div class="ns-channel-row"><span><span class="ns-dot-warn"></span>Model fold</span><span class="ns-channel-val">Fold 3</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:0.5px solid rgba(255,255,255,0.07);margin:16px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div class="ns-patient-chip">
        <div class="ns-avatar">JA</div>
        <div>
            <div class="ns-patient-name">James Adebayo</div>
            <div class="ns-patient-id">PID-00472</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


sex_val         = 1.0 if sex_input == "Male" else 0.0
metadata_tensor = torch.tensor([[age, sex_val, bmi]], dtype=torch.float32)


# ==========================================
# MAIN CONTENT
# ==========================================

# Top bar
st.markdown("""
<div class="ns-topbar">
    <div class="ns-breadcrumb">Screening / <b>Cognitive Impairment</b></div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span class="ns-status"><span class="ns-dot"></span>Model ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='padding:20px 28px;'>", unsafe_allow_html=True)

# ── Metric row ──
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    <div class="ns-metric">
        <div class="ns-metric-label">Inference model</div>
        <div class="ns-metric-value" style="font-size:15px;margin-top:4px;">1D-Inception</div>
        <div class="ns-metric-sub">Late fusion architecture</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown("""
    <div class="ns-metric">
        <div class="ns-metric-label">Metadata features</div>
        <div class="ns-metric-value">3</div>
        <div class="ns-metric-sub">Age · Sex · BMI</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="ns-metric">
        <div class="ns-metric-label">Signal length</div>
        <div class="ns-metric-value">120<span style="font-size:14px">k</span></div>
        <div class="ns-metric-sub">Samples at 100 Hz</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Upload + Run ──
st.markdown("""
<div class="ns-panel">
    <div class="ns-panel-header">
        <span class="ns-panel-title">📂 &nbsp;Upload polysomnography</span>
        <span class="ns-panel-tag">.EDF · EEG + ECG channels required</span>
    </div>
</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['edf'], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="ns-upload-zone">
        <div class="ns-upload-icon">⬆️</div>
        <div class="ns-upload-text">Drop .edf file here or use the uploader above</div>
        <div class="ns-upload-hint">Expects EEG Fpz-Cz and ECG channels after standard PSG recording</div>
    </div>""", unsafe_allow_html=True)

if uploaded_file is not None:
    run_col, _ = st.columns([1, 2])
    with run_col:
        run = st.button("▶  Run neural inference")

    if run:
        with st.spinner("Extracting waveforms · running Inception backbone · fusing metadata…"):

            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                matched = [ch for ch in raw.ch_names if any(t in ch.upper() for t in ['EEG', 'ECG'])]
                eeg_ch  = [c for c in matched if 'EEG' in c.upper()][0]
                ecg_ch  = [c for c in matched if 'ECG' in c.upper()][0]
                raw.pick_channels([eeg_ch, ecg_ch], verbose=False)
                raw.resample(100, verbose=False)
                signals = raw.get_data()

                max_len = 120_000
                if signals.shape[1] >= max_len:
                    signals = signals[:, :max_len]
                else:
                    signals = np.pad(signals, ((0,0),(0, max_len - signals.shape[1])))

                signals_tensor = torch.tensor(signals, dtype=torch.float32)

                # ── Inference ──
                with torch.no_grad():
                    logits = model(signals_tensor, metadata_tensor)
                    probs  = torch.softmax(logits, dim=1)
                    risk   = probs[0][1].item() * 100
                    normal = probs[0][0].item() * 100

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # ── Result risk badge ──
                if risk > 50:
                    risk_cls   = "ns-risk-high"
                    risk_color = "#dc2626"
                    verdict    = "High risk of cognitive impairment"
                    badge_bg   = "#fef2f2"; badge_color = "#991b1b"
                elif risk > 30:
                    risk_cls   = "ns-risk-mod"
                    risk_color = "#d97706"
                    verdict    = "Moderate risk — further evaluation recommended"
                    badge_bg   = "#fffbeb"; badge_color = "#92400e"
                else:
                    risk_cls   = "ns-risk-low"
                    risk_color = "#16a34a"
                    verdict    = "Low risk of cognitive impairment"
                    badge_bg   = "#f0fdf4"; badge_color = "#166534"

                bar_w = int(risk)

                r_col, d_col = st.columns([1, 1])

                with r_col:
                    st.markdown(f"""
                    <div class="{risk_cls}">
                        <div class="ns-risk-title" style="color:{risk_color};">{verdict}</div>
                        <div class="ns-risk-score" style="color:{risk_color};">{risk:.1f}<span style="font-size:18px;">%</span></div>
                        <div class="ns-risk-label">Impairment probability</div>
                        <div class="ns-conf-bar-track">
                            <div style="height:5px;width:{bar_w}%;background:{risk_color};border-radius:99px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with d_col:
                    st.markdown(f"""
                    <div class="ns-panel">
                        <div class="ns-panel-header">
                            <span class="ns-panel-title">🔬 &nbsp;Diagnostic breakdown</span>
                        </div>
                        <div class="ns-panel-body">
                            <div class="ns-detail-row">
                                <span class="ns-detail-key">Impairment probability</span>
                                <span class="ns-detail-val" style="color:{risk_color};">{risk:.1f}%</span>
                            </div>
                            <div class="ns-detail-row">
                                <span class="ns-detail-key">Normal probability</span>
                                <span class="ns-detail-val">{normal:.1f}%</span>
                            </div>
                            <div class="ns-detail-row">
                                <span class="ns-detail-key">Patient age</span>
                                <span class="ns-detail-val">{age} yrs</span>
                            </div>
                            <div class="ns-detail-row">
                                <span class="ns-detail-key">Sex</span>
                                <span class="ns-detail-val">{sex_input}</span>
                            </div>
                            <div class="ns-detail-row">
                                <span class="ns-detail-key">BMI</span>
                                <span class="ns-detail-val">{bmi:.1f}</span>
                            </div>
                            <div class="ns-detail-row">
                                <span class="ns-detail-key">Model checkpoint</span>
                                <span class="ns-detail-val">Fold 3 · best</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Waveform plot ──
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div class="ns-panel">
                    <div class="ns-panel-header">
                        <span class="ns-panel-title">〰️ &nbsp;Sleep waveform — first 10 seconds</span>
                        <span class="ns-panel-tag">100 Hz · 1,000 samples</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                fig, axes = plt.subplots(2, 1, figsize=(11, 3.6))
                fig.patch.set_facecolor('#ffffff')

                t = np.linspace(0, 10, 1000)
                colors  = ['#378ADD', '#E24B4A']
                labels  = [f'Cortical EEG  ·  {eeg_ch}', f'Cardiac ECG  ·  {ecg_ch}']
                channel = [signals[0, :1000], signals[1, :1000]]

                for i, ax in enumerate(axes):
                    ax.set_facecolor('#f9fafb')
                    ax.plot(t, channel[i], color=colors[i], linewidth=0.7, alpha=0.9)
                    ax.set_xlim(0, 10)
                    ax.set_ylabel(labels[i], fontsize=8, color='#6b7280',
                                  fontfamily='monospace', labelpad=6)
                    ax.tick_params(labelsize=8, colors='#9ca3af')
                    for spine in ax.spines.values():
                        spine.set_color('#e5e7eb')
                        spine.set_linewidth(0.5)
                    ax.grid(axis='x', color='#f3f4f6', linewidth=0.5)

                axes[1].set_xlabel('Time (seconds)', fontsize=8, color='#9ca3af')
                plt.tight_layout(h_pad=0.8)
                plt.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.12)
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.markdown(f"""
                <div style="background:#fef2f2;border:0.5px solid #fecaca;border-left:3px solid #ef4444;
                            border-radius:8px;padding:14px 16px;font-size:13px;color:#7f1d1d;">
                    <b>Processing error</b><br>
                    Ensure the EDF file contains EEG and ECG channels.<br>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#991b1b;">{e}</span>
                </div>
                """, unsafe_allow_html=True)

            finally:
                os.remove(tmp_path)

elif uploaded_file is None:
    st.markdown("""
    <div style="height:16px"></div>
    <div style="background:#f1f5f9;border:0.5px solid #e2e8f0;border-radius:10px;
                padding:28px;text-align:center;color:#94a3b8;font-size:13px;">
        Upload a .edf polysomnography file above to begin screening
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="padding:12px 28px;border-top:0.5px solid #e5e7eb;margin-top:8px;
            display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:10px;font-family:'IBM Plex Mono',monospace;color:#9ca3af;">
        NeuroScreen v1.0 · 1D-Inception Late Fusion · For research use only
    </span>
    <span style="font-size:10px;color:#9ca3af;">Not a clinical diagnostic tool</span>
</div>
""", unsafe_allow_html=True)
