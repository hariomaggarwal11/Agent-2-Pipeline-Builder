"""Custom CSS styles for NeuroPipeline - extends Agent 1 dark scientific theme."""


def get_custom_css():
    """Return the custom CSS for the NeuroPipeline theme.

    Extends Agent 1's base dark theme with pipeline-specific variables
    and styles for node cards, code panels, and status indicators.
    Uses Space Grotesk for headers and JetBrains Mono for code.
    """
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    /* Base theme from Agent 1 */
    --bg-primary:    #0a0e1a;
    --bg-surface:    #111827;
    --bg-elevated:   #1a2235;
    --accent-eeg:    #00d4ff;
    --accent-ecg:    #ff4d6d;
    --accent-ok:     #22c55e;
    --accent-warn:   #f59e0b;
    --accent-fail:   #ef4444;
    --text-primary:  #f1f5f9;
    --text-muted:    #64748b;
    --border:        #1e2d45;
    --font-display:  'Space Grotesk', system-ui, sans-serif;
    --font-body:     'Inter', system-ui, sans-serif;
    --font-code:     'JetBrains Mono', 'Courier New', monospace;

    /* Pipeline-specific variables */
    --pipeline-node-bg:      #141e30;
    --pipeline-node-border:  #1e3a5f;
    --pipeline-node-active:  #00d4ff22;
    --pipeline-edge-color:   #2563eb;
    --pipeline-done:         #22c55e;
    --pipeline-running:      #f59e0b;
    --pipeline-pending:      #334155;
    --code-bg:               #0d1117;
    --code-border:           #21262d;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main app styling */
.stApp {
    background-color: var(--bg-primary);
    font-family: var(--font-body);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface);
    border-right: 1px solid var(--border);
    width: 280px;
}

[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary);
}

/* Headers - use Space Grotesk */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-code);
    color: var(--accent-eeg);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: var(--bg-surface);
    border-radius: 8px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: var(--text-muted);
    font-family: var(--font-body);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background-color: var(--bg-elevated) !important;
    color: var(--accent-eeg) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background-color: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: var(--font-body);
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button:hover {
    background-color: var(--accent-eeg);
    color: var(--bg-primary);
    border-color: var(--accent-eeg);
}

.stButton > button[kind="primary"] {
    background-color: var(--accent-eeg);
    color: var(--bg-primary);
    font-weight: 600;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 20px;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
}

/* Info/Warning/Error boxes */
.stAlert {
    border-radius: 8px;
}

/* Radio buttons and selectbox */
.stRadio > label, .stSelectbox > label {
    color: var(--text-primary) !important;
    font-family: var(--font-body);
}

/* Pipeline node cards */
.pipeline-node {
    background-color: var(--pipeline-node-bg);
    border: 1px solid var(--pipeline-node-border);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: var(--font-body);
}

.pipeline-node:hover {
    border-color: var(--accent-eeg);
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.15);
}

.pipeline-node.active {
    background-color: var(--pipeline-node-active);
    border-color: var(--accent-eeg);
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.2);
}

.pipeline-node .node-number {
    font-family: var(--font-code);
    font-size: 0.75em;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.pipeline-node .node-name {
    font-family: var(--font-display);
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.9em;
}

.pipeline-node .node-status {
    margin-top: 6px;
    font-size: 0.8em;
}

/* Status badges */
.status-done {
    background-color: rgba(34, 197, 94, 0.12);
    color: var(--pipeline-done);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    display: inline-block;
}

.status-running {
    background-color: rgba(245, 158, 11, 0.12);
    color: var(--pipeline-running);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    display: inline-block;
}

.status-failed {
    background-color: rgba(239, 68, 68, 0.12);
    color: var(--accent-fail);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    display: inline-block;
}

.status-pending {
    background-color: rgba(51, 65, 85, 0.3);
    color: var(--pipeline-pending);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    display: inline-block;
}

.status-skipped {
    background-color: rgba(100, 116, 139, 0.12);
    color: var(--text-muted);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    display: inline-block;
}

/* Code preview panel */
.code-panel {
    background-color: var(--code-bg);
    border: 1px solid var(--code-border);
    border-radius: 8px;
    padding: 16px;
    font-family: var(--font-code);
    font-size: 0.85em;
    overflow-x: auto;
}

/* Stage connection arrow */
.stage-arrow {
    text-align: center;
    color: var(--pipeline-edge-color);
    font-size: 1.2em;
    padding: 2px 0;
    line-height: 1;
}

/* Metric card */
.metric-card {
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.metric-card .metric-label {
    font-family: var(--font-body);
    color: var(--text-muted);
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.metric-card .metric-value {
    font-family: var(--font-code);
    color: var(--accent-eeg);
    font-size: 1.6em;
    font-weight: 600;
}

.metric-card .metric-delta {
    font-family: var(--font-body);
    font-size: 0.8em;
    margin-top: 4px;
}

.metric-card .metric-delta.positive {
    color: var(--accent-ok);
}

.metric-card .metric-delta.negative {
    color: var(--accent-fail);
}

/* Progress bar */
.progress-container {
    background-color: var(--pipeline-pending);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin: 8px 0;
}

.progress-fill {
    background: linear-gradient(90deg, var(--accent-eeg), var(--pipeline-edge-color));
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}

/* Neuro card (general purpose) */
.neuro-card {
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

/* Landing page upload area */
.upload-area {
    background-color: var(--bg-surface);
    border: 2px dashed var(--pipeline-node-border);
    border-radius: 12px;
    padding: 40px 20px;
    text-align: center;
    transition: border-color 0.2s;
}

.upload-area:hover {
    border-color: var(--accent-eeg);
}

/* Dataset summary card */
.dataset-summary {
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}

.dataset-summary .summary-title {
    font-family: var(--font-display);
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1.1em;
    margin-bottom: 12px;
}

.dataset-summary .summary-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9em;
}

.dataset-summary .summary-row:last-child {
    border-bottom: none;
}

.dataset-summary .summary-key {
    color: var(--text-muted);
}

.dataset-summary .summary-value {
    color: var(--text-primary);
    font-family: var(--font-code);
}
</style>
"""
