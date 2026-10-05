import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from streamlit.errors import StreamlitSecretNotFoundError

# Import modules
from modules.input_module import DataLoader
from modules.data_visualization_module import Visualizer
from modules.data_analysis_module import StatisticalAnalyzer
from modules.processing_module import MLAnalyzer, GeminiAnalyzer

# Set page configuration
st.set_page_config(
    page_title="CSV Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the authentication screen styled too; it renders before the analyzer UI.
st.markdown(
    """
<style>
    .stApp { background: #eaf8f7; color: #31565b; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: max(5vh, 32px); }
    .login-layout { display: grid; grid-template-columns: 1.2fr .8fr; min-height: 590px; overflow: hidden; border-radius: 24px; background: #fafffe; box-shadow: 0 24px 65px rgba(53,133,132,.16); border: 1px solid rgba(255,255,255,.8); }
    .login-art { position: relative; overflow: hidden; min-height: 590px; padding: 48px; display: flex; flex-direction: column; justify-content: space-between; background: linear-gradient(145deg,#baf2e9 0%,#8be0d9 55%,#76d1d4 100%); }
    .login-art:before { content:''; position:absolute; width:780px; height:240px; left:-80px; top:155px; border-radius:50%; border:2px solid rgba(42,135,142,.2); transform:rotate(14deg); box-shadow:0 22px 0 rgba(255,255,255,.23),0 45px 0 rgba(50,157,158,.12); }
    .login-art:after { content:''; position:absolute; width:420px;height:420px;border-radius:50%;right:-170px;top:-230px;background:rgba(255,255,255,.2); }
    .login-brand,.login-art-copy,.login-chart { position:relative; z-index:1; }
    .login-brand { display:flex; align-items:center; gap:11px; color:#267c80; font-size:18px; font-weight:800; letter-spacing:-.4px; }
    .login-brand-mark { display:grid;place-items:center;width:40px;height:40px;border-radius:13px;background:rgba(255,255,255,.62);font-size:19px; }
    .login-chart { width:min(100%,430px); margin:8px auto; padding:19px; border:1px solid rgba(255,255,255,.72); border-radius:18px; background:rgba(255,255,255,.58); box-shadow:0 20px 40px rgba(45,123,129,.13); transform:rotate(-2deg); }
    .login-chart-top { display:flex; gap:6px; margin-bottom:18px; }.login-chart-top i { width:7px;height:7px;border-radius:50%;background:#fff; }
    .login-chart-grid { height:135px;display:flex;align-items:end;gap:10px;padding:13px 15px 0;border-radius:11px;background:rgba(255,255,255,.38); }
    .login-chart-grid i { flex:1;display:block;border-radius:6px 6px 0 0;background:#5fc4b9; }.login-chart-grid i:nth-child(1){height:34%;background:#f4c9a9}.login-chart-grid i:nth-child(2){height:64%;background:#7bcfc0}.login-chart-grid i:nth-child(3){height:48%;background:#b2a7ed}.login-chart-grid i:nth-child(4){height:83%;background:#58bdb3}.login-chart-grid i:nth-child(5){height:59%;background:#f3bfae}.login-chart-grid i:nth-child(6){height:94%;background:#79cdbf}
    .login-art-copy h2 { max-width:430px;margin:0 0 11px;color:#246b70;font-size:30px;font-weight:800;letter-spacing:-1px; }.login-art-copy p { max-width:390px;margin:0;color:#347e81;font-size:14px;line-height:1.7; }
    div[data-testid="column"]:has(.login-right) { min-height:590px; background:#fbfefd; border:1px solid #fff; border-radius:0 24px 24px 0; }
    div[data-testid="column"]:has(.login-right) > div > div[data-testid="stVerticalBlock"] { min-height:590px; justify-content:center; }
    div[data-testid="column"]:has(.login-right) div[data-testid="stButton"] { padding:0 48px 42px; }
    .login-right { padding:0 48px 16px;background:transparent; }
    .login-kicker { color:#55aaa3;font-size:10px;font-weight:800;letter-spacing:1.7px; }
    .login-right h1 { margin:12px 0 10px;color:#31565b;font-size:30px;font-weight:800;letter-spacing:-1px; }
    .login-right p { margin:0 0 25px;color:#81999a;font-size:14px;line-height:1.65; }
    div[data-testid="stButton"] button { width:100%; min-height:48px;border:0;border-radius:11px;background:#58bdb3;color:#fff;font-size:14px;font-weight:700;box-shadow:0 8px 18px rgba(74,178,168,.2);transition:transform .18s,background .18s; }
    div[data-testid="stButton"] button:hover { background:#43aaa2;color:#fff;transform:translateY(-1px); }
    @media(max-width:760px){[data-testid="stMainBlockContainer"]{padding:20px 14px}.login-layout{grid-template-columns:1fr}.login-art{min-height:380px;padding:28px}.login-art-copy h2{font-size:25px}.login-chart-grid{height:90px}div[data-testid="column"]:has(.login-right){min-height:300px;border-radius:0 0 24px 24px}div[data-testid="column"]:has(.login-right)>div>div[data-testid="stVerticalBlock"]{min-height:300px}.login-right{padding:34px 28px 12px}div[data-testid="column"]:has(.login-right) div[data-testid="stButton"]{padding:0 28px 34px}}
</style>
""",
    unsafe_allow_html=True,
)


def show_table(frame):
    """Render tabular results as HTML without Streamlit's PyArrow serializer."""
    if frame is None:
        return
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    html = frame.to_html(index=False, escape=True, border=0, classes="results-table")
    st.markdown(f'<div class="results-table-wrap">{html}</div>', unsafe_allow_html=True)

# Require Google sign-in before rendering any analyzer functionality.
try:
    auth_config = st.secrets.get("auth")
except StreamlitSecretNotFoundError:
    auth_config = None
if not auth_config:
    st.markdown(
        '<div class="login-layout"><section class="login-art"><div class="login-brand"><span class="login-brand-mark">▥</span> Data Studio</div><div class="login-chart"><div class="login-chart-top"><i></i><i></i><i></i></div><div class="login-chart-grid"><i></i><i></i><i></i><i></i><i></i><i></i></div></div><div class="login-art-copy"><h2>Good data starts with a clear view.</h2><p>Explore your CSV files, spot patterns, and turn numbers into useful insights.</p></div></section><section class="login-right"><div class="login-kicker">CSV DATA ANALYZER</div><h1>Almost there!</h1><p>Sign-in needs a quick setup before you can open your workspace.</p><p><b>Google sign-in isn’t configured yet.</b><br>Follow the setup steps in the project README, then refresh this page.</p></section></div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not st.user.is_logged_in:
    left, right = st.columns([1.2, .8], gap="small")
    with left:
        st.markdown(
            '<section class="login-art"><div class="login-brand"><span class="login-brand-mark">▥</span> Data Studio</div><div class="login-chart"><div class="login-chart-top"><i></i><i></i><i></i></div><div class="login-chart-grid"><i></i><i></i><i></i><i></i><i></i><i></i></div></div><div class="login-art-copy"><h2>Good data starts with a clear view.</h2><p>Explore your CSV files, spot patterns, and turn numbers into useful insights.</p></div></section>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<section class="login-right"><div class="login-kicker">CSV DATA ANALYZER</div><h1>Welcome back</h1><p>Sign in to open your workspace and pick up where your analysis begins.</p></section>',
            unsafe_allow_html=True,
        )
        st.button("Continue with Google", on_click=st.login, type="primary")
    st.stop()

# Add custom CSS
st.markdown(
    """
<style>
    :root { --ink: #182333; --muted: #748094; --line: #e7ebf1; --accent: #6558d3; }
    .stApp { background: #f7f9ff; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,249,255,.92); }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e9edfa; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #293554; font-weight: 700; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] { padding: 10px; border: 1px dashed #c9c5f5; border-radius: 13px; background: #faf9ff; }
    [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; max-width: 1500px; }
    .main-header {
        font: 800 2rem/1.2 'Segoe UI', sans-serif;
        letter-spacing: -1px;
        color: #293554;
        text-align: left;
        margin: 0 0 1.4rem;
    }
    .sub-header {
        font: 700 1.35rem/1.3 'Segoe UI', sans-serif;
        color: #263449;
        margin: 1.6rem 0 .9rem;
    }
    .info-text {
        background: #f1f0ff;
        border: 1px solid #e6e3ff;
        color: #4e5870;
        padding: 1rem 1.15rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e8ebf4; border-radius: 14px; padding: 16px 18px; box-shadow: 0 7px 20px rgba(84,101,152,.055); }
    [data-testid="stMetricLabel"] { color: #788397; font-size: .78rem; }
    [data-testid="stMetricValue"] { color: #253247; font-weight: 700; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 7px; border-bottom: 1px solid #e8ebf4; }
    [data-testid="stTabs"] button[role="tab"] { color: #788397; font-weight: 600; border-radius: 12px 12px 0 0; }
    [data-testid="stTabs"] button[role="tab"]:hover { color: var(--accent); }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--accent); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    .results-table-wrap { width:100%; overflow-x:auto; border:1px solid #e8ebf4; border-radius:12px; background:#fff; margin:.35rem 0 1.1rem; }
    table.results-table { width:100%; border-collapse:collapse; color:#465166; font-size:13px; }
    table.results-table th { padding:11px 13px; text-align:left; white-space:nowrap; background:#f5f7ff; color:#69758b; font-size:10px; letter-spacing:.6px; text-transform:uppercase; border-bottom:1px solid #e8ebf4; }
    table.results-table td { padding:10px 13px; border-bottom:1px solid #eef0f6; }
    table.results-table tr:last-child td { border-bottom:0; }
    table.results-table tbody tr:hover { background:#fafaff; }
    .stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 650; border-color: #dedbff; }
    .stButton button[kind="primary"], .stDownloadButton button[kind="primary"] { background: #7064e8; border-color: #7064e8; color: #fff; }
    .welcome-card { display: grid; grid-template-columns: 1.05fr .95fr; gap: 34px; align-items: center; padding: clamp(28px,5vw,58px); margin: 3vh 0 1rem; border: 1px solid #e7e8fb; border-radius: 22px; background: linear-gradient(118deg,#fff 0%,#fff 53%,#f1efff 100%); box-shadow: 0 18px 50px rgba(84,101,152,.08); }
    .welcome-eyebrow { color: var(--accent); font-size: 10px; letter-spacing: 1.7px; font-weight: 700; }
    .welcome-card h1 { font: 800 clamp(34px,4.4vw,54px)/1.08 'Segoe UI',sans-serif; letter-spacing: -2px; margin: 18px 0 14px; color: #293554; }
    .welcome-card h1 span { color: var(--accent); }
    .welcome-card p { max-width: 420px; color: #737f91; font-size: 15px; line-height: 1.7; }
    .welcome-hint { margin-top: 24px; color: #778195; font-size: 12px; }
    .welcome-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .welcome-tile { min-height: 126px; padding: 18px; border: 1px solid #e4e8f4; border-radius: 15px; background: rgba(255,255,255,.9); color: #303b4d; font-weight: 700; font-size: 13px; box-shadow: 0 6px 18px rgba(84,101,152,.045); }
    .welcome-sample { height: 186px; margin: 20px auto 25px; max-width: 410px; padding: 16px; border: 1px solid #e1dfff; border-radius: 15px; background: rgba(255,255,255,.82); box-shadow: 0 14px 34px rgba(84,75,180,.10); transform: rotate(-2deg); }
    .sample-bar { height: 9px; width: 40%; border-radius: 6px; background: #d9d5ff; margin: 0 0 12px 2px; }
    .sample-chart { height: 126px; display: flex; align-items: end; gap: 9px; padding: 12px 15px 0; border-radius: 10px; background: linear-gradient(#fbfaff,#f6f5ff); }
    .sample-chart i { display: block; flex: 1; border-radius: 6px 6px 0 0; background: linear-gradient(180deg,#aaa2ff,#7569e9); }
    .sample-chart i:nth-child(1) { height: 36%; background: #b8e6d2; } .sample-chart i:nth-child(2) { height: 63%; } .sample-chart i:nth-child(3) { height: 48%; background: #ffc8b9; } .sample-chart i:nth-child(4) { height: 81%; } .sample-chart i:nth-child(5) { height: 57%; background: #93d9c1; } .sample-chart i:nth-child(6) { height: 92%; }
    .welcome-tile span { display: block; color: #8a94a4; font-size: 11px; font-weight: 400; margin-top: 5px; }
    @media (max-width: 760px) { .welcome-card { grid-template-columns: 1fr; gap: 22px; padding: 25px; margin-top: 1vh; } .welcome-card h1 { font-size: 38px; } }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize modules
data_loader = DataLoader()
visualizer = Visualizer(data_loader)
statistical_analyzer = StatisticalAnalyzer(data_loader)
ml_analyzer = MLAnalyzer(data_loader)
gemini_analyzer = GeminiAnalyzer(data_loader)

# Main header
st.markdown('<div class="welcome-eyebrow">YOUR DATA WORKSPACE</div><h1 class="main-header">CSV File Analyser</h1>', unsafe_allow_html=True)

# Sidebar for file upload and settings
with st.sidebar:
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:0 0 18px"><span style="display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#f0efff;color:#7064e8;font-size:19px">▥</span><span style="font-size:15px;font-weight:800;color:#293554">Data Studio</span></div>', unsafe_allow_html=True)
    st.caption(f"Signed in as {st.user.get('email', st.user.get('name', 'Google user'))}")
    st.button("Sign out", on_click=st.logout)
    st.markdown("---")
    st.header("Upload & Settings")

    # File upload
    uploaded_file = data_loader.create_upload_widget()

    if uploaded_file is not None:
        success = data_loader.load_data(uploaded_file)
        if success:
            st.success(f"Successfully loaded: {data_loader.get_filename()}")

    # Gemini API key input
    st.markdown("---")
    st.subheader("Gemini AI Integration")
    gemini_configured = gemini_analyzer.create_api_key_input()

# Main content area
if data_loader.get_data() is not None:
    # Display tabs for different analysis options
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "📈 Visualizations",
            "🧮 Statistical Analysis",
            "🤖 ML Insights",
            "🧠 Gemini AI Analysis",
        ]
    )

    # Tab 1: Overview
    with tab1:
        st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)

        # Get data info
        data_info = data_loader.get_data_info()

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filename:** {data_info['filename']}")
            st.write(f"**Rows:** {data_info['rows']}")
            st.write(f"**Columns:** {data_info['columns']}")

        with col2:
            st.write("**Memory Usage:**")
            st.write(data_info["memory_usage"])

            # Missing values summary
            st.write(f"**Missing Values:** {data_info['missing_values']}")

        st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
        show_table(data_loader.get_data().head(10))

        st.markdown(
            '<h3 class="sub-header">Column Information</h3>', unsafe_allow_html=True
        )

        # Column information
        col_info = data_loader.get_column_info()
        show_table(pd.DataFrame(col_info))

        # Download processed data
        st.markdown(
            '<h3 class="sub-header">Download Processed Data</h3>',
            unsafe_allow_html=True,
        )
        download_link = data_loader.create_download_link()
        if download_link:
            st.markdown(download_link, unsafe_allow_html=True)

    # Tab 2: Visualizations
    with tab2:
        st.markdown(
            '<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True
        )

        # Get column lists
        numeric_cols = data_loader.get_numeric_columns()
        categorical_cols = data_loader.get_categorical_columns()

        # Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            st.markdown(
                '<h3 class="sub-header">Distribution Plots</h3>', unsafe_allow_html=True
            )

            selected_num_col = st.selectbox(
                "Select a numeric column for distribution analysis:", numeric_cols
            )

            col1, col2 = st.columns(2)

            # Create distribution plots
            hist_fig, box_fig = visualizer.create_distribution_plots(selected_num_col)

            with col1:
                if hist_fig:
                    st.pyplot(hist_fig)

            with col2:
                if box_fig:
                    st.pyplot(box_fig)

            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.markdown(
                    '<h3 class="sub-header">Correlation Heatmap</h3>',
                    unsafe_allow_html=True,
                )

                corr_fig = visualizer.create_correlation_heatmap()
                if corr_fig:
                    st.pyplot(corr_fig)

        # Categorical analysis
        if len(categorical_cols) > 0:
            st.markdown(
                '<h3 class="sub-header">Categorical Analysis</h3>',
                unsafe_allow_html=True,
            )

            selected_cat_col = st.selectbox(
                "Select a categorical column:", categorical_cols
            )

            # Create categorical plot
            cat_fig = visualizer.create_categorical_plot(selected_cat_col)
            if cat_fig:
                st.pyplot(cat_fig)

        # Scatter plot for numeric columns
        if len(numeric_cols) >= 2:
            st.markdown(
                '<h3 class="sub-header">Scatter Plot</h3>', unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox("Select X-axis:", numeric_cols, index=0)

            with col2:
                remaining_cols = [col for col in numeric_cols if col != x_col]
                y_col = st.selectbox(
                    "Select Y-axis:",
                    remaining_cols,
                    index=0 if len(remaining_cols) > 0 else None,
                )

            if len(remaining_cols) > 0:
                scatter_fig = visualizer.create_scatter_plot(x_col, y_col)
                if scatter_fig:
                    st.pyplot(scatter_fig)

    # Tab 3: Statistical Analysis
    with tab3:
        st.markdown(
            '<h2 class="sub-header">Statistical Analysis</h2>', unsafe_allow_html=True
        )

        if len(numeric_cols) > 0:
            # Descriptive statistics
            st.markdown(
                '<h3 class="sub-header">Descriptive Statistics</h3>',
                unsafe_allow_html=True,
            )
            desc_stats = statistical_analyzer.get_descriptive_statistics()
            if desc_stats is not None:
                show_table(desc_stats)

            # Group by analysis
            if len(categorical_cols) > 0:
                st.markdown(
                    '<h3 class="sub-header">Group By Analysis</h3>',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)

                with col1:
                    group_col = st.selectbox(
                        "Select column to group by:", categorical_cols
                    )

                with col2:
                    agg_col = st.selectbox("Select column to aggregate:", numeric_cols)

                agg_func = st.selectbox(
                    "Select aggregation function:",
                    ["Mean", "Median", "Sum", "Min", "Max", "Count", "Std Dev"],
                )

                # Perform group by analysis
                grouped_data = statistical_analyzer.get_group_by_statistics(
                    group_col, agg_col, agg_func
                )

                if grouped_data is not None:
                    # Display results
                    st.write(f"**{agg_func} of {agg_col} grouped by {group_col}:**")
                    show_table(grouped_data)

                    # Create group by plot
                    grouped_data, group_fig = visualizer.create_group_by_plot(
                        group_col, agg_col, agg_func
                    )
                    if group_fig:
                        st.pyplot(group_fig)

            # Missing values analysis
            st.markdown(
                '<h3 class="sub-header">Missing Values Analysis</h3>',
                unsafe_allow_html=True,
            )
            missing_summary = statistical_analyzer.get_missing_values_summary()
            if missing_summary is not None and len(missing_summary) > 0:
                show_table(missing_summary)
            else:
                st.info("No missing values found in the dataset.")

            # Outlier analysis
            st.markdown(
                '<h3 class="sub-header">Outlier Analysis</h3>', unsafe_allow_html=True
            )

            outlier_method = st.radio(
                "Select outlier detection method:", ["IQR", "Z-Score"], horizontal=True
            )
            method = "iqr" if outlier_method == "IQR" else "zscore"

            outlier_summary = statistical_analyzer.get_outliers_summary(method=method)
            if outlier_summary is not None and len(outlier_summary) > 0:
                show_table(outlier_summary)
            else:
                st.info("No outliers found using the selected method.")

    # Tab 4: ML Insights
    with tab4:
        st.markdown(
            '<h2 class="sub-header">Machine Learning Insights</h2>',
            unsafe_allow_html=True,
        )

        if len(numeric_cols) >= 2:
            st.markdown(
                '<div class="info-text">This section provides machine learning insights from your data.</div>',
                unsafe_allow_html=True,
            )

            # Create tabs for different ML analyses
            ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(
                ["PCA Analysis", "Clustering", "Regression", "Classification"]
            )

            # PCA Analysis tab
            with ml_tab1:
                st.markdown(
                    '<h3 class="sub-header">Principal Component Analysis (PCA)</h3>',
                    unsafe_allow_html=True,
                )

                # Select columns for PCA
                pca_cols = st.multiselect(
                    "Select numeric columns for PCA:",
                    numeric_cols,
                    default=numeric_cols[: min(5, len(numeric_cols))],
                )

                if len(pca_cols) >= 2:
                    # Number of components
                    n_components = st.slider(
                        "Number of components:",
                        min_value=2,
                        max_value=min(len(pca_cols), 10),
                        value=min(3, len(pca_cols)),
                    )

                    if st.button("Run PCA Analysis"):
                        with st.spinner("Performing PCA analysis..."):
                            # Perform PCA
                            pca_result = ml_analyzer.perform_pca(pca_cols, n_components)

                            if pca_result is not None:
                                # Display explained variance
                                explained_variance = pca_result["explained_variance"]

                                st.write(
                                    "**Explained Variance by Principal Components:**"
                                )
                                for i, var in enumerate(explained_variance):
                                    st.write(f"PC{i+1}: {var:.2f}%")

                                # Create PCA plots
                                pca_plots = ml_analyzer.create_pca_plots(pca_result)

                                if pca_plots is not None:
                                    # Display plots
                                    if "explained_variance" in pca_plots:
                                        st.pyplot(pca_plots["explained_variance"])

                                    if "scatter" in pca_plots:
                                        st.pyplot(pca_plots["scatter"])

                                    # Feature importance
                                    st.markdown(
                                        "<h4>Feature Importance</h4>",
                                        unsafe_allow_html=True,
                                    )

                                    show_table(pca_result["loadings"])

                                    if "importance" in pca_plots:
                                        st.pyplot(pca_plots["importance"])

            # Clustering tab
            with ml_tab2:
                st.markdown(
                    '<h3 class="sub-header">K-Means Clustering</h3>',
                    unsafe_allow_html=True,
                )

                # Select columns for clustering
                cluster_cols = st.multiselect(
                    "Select numeric columns for clustering:",
                    numeric_cols,
                    default=numeric_cols[: min(3, len(numeric_cols))],
                    key="cluster_cols",
                )

                if len(cluster_cols) >= 2:
                    # Number of clusters
                    k = st.slider(
                        "Number of clusters (k):", min_value=2, max_value=10, value=3
                    )

                    if st.button("Run Clustering Analysis"):
                        with st.spinner("Performing clustering analysis..."):
                            # Perform clustering
                            clustering_result = ml_analyzer.perform_clustering(
                                cluster_cols, k
                            )

                            if clustering_result is not None:
                                # Display cluster centers
                                centers = clustering_result["centers"]

                                st.write("**Cluster Centers:**")
                                show_table(centers)

                                # Create clustering plots
                                clustering_plots = ml_analyzer.create_clustering_plots(
                                    clustering_result
                                )

                                if clustering_plots is not None:
                                    # Display plots
                                    if "scatter" in clustering_plots:
                                        st.pyplot(clustering_plots["scatter"])

                                    if "distribution" in clustering_plots:
                                        st.pyplot(clustering_plots["distribution"])

                                    if "parallel" in clustering_plots:
                                        st.pyplot(clustering_plots["parallel"])

                                # Download clustered data
                                st.markdown(
                                    "<h4>Download Clustered Data</h4>",
                                    unsafe_allow_html=True,
                                )

                                download_link = ml_analyzer.create_download_link(
                                    clustering_result["cluster_df"],
                                    "clustered_data.csv",
                                )

                                if download_link:
                                    st.markdown(download_link, unsafe_allow_html=True)

            # Regression tab
            with ml_tab3:
                st.markdown(
                    '<h3 class="sub-header">Regression Analysis</h3>',
                    unsafe_allow_html=True,
                )

                # Select target column
                target_col = st.selectbox(
                    "Select target column for regression:",
                    numeric_cols,
                    key="reg_target",
                )

                # Select feature columns
                feature_cols = st.multiselect(
                    "Select feature columns (leave empty to use all numeric columns except target):",
                    [col for col in numeric_cols if col != target_col],
                    default=[],
                    key="reg_features",
                )

                # Use all numeric columns except target if none selected
                if len(feature_cols) == 0:
                    feature_cols = None

                # Test size
                test_size = (
                    st.slider(
                        "Test size (%):", min_value=10, max_value=50, value=20, step=5
                    )
                    / 100
                )

                if st.button("Run Regression Analysis"):
                    with st.spinner("Training regression model..."):
                        # Train regression model
                        regression_result = ml_analyzer.train_regression_model(
                            target_col, feature_cols, test_size
                        )

                        if regression_result is not None:
                            # Display metrics
                            st.write("**Regression Metrics:**")
                            metrics_df = pd.DataFrame(
                                {
                                    "Metric": [
                                        "Mean Squared Error",
                                        "Root Mean Squared Error",
                                        "R² Score",
                                    ],
                                    "Value": [
                                        regression_result["mse"],
                                        regression_result["rmse"],
                                        regression_result["r2"],
                                    ],
                                }
                            )
                            show_table(metrics_df)

                            # Create regression plots
                            regression_plots = ml_analyzer.create_regression_plots(
                                regression_result
                            )

                            if regression_plots is not None:
                                # Display plots
                                col1, col2 = st.columns(2)

                                with col1:
                                    if "actual_vs_predicted" in regression_plots:
                                        st.pyplot(
                                            regression_plots["actual_vs_predicted"]
                                        )

                                with col2:
                                    if "residuals" in regression_plots:
                                        st.pyplot(regression_plots["residuals"])

                                # Feature importance
                                st.markdown(
                                    "<h4>Feature Importance</h4>",
                                    unsafe_allow_html=True,
                                )

                                show_table(regression_result["feature_importance"])

                                if "importance" in regression_plots:
                                    st.pyplot(regression_plots["importance"])

            # Classification tab
            with ml_tab4:
                st.markdown(
                    '<h3 class="sub-header">Classification Analysis</h3>',
                    unsafe_allow_html=True,
                )

                # Get categorical columns with limited unique values
                potential_targets = []
                for col in data_loader.get_data().columns:
                    if col in categorical_cols:
                        n_unique = data_loader.get_data()[col].nunique()
                        if 2 <= n_unique <= 10:  # Reasonable number of classes
                            potential_targets.append(col)
                    elif col in numeric_cols:
                        n_unique = data_loader.get_data()[col].nunique()
                        if 2 <= n_unique <= 10:  # Reasonable number of classes
                            potential_targets.append(col)

                if len(potential_targets) > 0:
                    # Select target column
                    target_col = st.selectbox(
                        "Select target column for classification:",
                        potential_targets,
                        key="class_target",
                    )

                    # Select feature columns
                    feature_cols = st.multiselect(
                        "Select feature columns (leave empty to use all numeric columns except target):",
                        [col for col in numeric_cols if col != target_col],
                        default=[],
                        key="class_features",
                    )

                    # Use all numeric columns except target if none selected
                    if len(feature_cols) == 0:
                        feature_cols = None

                    # Test size
                    test_size = (
                        st.slider(
                            "Test size (%):",
                            min_value=10,
                            max_value=50,
                            value=20,
                            step=5,
                            key="class_test_size",
                        )
                        / 100
                    )

                    if st.button("Run Classification Analysis"):
                        with st.spinner("Training classification model..."):
                            # Train classification model
                            classification_result = ml_analyzer.train_classification_model(
                                target_col, feature_cols, test_size
                            )

                            if classification_result is not None:
                                # Display metrics
                                st.write("**Classification Metrics:**")
                                st.write(
                                    f"Accuracy: {classification_result['accuracy']:.4f}"
                                )

                                # Create classification plots
                                classification_plots = ml_analyzer.create_classification_plots(
                                    classification_result
                                )

                                if classification_plots is not None:
                                    # Display plots
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        if "confusion_matrix" in classification_plots:
                                            st.pyplot(
                                                classification_plots["confusion_matrix"]
                                            )

                                    with col2:
                                        if "class_report" in classification_plots:
                                            st.pyplot(
                                                classification_plots["class_report"]
                                            )

                                    # Feature importance
                                    st.markdown(
                                        "<h4>Feature Importance</h4>",
                                        unsafe_allow_html=True,
                                    )

                                    show_table(
                                        classification_result["feature_importance"]
                                    )

                                    if "importance" in classification_plots:
                                        st.pyplot(classification_plots["importance"])
                else:
                    st.info(
                        "No suitable target columns found for classification. Target columns should have between 2 and 10 unique values."
                    )

    # Tab 5: Gemini AI Analysis
    with tab5:
        st.markdown(
            '<h2 class="sub-header">Gemini AI Analysis</h2>', unsafe_allow_html=True
        )

        if not gemini_analyzer.is_configured():
            st.warning(
                "Please configure your Gemini API key in the sidebar to use AI analysis."
            )
        else:
            st.markdown(
                '<div class="info-text">This section uses Google\'s Gemini AI to provide insights about your data.</div>',
                unsafe_allow_html=True,
            )

            # Analysis options
            analysis_type = st.selectbox(
                "Select analysis type:", gemini_analyzer.get_analysis_types()
            )

            # Custom question for custom analysis
            custom_question = None
            if analysis_type == "Custom Analysis":
                custom_question = st.text_area(
                    "Enter your specific question about the data:",
                    "What are the most interesting insights from this dataset and what actions would you recommend based on them?",
                )

            # Run analysis button
            if st.button("Run AI Analysis"):
                with st.spinner("Gemini AI is analyzing your data..."):
                    # Generate analysis
                    response = gemini_analyzer.analyze_data(
                        analysis_type, custom_question
                    )

                    if response:
                        # Display response
                        st.markdown(
                            '<h3 class="sub-header">AI Analysis Results</h3>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(response)

                        # Save analysis to file option
                        st.download_button(
                            label="Download Analysis",
                            data=response,
                            file_name="gemini_analysis.txt",
                            mime="text/plain",
                        )

else:
    # Display welcome message when no data is loaded
    st.markdown(
        """
        <section class="welcome-card">
          <div>
            <div class="welcome-eyebrow">✦ &nbsp;YOUR DATA, IN FOCUS</div>
            <h1>Make sense of<br><span>your data.</span></h1>
            <p>Turn a CSV into clear answers. Explore patterns, check data quality, and uncover insights in one place.</p>
            <div class="welcome-sample"><div class="sample-bar"></div><div class="sample-chart"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
            <div class="welcome-hint">↖ &nbsp; <b>Start with a CSV</b><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use the upload panel to begin your analysis.</div>
          </div>
          <div class="welcome-tiles">
            <div class="welcome-tile">▦ &nbsp;Quick overview<span>Know what’s in your data</span></div>
            <div class="welcome-tile">▥ &nbsp;Visual patterns<span>See trends at a glance</span></div>
            <div class="welcome-tile">⌗ &nbsp;Deeper analysis<span>Statistics and ML insights</span></div>
            <div class="welcome-tile">✧ &nbsp;AI assistance<span>Ask better questions</span></div>
          </div>
        </section>
        """
        , unsafe_allow_html=True,
    )
