import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64

# Import modules
from modules.input_module import DataLoader
from modules.data_visualization_module import Visualizer, AdvancedVisualizer
from modules.data_analysis_module import (
    StatisticalAnalyzer, DataProfiler, DataQualityAnalyzer,
    AdvancedStatisticalAnalyzer, AutomatedInsights, TimeSeriesAnalyzer,
    build_dataset_profile, build_column_profile, build_quality_report,
    generate_insights, pairwise_analysis, benjamini_hochberg, time_series_profile,
)
from modules.processing_module import GeminiAnalyzer
from modules.activity_log import ActivityLog

# Set page configuration
st.set_page_config(
    page_title="CSV Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('''<style>
.stApp{background:#f4f7ff}
.login-left{min-height:78vh;padding:42px 7%;border-radius:26px 0 0 26px;background:linear-gradient(135deg,#a9eee4,#70d0cd);color:#286f78}
.login-right{min-height:0;padding:52px 9% 12px;border-radius:0 26px 0 0;background:#fbfefe}
div[data-testid=stHorizontalBlock]>div[data-testid=column]:nth-child(2){background:#fbfefe;border-radius:0 26px 26px 0}
div[data-testid=stButton] button{background:#55bdb4!important;color:white!important;border:0!important;border-radius:14px;padding:.8rem 1.2rem;box-shadow:0 8px 20px #55bdb433}
div[data-testid=stButton] button:hover{background:#43aaa2!important;color:white!important}
.login-brand{font-size:1.1rem;font-weight:850;letter-spacing:.04em}
.login-chart{height:185px;margin:26px auto;padding:18px;border:1px solid #ffffffbd;border-radius:22px;background:#ffffff66;display:flex;align-items:end;justify-content:space-around}
.login-chart i{display:block;width:13%;border-radius:7px 7px 0 0;background:#f4c1a8}
.login-chart i:nth-child(2),.login-chart i:nth-child(5){height:65%;background:#79cbbb}
.login-chart i:nth-child(3){height:48%;background:#aaa0eb}.login-chart i:nth-child(4){height:85%;background:#4dbbb0}
.login-chart i:nth-child(1){height:35%}.login-chart i:nth-child(6){height:95%;background:#79cbbb}
.login-left h1{color:#236d78;font-weight:850}.login-left p{color:#347e87;line-height:1.8}
.login-details{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:24px;color:#347e87;font-size:.92rem;line-height:1.55}.login-details ol{padding-left:1.3rem;margin:.45rem 0}.login-details li{padding:.12rem 0}.login-details p{margin:.35rem 0 .8rem}.login-description{margin:0 0 18px}.login-copyright{display:block;margin-top:18px;color:#347e87}
.login-right h1{font-size:2.7rem;color:#315b60;font-weight:850;margin-top:3rem}
.login-right p{font-size:1.1rem;color:#80999d;line-height:1.8;margin:1.5rem 0}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e7e8f2}
.main-header{color:#17283d;font-weight:850}
.sub-header{color:#23384d}
.studio-hero{padding:38px;border:1px solid #e8e7f4;border-radius:24px;background:linear-gradient(115deg,#fff 15%,#f4f1ff);box-shadow:0 12px 40px #49408212}
.studio-kicker{color:#7267ce;font-size:.72rem;font-weight:850;letter-spacing:.22em}
.studio-title{color:#6657d5;font-size:3rem;line-height:1.05;font-weight:900;margin:26px 0 18px}
.studio-copy{color:#7b8798;line-height:1.8;max-width:520px}
.studio-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:30px}
.studio-card{min-height:90px;padding:18px;border:1px solid #e9e8f0;border-radius:16px;background:white;color:#435268;font-weight:750}
.studio-card small{display:block;color:#929bad;font-weight:400;margin-top:10px}
</style>''',unsafe_allow_html=True)

# Require Google sign-in before rendering any analyzer functionality.
activity_log = ActivityLog()

def begin_google_signin():
    activity_log.record(None, "login_started")
    st.login()

auth_config = st.secrets.get("auth")
if not auth_config:
    st.title("Sign-in setup required")
    st.error("Google sign-in is not configured yet. Follow the setup steps in README.md.")
    st.stop()

if not st.user.is_logged_in:
    left,right=st.columns([1.35,.9],gap="small")
    with left:
        st.markdown('''<div class="login-left"><div class="login-brand">▥ &nbsp; CSV FILE ANALYSER</div><div class="login-chart"><i></i><i></i><i></i><i></i><i></i><i></i></div><h1>CSV FILE ANALYSER</h1><p class='login-description'>In fulfilment of the requirements for the course IT303 Software Engineering.</p><div class='login-details'><div><b>Team members</b><ol><li>Apurv Rohom</li><li>Disha A</li><li>Pranathi Udaya Kumar</li></ol></div><div><b>Under the guidance of</b><p>Prof. Jaidh ar C D</p><b>Course</b><p>IT303 Software Engineering</p></div></div><small class='login-copyright'>Copyright © 2026 CSV File Analyser</small></div>''',unsafe_allow_html=True)
    with right:
        st.markdown('''<div class="login-right"><small style="color:#58aaa9;letter-spacing:.2em;font-weight:bold">CSV DATA ANALYSER</small><h1>Welcome back</h1><p>Continue to your workspace and start exploring your data.</p></div>''',unsafe_allow_html=True)
        st.button("Sign in",on_click=begin_google_signin,type="primary")
    st.stop()

# Record a successful Google-authenticated session once, then retain its prior login.
from datetime import datetime
from zoneinfo import ZoneInfo

user_email = str(st.user.get("email") or st.user.get("sub") or "unknown").strip().lower()
if st.session_state.get("_activity_logged_user") != user_email:
    previous_login = activity_log.latest_login(user_email)
    activity_log.record(user_email, "login_success")
    st.session_state["_activity_logged_user"] = user_email
    st.session_state["_activity_previous_login"] = previous_login
else:
    previous_login = st.session_state.get("_activity_previous_login")

def format_login_time(timestamp):
    if not timestamp:
        return "First recorded sign-in"
    try:
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return value.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")
    except (ValueError, TypeError):
        return timestamp

def sign_out_and_record():
    activity_log.record(user_email, "logout")
    st.session_state.pop("_activity_logged_user", None)
    st.session_state.pop("_activity_previous_login", None)
    st.logout()

# Add custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize modules
data_loader = DataLoader()
visualizer = Visualizer(data_loader)
statistical_analyzer = StatisticalAnalyzer(data_loader)
data_profiler = DataProfiler(data_loader)
data_quality_analyzer = DataQualityAnalyzer(data_loader)
advanced_statistical_analyzer = AdvancedStatisticalAnalyzer(data_loader)
automated_insights = AutomatedInsights(data_loader, data_profiler, data_quality_analyzer, advanced_statistical_analyzer)
time_series_analyzer = TimeSeriesAnalyzer(data_loader)
advanced_visualizer = AdvancedVisualizer(data_loader)
gemini_analyzer = GeminiAnalyzer(data_loader)

# Main header
# Sidebar for file upload and settings
with st.sidebar:
    st.caption(f"Signed in as {st.user.get('email', st.user.get('name', 'Google user'))}")
    st.caption(f"Last login: {format_login_time(previous_login)}")
    st.button("Sign out", on_click=sign_out_and_record)
    admin_emails = st.secrets.get("activity", {}).get("admin_emails", [])
    is_activity_admin = user_email in {str(email).strip().lower() for email in admin_emails}
    old_activity_expander_state = st.session_state.get("_activity_expander_open", False)
    with st.expander("Activity history", key="activity_history_expander", on_change="rerun"):
        visible_events = activity_log.recent(None if is_activity_admin else user_email, limit=100 if is_activity_admin else 25)
        if visible_events:
            import json
            activity_rows = [
                {
                    **({"User": item["user"]} if is_activity_admin else {}),
                    "Time (UTC)": item["timestamp"],
                    "Activity": item["action"].replace("_", " ").title(),
                    "Details": json.dumps(item["details"], ensure_ascii=False, sort_keys=True),
                }
                for item in visible_events
            ]
            st.dataframe(pd.DataFrame(activity_rows), hide_index=True, use_container_width=True, height=260)
            st.download_button(
                "Download activity log",
                data=activity_log.export_csv(None if is_activity_admin else user_email),
                file_name="csv_analyser_activity.csv",
                mime="text/csv",
                key="download_activity_log",
                on_click=lambda: activity_log.record(user_email, "activity_log_exported"),
            )
        else:
            st.caption("Your activity history will appear here.")
    activity_expander_open = st.session_state.get("activity_history_expander", False)
    if activity_expander_open != old_activity_expander_state:
        if activity_expander_open:
            activity_log.record(user_email, "activity_history_opened")
        st.session_state["_activity_expander_open"] = activity_expander_open
    st.markdown("---")
    st.header("Upload & Settings")

    # File upload
    uploaded_file = data_loader.create_upload_widget()

    if uploaded_file is not None:
        success = data_loader.load_data(uploaded_file)
        if success:
            st.success(f"Successfully loaded: {data_loader.get_filename()}")
            upload_event_id = str(getattr(uploaded_file, "file_id", "") or f"{uploaded_file.name}:{uploaded_file.size}")
            if st.session_state.get("_activity_last_upload") != upload_event_id:
                frame = data_loader.get_data()
                activity_log.record(
                    user_email,
                    "csv_uploaded",
                    {"rows": int(frame.shape[0]), "columns": int(frame.shape[1]), "size_bytes": int(uploaded_file.size)},
                )
                st.session_state["_activity_last_upload"] = upload_event_id
        else:
            upload_event_id = str(getattr(uploaded_file, "file_id", "") or f"{uploaded_file.name}:{uploaded_file.size}")
            if st.session_state.get("_activity_last_failed_upload") != upload_event_id:
                activity_log.record(user_email, "csv_upload_failed")
                st.session_state["_activity_last_failed_upload"] = upload_event_id

    # Gemini API key input
    st.markdown("---")
    st.subheader("Gemini AI Integration")
    gemini_configured = gemini_analyzer.create_api_key_input()
    if gemini_configured and not st.session_state.get("_activity_gemini_key_logged"):
        activity_log.record(user_email, "gemini_api_key_configured")
        st.session_state["_activity_gemini_key_logged"] = True

if data_loader.get_data() is None:
    st.markdown('<p style="color:#7267ce;letter-spacing:.2em;font-size:.75rem;font-weight:bold">YOUR DATA WORKSPACE</p><h1 class="main-header">CSV Data Studio</h1>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:#7267ce;letter-spacing:.2em;font-size:.75rem;font-weight:bold">YOUR DATA WORKSPACE</p><h1 class="main-header">CSV File Analyser</h1>', unsafe_allow_html=True)

# Main content area
if data_loader.get_data() is not None:
    # Display tabs for different analysis options
    analysis_tab_labels = [
        "📊 Overview",
        "🔎 EDA & Data Quality",
        "📈 Visualizations",
        "✨ Advanced Visualizations",
        "🧮 Statistical Analysis",
        "📐 Advanced Statistics",
        "🕒 Time Series",
        "🧠 Gemini AI Analysis",
    ]
    tab1, tab_eda, tab2, tab_advanced_viz, tab3, tab_advanced_stats, tab_time, tab5 = st.tabs(
        analysis_tab_labels, key="analysis_tabs", on_change="rerun"
    )
    active_analysis_tab = st.session_state.get("analysis_tabs", analysis_tab_labels[0])
    if st.session_state.get("_activity_last_tab") != active_analysis_tab:
        activity_log.record(user_email, "analysis_tab_opened", {"tab": active_analysis_tab})
        st.session_state["_activity_last_tab"] = active_analysis_tab

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
        st.dataframe(data_loader.get_data().head(10))

        st.markdown(
            '<h3 class="sub-header">Column Information</h3>', unsafe_allow_html=True
        )

        # Column information
        col_info = data_loader.get_column_info()
        st.dataframe(pd.DataFrame(col_info))

        # Download processed data
        st.markdown(
            '<h3 class="sub-header">Download Processed Data</h3>',
            unsafe_allow_html=True,
        )
        processed_data = data_loader.get_data()
        if processed_data is not None:
            processed_name = (data_loader.get_filename() or "data.csv").rsplit(".", 1)[0]
            st.download_button(
                "Download CSV File",
                data=processed_data.to_csv(index=False),
                file_name=f"{processed_name}_processed.csv",
                mime="text/csv",
                on_click=lambda: activity_log.record(user_email, "processed_csv_downloaded"),
            )

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
                st.dataframe(desc_stats)

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
                    st.dataframe(grouped_data)

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
                st.dataframe(missing_summary)
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
                st.dataframe(outlier_summary)
            else:
                st.info("No outliers found using the selected method.")

    # EDA & Data Quality: combine supplied workbench and analyzer modules.
    with tab_eda:
        st.markdown('<h2 class="sub-header">Exploratory Data Analysis & Quality</h2>', unsafe_allow_html=True)
        frame = data_loader.get_data()
        profile = data_profiler.get_dataset_profile()
        if profile:
            metrics = st.columns(4)
            metrics[0].metric("Rows", f"{profile['rows']:,}")
            metrics[1].metric("Columns", profile["columns"])
            metrics[2].metric("Missing cells", f"{profile['missing_cells']:,}", f"{profile['missing_percentage']:.2f}%")
            metrics[3].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
            st.caption(f"{profile['numeric_count']} numeric · {profile['categorical_count']} categorical · {profile['datetime_count']} datetime · {profile['boolean_count']} boolean columns")
            with st.expander("Full dataset profile"):
                st.json(__import__("json").loads(__import__("json").dumps(profile, default=str)))
            st.subheader("Column profile")
            selected_profile_column = st.selectbox("Choose a column", frame.columns, key="eda_column_profile")
            col_profile = data_profiler.get_column_profile(selected_profile_column)
            st.json(__import__("json").loads(__import__("json").dumps(col_profile, default=str)))
            st.dataframe(pd.DataFrame([build_column_profile(frame, selected_profile_column)]), use_container_width=True)
            quality = data_quality_analyzer.get_quality_report() or {}
            q1, q2 = st.columns(2)
            q1.metric("Data health score", f"{quality.get('health_score', 0)}/100")
            q2.metric("Quality issues", quality.get("total_issues", 0))
            issues = quality.get("issues", [])
            if issues:
                st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
            else:
                st.success("No data quality issues detected.")
            with st.expander("Additional quality checks"):
                st.write("Workbench dataset profile")
                st.json(__import__("json").loads(__import__("json").dumps(build_dataset_profile(frame), default=str)))
                st.write("Workbench checks")
                st.dataframe(pd.DataFrame(build_quality_report(frame)), use_container_width=True, hide_index=True)
                st.write("Missingness analysis")
                missing_report = data_quality_analyzer.get_missingness_analysis()
                if missing_report and missing_report.get("missing_summary") is not None:
                    st.dataframe(missing_report["missing_summary"], use_container_width=True, hide_index=True)
                st.write("Duplicate analysis")
                st.json(__import__("json").loads(__import__("json").dumps(data_quality_analyzer.get_duplicate_analysis(), default=str)))
            st.subheader("Automated EDA report")
            if st.button("Generate EDA report", key="generate_eda_report"):
                report = automated_insights.generate_eda_report()
                st.markdown(automated_insights.format_report_for_display(report))
                st.write("Quick insights")
                for insight in generate_insights(frame):
                    st.write(f"• {insight}")
                activity_log.record(user_email, "eda_report_generated", {"columns": int(frame.shape[1])})
            st.divider()
            st.subheader("Reversible data cleanup")
            duplicate_count = int(frame.duplicated().sum())
            clean_col, reset_col = st.columns(2)
            with clean_col:
                st.caption(f"Exact duplicate rows available to remove: {duplicate_count}")
                if st.button("Remove exact duplicate rows", disabled=duplicate_count == 0, key="remove_duplicate_rows"):
                    removed = data_loader.remove_duplicates()
                    activity_log.record(user_email, "duplicate_rows_removed", {"rows_removed": int(removed)})
                    st.success(f"Removed {removed} duplicate rows. You can restore the uploaded data with Reset data.")
                    st.rerun()
            with reset_col:
                if st.button("Reset to uploaded data", key="reset_uploaded_data", disabled=not st.session_state.get("transformations")):
                    data_loader.reset_data()
                    activity_log.record(user_email, "data_reset_to_upload")
                    st.rerun()

    # Advanced Visualizations use the supplied AdvancedVisualizer class.
    with tab_advanced_viz:
        st.markdown('<h2 class="sub-header">Advanced Visualizations</h2>', unsafe_allow_html=True)
        numeric_cols = data_loader.get_numeric_columns()
        categorical_cols = data_loader.get_categorical_columns()
        chart = st.selectbox("Chart", ["ECDF", "Q-Q plot", "Violin plot", "Boxen plot", "Pareto chart", "Missingness heatmap", "Missingness bar chart", "Hexbin", "2D density", "Parallel coordinates", "Residual plot", "Grouped box plot", "Grouped violin plot"], key="advanced_chart")
        fig = None
        if chart in ("ECDF", "Q-Q plot", "Violin plot", "Boxen plot") and numeric_cols:
            col = st.selectbox("Numeric column", numeric_cols, key="advanced_chart_numeric")
            method = {"ECDF": advanced_visualizer.create_ecdf_plot, "Q-Q plot": advanced_visualizer.create_qq_plot, "Violin plot": advanced_visualizer.create_violin_plot, "Boxen plot": advanced_visualizer.create_boxen_plot}[chart]
            fig = method(col)
        elif chart == "Pareto chart" and categorical_cols:
            fig = advanced_visualizer.create_pareto_chart(st.selectbox("Category column", categorical_cols, key="advanced_pareto_cat"))
        elif chart == "Missingness heatmap":
            fig = advanced_visualizer.create_missingness_heatmap()
        elif chart == "Missingness bar chart":
            fig = advanced_visualizer.create_missingness_bar_chart()
        elif chart in ("Hexbin", "2D density") and len(numeric_cols) >= 2:
            xcol, ycol = st.columns(2)
            x = xcol.selectbox("X column", numeric_cols, key="advanced_xy_x")
            y = ycol.selectbox("Y column", [c for c in numeric_cols if c != x], key="advanced_xy_y")
            fig = advanced_visualizer.create_hexbin_plot(x, y) if chart == "Hexbin" else advanced_visualizer.create_density_2d_plot(x, y)
        elif chart == "Parallel coordinates" and len(numeric_cols) >= 3:
            cols = st.multiselect("Numeric columns (choose at least three)", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key="parallel_columns")
            hue_options = ["None"] + categorical_cols
            hue = st.selectbox("Color group", hue_options, key="parallel_hue")
            if len(cols) >= 3:
                fig = advanced_visualizer.create_parallel_coordinates_plot(cols, None if hue == "None" else hue)
        elif chart == "Residual plot" and len(numeric_cols) >= 2:
            x, y = st.columns(2)
            xcol = x.selectbox("Predictor", numeric_cols, key="residual_x")
            ycol = y.selectbox("Response", [c for c in numeric_cols if c != xcol], key="residual_y")
            fig = advanced_visualizer.create_residual_plot(xcol, ycol)
        elif chart in ("Grouped box plot", "Grouped violin plot") and numeric_cols and categorical_cols:
            numeric = st.selectbox("Numeric value", numeric_cols, key="grouped_numeric")
            category = st.selectbox("Group by", categorical_cols, key="grouped_category")
            fig = advanced_visualizer.create_grouped_box_plot(numeric, category) if chart == "Grouped box plot" else advanced_visualizer.create_grouped_violin_plot(numeric, category)
        else:
            st.info("This chart needs more numeric or categorical columns in the uploaded CSV.")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            activity_log.record(user_email, "advanced_visualization_opened", {"chart": chart})

    # Advanced statistical tests and pairwise analysis.
    with tab_advanced_stats:
        st.markdown('<h2 class="sub-header">Advanced Statistical Analysis</h2>', unsafe_allow_html=True)
        frame = data_loader.get_data()
        all_columns = frame.columns.tolist()
        pair_cols = st.columns(2)
        first = pair_cols[0].selectbox("First variable", all_columns, key="advanced_pair_first")
        second = pair_cols[1].selectbox("Second variable", [c for c in all_columns if c != first] or all_columns, key="advanced_pair_second")
        pair_result = pairwise_analysis(frame, first, second)
        if "contingency" in pair_result:
            st.dataframe(pair_result["contingency"], use_container_width=True)
            pair_result = {k: v for k, v in pair_result.items() if k != "contingency"}
        st.json(__import__("json").loads(__import__("json").dumps(pair_result, default=str)))
        if len(data_loader.get_numeric_columns()) >= 2:
            st.subheader("Correlation matrix")
            numeric_cols = data_loader.get_numeric_columns()
            corr_cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:min(12, len(numeric_cols))], key="advanced_corr_cols")
            corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="advanced_corr_method")
            if len(corr_cols) >= 2:
                corr_matrix = advanced_statistical_analyzer.get_correlation_matrix(method=corr_method, columns=corr_cols)
                st.dataframe(corr_matrix, use_container_width=True)
                p_values = []
                for i, left in enumerate(corr_cols):
                    for right in corr_cols[i+1:]:
                        result = pairwise_analysis(frame, left, right).get("correlations", {}).get(corr_method, {})
                        p_values.append(result.get("p_value", np.nan))
                if p_values:
                    adjusted = benjamini_hochberg(p_values)
                    st.caption(f"Benjamini–Hochberg adjusted p-values for {len(p_values)} tested pairs: " + ", ".join("NA" if not np.isfinite(v) else f"{v:.4g}" for v in adjusted))
        if data_loader.get_numeric_columns():
            st.subheader("Univariate detail")
            selected = st.selectbox("Numeric variable", data_loader.get_numeric_columns(), key="advanced_univariate")
            st.json(__import__("json").loads(__import__("json").dumps(advanced_statistical_analyzer.get_univariate_analysis(selected), default=str)))

    # Time-series tools use copy-safe analyzer and explicit profile helper.
    with tab_time:
        st.markdown('<h2 class="sub-header">Time-Series Analysis</h2>', unsafe_allow_html=True)
        date_cols = time_series_analyzer.get_datetime_columns()
        numeric_cols = data_loader.get_numeric_columns()
        if date_cols and numeric_cols:
            date_col = st.selectbox("Date/time column", date_cols, key="time_date_col")
            value_col = st.selectbox("Numeric value column", numeric_cols, key="time_value_col")
            analysis = time_series_analyzer.analyze_time_series(date_col, value_col)
            if analysis and "error" not in analysis:
                st.json(__import__("json").loads(__import__("json").dumps(analysis, default=str)))
                chart_kind = st.selectbox("Time-series view", ["Trend line", "Rolling statistics", "Seasonal decomposition", "ACF/PACF", "Lag plot", "Percentage change", "Resampled values"], key="time_chart_kind")
                fig = None
                try:
                    if chart_kind == "Trend line":
                        fig = time_series_analyzer.create_time_series_plot(date_col, value_col)
                    elif chart_kind == "Rolling statistics":
                        max_window = max(2, min(30, int(data_loader.get_data().shape[0] // 2)))
                        window = st.slider("Rolling window", 2, max_window, min(7, max_window), key="time_rolling_window")
                        fig = time_series_analyzer.create_rolling_statistics_plot(date_col, value_col, window)
                    elif chart_kind == "Seasonal decomposition":
                        period = st.number_input("Seasonal period", min_value=2, max_value=max(2, len(data_loader.get_data()) // 2), value=min(7, max(2, len(data_loader.get_data()) // 2)), key="time_season_period")
                        fig = time_series_analyzer.create_seasonal_decomposition_plot(date_col, value_col, period=int(period))
                    elif chart_kind == "ACF/PACF":
                        lags = st.slider("Lags", 1, max(1, min(40, len(data_loader.get_data()) - 1)), min(20, max(1, len(data_loader.get_data()) - 1)), key="time_lags")
                        fig = time_series_analyzer.create_acf_pacf_plots(date_col, value_col, lags=lags)
                    elif chart_kind == "Lag plot":
                        lag = st.number_input("Lag", min_value=1, max_value=max(1, len(data_loader.get_data()) - 1), value=1, key="time_lag")
                        fig = time_series_analyzer.create_lag_plot(value_col, lag=int(lag))
                    elif chart_kind == "Percentage change":
                        fig = time_series_analyzer.create_percentage_change_plot(date_col, value_col)
                    else:
                        frequency = st.selectbox("Frequency", ["D", "W", "ME", "QE", "YE"], key="time_resample_frequency")
                        agg = st.selectbox("Aggregation", ["mean", "sum", "min", "max", "median", "count"], key="time_resample_agg")
                        resampled = time_series_analyzer.resample_time_series(date_col, value_col, frequency, agg)
                        if resampled is not None:
                            st.dataframe(resampled.rename("value").to_frame(), use_container_width=True)
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                except (ValueError, TypeError, ZeroDivisionError) as exc:
                    st.warning(f"Could not create this time-series view: {exc}")
                with st.expander("Time-series profile options"):
                    duplicate_rule = st.selectbox("Duplicate timestamps", ["mean", "keep first", "keep last"], key="time_duplicate_rule")
                    frequency = st.selectbox("Optional resampling", ["None", "D", "W", "MS", "QS", "YS"], key="time_frequency")
                    if st.button("Build time-series profile", key="build_time_profile"):
                        result = time_series_profile(data_loader.get_data(), date_col, value_col, duplicate_rule, None if frequency == "None" else frequency)
                        if "data" in result:
                            result["data"] = result["data"].to_dict()
                        st.json(__import__("json").loads(__import__("json").dumps(result, default=str)))
                        activity_log.record(user_email, "time_series_profile_generated", {"date_column": str(date_col), "value_column": str(value_col)})
            elif analysis:
                st.warning(analysis.get("error", "Could not analyze these columns."))
        else:
            st.info("Upload a CSV containing a date/time column and at least one numeric column to use time-series analysis.")

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
                activity_log.record(user_email, "gemini_analysis_run", {"analysis_type": analysis_type})
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
                            on_click=lambda: activity_log.record(user_email, "gemini_analysis_downloaded"),
                        )

else:
    st.markdown("""<div class="studio-hero"><div class="studio-kicker">✦ &nbsp; YOUR DATA, IN FOCUS</div><div class="studio-title">Make sense of<br>your data.</div><div class="studio-copy">Turn a CSV into clear answers. Explore patterns, check data quality, and uncover insights in one place.</div><div class="studio-grid"><div class="studio-card">▦ &nbsp; Quick overview<small>Know what’s in your data</small></div><div class="studio-card">▥ &nbsp; Visual patterns<small>See trends at a glance</small></div><div class="studio-card">⌗ &nbsp; Deeper analysis<small>Statistics and data quality</small></div><div class="studio-card">✦ &nbsp; AI assistance<small>Ask better questions</small></div></div></div>""",unsafe_allow_html=True)
    st.caption("Start with a CSV — use the upload panel on the left to begin your analysis.")
