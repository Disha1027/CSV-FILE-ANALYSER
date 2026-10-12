import io
import datetime
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_footer(num_pages)
            super().showPage()
        super().save()

    def draw_page_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 30, "CSV File Analyser • Data Analysis Report")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 30, page_str)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(54, 42, 612 - 54, 42)
        self.restoreState()


def _fig_to_image(fig, width=6.2 * inch, height=3.0 * inch):
    """Convert a Matplotlib figure into a ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    return img


def generate_pdf_report(
    data_loader,
    data_profiler=None,
    data_quality_analyzer=None,
    statistical_analyzer=None,
    gemini_analysis_text=None,
    automated_insights=None,
    advanced_visualizer=None,
    time_series_analyzer=None,
    section: str = "all",
) -> bytes:
    """
    Generate a formatted PDF report for the loaded dataset.
    Supports either a specific section ('overview', 'eda', 'visualizations',
    'advanced_viz', 'statistics', 'advanced_stats', 'time_series', 'gemini')
    or 'all' for the master comprehensive report covering all sections.

    Returns:
        bytes: Raw PDF file bytes ready for download.
    """
    df = data_loader.get_data()
    if df is None or df.empty:
        raise ValueError("No dataset loaded to generate analysis report.")

    filename = data_loader.get_filename() or "dataset.csv"
    data_info = data_loader.get_data_info()
    numeric_cols = data_loader.get_numeric_columns()
    categorical_cols = data_loader.get_categorical_columns()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#17283d'),
    )

    kicker_style = ParagraphStyle(
        'DocKicker',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#6657d5'),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#17283d'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#334155'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
        leftIndent=14,
        spaceAfter=3,
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1e293b'),
    )

    cell_header_style = ParagraphStyle(
        'TableCellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a'),
    )

    elements = []
    sec = section.lower()

    # Section display names
    section_titles = {
        "all": "Master Comprehensive Analysis Report",
        "overview": "Section 1: Data Overview & Schema Report",
        "eda": "Section 2: EDA & Data Quality Report",
        "visualizations": "Section 3: Visualizations & Distributions Report",
        "advanced_viz": "Section 4: Advanced Visualizations Report",
        "statistics": "Section 5: Statistical Analysis Report",
        "advanced_stats": "Section 6: Advanced Statistics Report",
        "time_series": "Section 7: Time-Series Analysis Report",
        "gemini": "Section 8: Gemini AI Analysis Report",
    }
    report_title = section_titles.get(sec, "Data Analysis Report")

    # Header / Title block
    elements.append(Paragraph("DATA WORKSPACE REPORT", kicker_style))
    elements.append(Paragraph(report_title, title_style))
    gen_time = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(f"<b>Dataset:</b> {filename} &nbsp;|&nbsp; <b>Generated:</b> {gen_time}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#6657d5"), spaceAfter=14))

    # Helper: Add Overview Elements
    def add_overview_section():
        elements.append(Paragraph("Dataset Overview & Metrics", h1_style))
        overview_data = [
            [Paragraph("Filename", cell_header_style), Paragraph(str(filename), cell_style),
             Paragraph("Total Records", cell_header_style), Paragraph(f"{data_info.get('rows', 0):,}", cell_style)],
            [Paragraph("Total Columns", cell_header_style), Paragraph(str(data_info.get('columns', 0)), cell_style),
             Paragraph("Memory Usage", cell_header_style), Paragraph(str(data_info.get('memory_usage', 'N/A')), cell_style)],
            [Paragraph("Numeric Columns", cell_header_style), Paragraph(str(len(numeric_cols)), cell_style),
             Paragraph("Categorical Columns", cell_header_style), Paragraph(str(len(categorical_cols)), cell_style)],
            [Paragraph("Missing Values", cell_header_style), Paragraph(f"{data_info.get('missing_values', 0):,}", cell_style),
             Paragraph("Duplicate Rows", cell_header_style), Paragraph(f"{int(df.duplicated().sum()):,}", cell_style)],
        ]
        t_overview = Table(overview_data, colWidths=[1.3 * inch, 1.7 * inch, 1.4 * inch, 1.8 * inch])
        t_overview.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafd')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(t_overview)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Column Schema & Completeness", h2_style))
        col_table_data = [[
            Paragraph("Column Name", cell_header_style),
            Paragraph("Data Type", cell_header_style),
            Paragraph("Non-Null", cell_header_style),
            Paragraph("Missing", cell_header_style),
            Paragraph("Missing %", cell_header_style),
            Paragraph("Unique Values", cell_header_style),
        ]]
        for col in df.columns[:25]:
            null_count = int(df[col].isnull().sum())
            non_null = int(df[col].count())
            pct_missing = (null_count / len(df)) * 100 if len(df) > 0 else 0
            unique_cnt = int(df[col].nunique())
            col_table_data.append([
                Paragraph(str(col)[:22], cell_style),
                Paragraph(str(df[col].dtype), cell_style),
                Paragraph(f"{non_null:,}", cell_style),
                Paragraph(f"{null_count:,}", cell_style),
                Paragraph(f"{pct_missing:.1f}%", cell_style),
                Paragraph(f"{unique_cnt:,}", cell_style),
            ])
        t_cols = Table(col_table_data, colWidths=[1.8 * inch, 1.0 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch])
        t_cols.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafd')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_cols)
        elements.append(Spacer(1, 12))

    # Helper: Add EDA & Data Quality Section
    def add_eda_section():
        elements.append(Paragraph("Exploratory Data Analysis & Quality Report", h1_style))
        if data_quality_analyzer:
            try:
                quality = data_quality_analyzer.get_quality_report() or {}
                health_score = quality.get('health_score', 100)
                total_issues = quality.get('total_issues', 0)
                elements.append(Paragraph(f"<b>Data Health Score:</b> {health_score}/100 &nbsp;|&nbsp; <b>Total Quality Issues:</b> {total_issues}", body_style))
                elements.append(Spacer(1, 6))
                issues = quality.get('issues', [])
                if issues:
                    issue_rows = [[
                        Paragraph("Category", cell_header_style),
                        Paragraph("Severity", cell_header_style),
                        Paragraph("Description", cell_header_style),
                    ]]
                    for iss in issues[:12]:
                        issue_rows.append([
                            Paragraph(str(iss.get('category', 'General')), cell_style),
                            Paragraph(str(iss.get('severity', 'Info')).capitalize(), cell_style),
                            Paragraph(str(iss.get('description', '')), cell_style),
                        ])
                    t_issues = Table(issue_rows, colWidths=[1.4 * inch, 1.0 * inch, 3.8 * inch])
                    t_issues.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fee2e2' if health_score < 70 else '#fef3c7')),
                        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(t_issues)
                else:
                    elements.append(Paragraph("No data quality issues detected.", body_style))
            except Exception:
                pass
        elements.append(Spacer(1, 12))

    # Helper: Add Visualizations Section
    def add_viz_section():
        elements.append(Paragraph("Visualizations & Distribution Charts", h1_style))
        if len(numeric_cols) >= 2:
            try:
                corr_cols = numeric_cols[:10]
                corr_matrix = df[corr_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
                ax.set_title("Correlation Matrix Heatmap", fontsize=11, fontweight='bold', pad=8)
                plt.tight_layout()
                elements.append(Paragraph("Correlation Matrix Heatmap", h2_style))
                elements.append(_fig_to_image(fig, width=6.2 * inch, height=3.2 * inch))
                elements.append(Spacer(1, 10))
            except Exception:
                pass

        if numeric_cols:
            elements.append(Paragraph("Numeric Feature Distributions & Boxplots", h2_style))
            for col in numeric_cols[:4]:
                try:
                    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(9, 3.4))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax_h, color='#6657d5')
                    ax_h.set_title(f"Distribution of {col}", fontsize=9, fontweight='bold')
                    ax_h.grid(True, alpha=0.2)
                    sns.boxplot(y=df[col].dropna(), ax=ax_b, color='#55bdb4')
                    ax_b.set_title(f"Boxplot of {col}", fontsize=9, fontweight='bold')
                    ax_b.grid(True, alpha=0.2)
                    plt.tight_layout()
                    elements.append(_fig_to_image(fig, width=6.2 * inch, height=2.4 * inch))
                    elements.append(Spacer(1, 8))
                except Exception:
                    pass

        if categorical_cols:
            elements.append(Paragraph("Categorical Feature Distributions", h2_style))
            for col in categorical_cols[:3]:
                try:
                    vc = df[col].value_counts().head(10)
                    if len(vc) > 0:
                        fig, ax = plt.subplots(figsize=(8, 3.0))
                        sns.barplot(x=vc.values, y=vc.index.astype(str), hue=vc.index.astype(str), legend=False, ax=ax, palette='mako')
                        ax.set_title(f"Top Categories in {col}", fontsize=9, fontweight='bold')
                        ax.grid(True, axis='x', alpha=0.2)
                        plt.tight_layout()
                        elements.append(_fig_to_image(fig, width=6.2 * inch, height=2.3 * inch))
                        elements.append(Spacer(1, 8))
                except Exception:
                    pass
        elements.append(Spacer(1, 10))

    # Helper: Add Advanced Visualizations Section
    def add_advanced_viz_section():
        elements.append(Paragraph("Advanced Visualizations", h1_style))
        if numeric_cols:
            col = numeric_cols[0]
            try:
                # ECDF
                sorted_data = np.sort(df[col].dropna())
                n = len(sorted_data)
                ecdf = np.arange(1, n + 1) / n
                fig, ax = plt.subplots(figsize=(8, 3.4))
                ax.plot(sorted_data, ecdf, marker='.', linestyle='none', color='#6657d5')
                ax.set_title(f"Empirical Cumulative Distribution Function (ECDF) - {col}", fontsize=10, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel("ECDF")
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                elements.append(_fig_to_image(fig, width=6.2 * inch, height=2.5 * inch))
                elements.append(Spacer(1, 8))
            except Exception:
                pass

        # Missingness bar chart
        missing_s = df.isnull().sum()
        missing_c = missing_s[missing_s > 0]
        if len(missing_c) > 0:
            try:
                fig, ax = plt.subplots(figsize=(8, 3.0))
                sns.barplot(x=missing_c.values, y=missing_c.index, hue=missing_c.index, legend=False, ax=ax, palette='Reds_r')
                ax.set_title("Missing Values by Feature", fontsize=10, fontweight='bold')
                ax.grid(True, axis='x', alpha=0.25)
                plt.tight_layout()
                elements.append(_fig_to_image(fig, width=6.2 * inch, height=2.4 * inch))
                elements.append(Spacer(1, 8))
            except Exception:
                pass
        elements.append(Spacer(1, 10))

    # Helper: Add Statistical Analysis Section
    def add_stats_section():
        elements.append(Paragraph("Statistical Analysis & Descriptive Metrics", h1_style))
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            desc_data = [[
                Paragraph("Column", cell_header_style),
                Paragraph("Mean", cell_header_style),
                Paragraph("Std Dev", cell_header_style),
                Paragraph("Min", cell_header_style),
                Paragraph("25%", cell_header_style),
                Paragraph("Median", cell_header_style),
                Paragraph("75%", cell_header_style),
                Paragraph("Max", cell_header_style),
            ]]
            for col in numeric_cols[:15]:
                rv = desc.loc[col]
                desc_data.append([
                    Paragraph(str(col)[:20], cell_style),
                    Paragraph(f"{rv['mean']:.2f}", cell_style),
                    Paragraph(f"{rv['std']:.2f}", cell_style),
                    Paragraph(f"{rv['min']:.2f}", cell_style),
                    Paragraph(f"{rv['25%']:.2f}", cell_style),
                    Paragraph(f"{rv['50%']:.2f}", cell_style),
                    Paragraph(f"{rv['75%']:.2f}", cell_style),
                    Paragraph(f"{rv['max']:.2f}", cell_style),
                ])
            t_desc = Table(desc_data, colWidths=[1.5 * inch, 0.7 * inch, 0.7 * inch, 0.6 * inch, 0.6 * inch, 0.7 * inch, 0.6 * inch, 0.8 * inch])
            t_desc.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafd')]),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(t_desc)
        elements.append(Spacer(1, 12))

    # Helper: Add Advanced Statistics Section
    def add_advanced_stats_section():
        elements.append(Paragraph("Advanced Statistics & Correlation Matrix", h1_style))
        if len(numeric_cols) >= 2:
            corr_m = df[numeric_cols[:10]].corr()
            corr_rows = [[Paragraph("Feature", cell_header_style)] + [Paragraph(str(c)[:8], cell_header_style) for c in corr_m.columns]]
            for r in corr_m.index:
                row_items = [Paragraph(str(r)[:14], cell_style)]
                for val in corr_m.loc[r]:
                    row_items.append(Paragraph(f"{val:.2f}", cell_style))
                corr_rows.append(row_items)
            col_w = max(0.5 * inch, 5.2 * inch / (len(corr_m.columns) + 1))
            t_c = Table(corr_rows, colWidths=[1.2 * inch] + [col_w] * len(corr_m.columns))
            t_c.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(t_c)
        elements.append(Spacer(1, 12))

    # Helper: Add Time-Series Section
    def add_time_series_section():
        elements.append(Paragraph("Time-Series Analysis", h1_style))
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or 'date' in c.lower() or 'time' in c.lower()]
        if date_cols and numeric_cols:
            d_col, v_col = date_cols[0], numeric_cols[0]
            try:
                parsed_dates = pd.to_datetime(df[d_col], errors='coerce')
                valid_mask = parsed_dates.notnull()
                if valid_mask.sum() > 2:
                    ts_df = pd.DataFrame({'date': parsed_dates[valid_mask], 'val': df[v_col][valid_mask]}).sort_values('date')
                    fig, ax = plt.subplots(figsize=(8, 3.2))
                    ax.plot(ts_df['date'], ts_df['val'], color='#6657d5', linewidth=1.5)
                    ax.set_title(f"Time Series Trend: {v_col} over {d_col}", fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.25)
                    plt.tight_layout()
                    elements.append(_fig_to_image(fig, width=6.2 * inch, height=2.5 * inch))
            except Exception:
                pass
        else:
            elements.append(Paragraph("No time-series date column detected in dataset.", body_style))
        elements.append(Spacer(1, 12))

    # Helper: Add Gemini AI Section
    def add_gemini_section():
        elements.append(Paragraph("Gemini AI Analysis & Intelligent Insights", h1_style))
        if gemini_analysis_text:
            for para in str(gemini_analysis_text).split("\n\n"):
                clean_p = para.strip().replace("\n", " ")
                if clean_p:
                    elements.append(Paragraph(clean_p, body_style))
                    elements.append(Spacer(1, 4))
        else:
            try:
                from modules.data_analysis_module import generate_insights
                insights = generate_insights(df)
                if insights:
                    elements.append(Paragraph("Automated Data Insights:", h2_style))
                    for ins in insights:
                        elements.append(Paragraph(f"• {ins}", bullet_style))
            except Exception:
                elements.append(Paragraph("No AI analysis run yet for this dataset.", body_style))
        elements.append(Spacer(1, 12))

    # Assemble elements based on requested section
    if sec == "overview":
        add_overview_section()
    elif sec == "eda":
        add_overview_section()
        add_eda_section()
    elif sec == "visualizations":
        add_viz_section()
    elif sec == "advanced_viz":
        add_advanced_viz_section()
    elif sec == "statistics":
        add_stats_section()
    elif sec == "advanced_stats":
        add_advanced_stats_section()
    elif sec == "time_series":
        add_time_series_section()
    elif sec == "gemini":
        add_gemini_section()
    else:  # "all" - Full master report
        add_overview_section()
        add_stats_section()
        add_eda_section()
        elements.append(PageBreak())
        add_viz_section()
        elements.append(PageBreak())
        add_advanced_viz_section()
        add_advanced_stats_section()
        add_time_series_section()
        elements.append(PageBreak())
        add_gemini_section()

    doc.build(elements, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


def generate_full_archive_zip(
    data_loader,
    data_profiler=None,
    data_quality_analyzer=None,
    statistical_analyzer=None,
    gemini_analysis_text=None,
    automated_insights=None,
    advanced_visualizer=None,
    time_series_analyzer=None,
) -> bytes:
    """
    Generate a complete ZIP archive downloading everything across all sections:
    1. Master Analysis PDF Report (all sections)
    2. Processed CSV Dataset
    3. Descriptive Statistics CSV
    4. Correlation Matrix CSV
    5. Data Quality Issues CSV
    6. Gemini AI Analysis (txt)
    7. README summary

    Returns:
        bytes: Raw ZIP archive bytes.
    """
    df = data_loader.get_data()
    raw_name = (data_loader.get_filename() or "data.csv").rsplit(".", 1)[0]
    zip_buf = io.BytesIO()

    master_pdf = generate_pdf_report(
        data_loader=data_loader,
        data_profiler=data_profiler,
        data_quality_analyzer=data_quality_analyzer,
        statistical_analyzer=statistical_analyzer,
        gemini_analysis_text=gemini_analysis_text,
        automated_insights=automated_insights,
        advanced_visualizer=advanced_visualizer,
        time_series_analyzer=time_series_analyzer,
        section="all",
    )

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{raw_name}_master_analysis_report.pdf", master_pdf)
        if df is not None:
            zf.writestr(f"{raw_name}_processed_data.csv", df.to_csv(index=False))

            num_cols = data_loader.get_numeric_columns()
            if num_cols:
                try:
                    desc_df = df[num_cols].describe().T
                    zf.writestr(f"{raw_name}_descriptive_statistics.csv", desc_df.to_csv())
                    if len(num_cols) > 1:
                        corr_df = df[num_cols].corr()
                        zf.writestr(f"{raw_name}_correlation_matrix.csv", corr_df.to_csv())
                except Exception:
                    pass

            if data_quality_analyzer:
                try:
                    q_rep = data_quality_analyzer.get_quality_report() or {}
                    issues = q_rep.get("issues", [])
                    if issues:
                        zf.writestr(f"{raw_name}_quality_issues.csv", pd.DataFrame(issues).to_csv(index=False))
                except Exception:
                    pass

            if gemini_analysis_text:
                zf.writestr(f"{raw_name}_gemini_ai_analysis.txt", str(gemini_analysis_text))

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            readme_text = (
                f"CSV FILE ANALYSER - MASTER EXPORT ARCHIVE\n"
                f"Dataset: {data_loader.get_filename()}\n"
                f"Export Timestamp: {now_str}\n\n"
                f"Archived Files:\n"
                f"• {raw_name}_master_analysis_report.pdf (Comprehensive PDF Report covering all sections)\n"
                f"• {raw_name}_processed_data.csv (Processed dataset)\n"
                f"• {raw_name}_descriptive_statistics.csv (Numeric statistical metrics)\n"
                f"• {raw_name}_correlation_matrix.csv (Feature correlation values)\n"
                f"• {raw_name}_quality_issues.csv (Data quality and health issues)\n"
            )
            zf.writestr("README_ANALYSIS.txt", readme_text)

    zip_buf.seek(0)
    return zip_buf.getvalue()


def generate_section_archive_zip(
    section_id: str,
    data_loader,
    data_profiler=None,
    data_quality_analyzer=None,
    statistical_analyzer=None,
    gemini_analysis_text=None,
    automated_insights=None,
    advanced_visualizer=None,
    time_series_analyzer=None,
    extra_data=None,
    extra_filename=None,
) -> bytes:
    """
    Generate a complete ZIP archive for a single specific section, containing:
    1. The section's comprehensive PDF Report
    2. Any section data exports (CSV / JSON / TXT)
    3. Section README summary
    """
    df = data_loader.get_data()
    raw_name = (data_loader.get_filename() or "data.csv").rsplit(".", 1)[0]
    zip_buf = io.BytesIO()

    # 1. Section PDF report
    sec_pdf = generate_pdf_report(
        data_loader=data_loader,
        data_profiler=data_profiler,
        data_quality_analyzer=data_quality_analyzer,
        statistical_analyzer=statistical_analyzer,
        gemini_analysis_text=gemini_analysis_text,
        automated_insights=automated_insights,
        advanced_visualizer=advanced_visualizer,
        time_series_analyzer=time_series_analyzer,
        section=section_id,
    )

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{raw_name}_{section_id}_report.pdf", sec_pdf)

        # 2. Section data files
        if extra_data is not None and extra_filename:
            payload = extra_data.encode('utf-8') if isinstance(extra_data, str) else (extra_data if isinstance(extra_data, bytes) else str(extra_data).encode('utf-8'))
            zf.writestr(extra_filename, payload)

        # Add additional section-specific data if applicable
        if section_id == "overview" and df is not None:
            zf.writestr(f"{raw_name}_processed_data.csv", df.to_csv(index=False))
            try:
                col_info = pd.DataFrame(data_loader.get_column_info())
                zf.writestr(f"{raw_name}_column_schema.csv", col_info.to_csv(index=False))
            except Exception:
                pass
        elif section_id == "eda" and data_quality_analyzer:
            try:
                q_rep = data_quality_analyzer.get_quality_report() or {}
                issues = q_rep.get("issues", [])
                if issues:
                    zf.writestr(f"{raw_name}_quality_issues.csv", pd.DataFrame(issues).to_csv(index=False))
                if data_profiler:
                    prof = data_profiler.get_dataset_profile()
                    if prof:
                        import json
                        zf.writestr(f"{raw_name}_dataset_profile.json", json.dumps(prof, default=str, indent=2))
            except Exception:
                pass
        elif section_id in ("statistics", "advanced_stats") and df is not None:
            num_cols = data_loader.get_numeric_columns()
            if num_cols:
                try:
                    desc_df = df[num_cols].describe().T
                    zf.writestr(f"{raw_name}_descriptive_statistics.csv", desc_df.to_csv())
                    if len(num_cols) > 1:
                        corr_df = df[num_cols].corr()
                        zf.writestr(f"{raw_name}_correlation_matrix.csv", corr_df.to_csv())
                except Exception:
                    pass
        elif section_id == "gemini" and gemini_analysis_text:
            zf.writestr(f"{raw_name}_gemini_ai_analysis.txt", str(gemini_analysis_text))

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        readme = (
            f"CSV FILE ANALYSER - SECTION EXPORT ({section_id.upper()})\n"
            f"Dataset: {data_loader.get_filename()}\n"
            f"Section: {section_id}\n"
            f"Generated: {now_str}\n\n"
            f"This package contains all generated reports and extracted data for this section.\n"
        )
        zf.writestr(f"README_{section_id.upper()}.txt", readme)

    zip_buf.seek(0)
    return zip_buf.getvalue()

