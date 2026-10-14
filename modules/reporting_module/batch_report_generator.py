"""Build one PDF analysis report for each uploaded CSV and package them."""

from __future__ import annotations

import io
import html
import re
import zipfile
from pathlib import PurePosixPath

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _safe_stem(filename: str) -> str:
    stem = PurePosixPath(filename.replace("\\", "/")).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "dataset"


def _make_pdf(frame: pd.DataFrame, filename: str, gemini_api_key: str | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, title=f"Analysis report: {filename}")
    styles = getSampleStyleSheet()
    story = [Paragraph("CSV Analysis Report", styles["Title"]), Paragraph(f"Dataset: {html.escape(filename)}", styles["Heading2"]), Spacer(1, 10)]
    overview = [
        ["Rows", f"{len(frame):,}", "Columns", str(frame.shape[1])],
        ["Missing cells", f"{int(frame.isna().sum().sum()):,}", "Duplicate rows", f"{int(frame.duplicated().sum()):,}"],
        ["Numeric columns", str(len(frame.select_dtypes(include="number").columns)), "Text/category columns", str(len(frame.select_dtypes(include=["object", "category"]).columns))],
    ]
    table = Table(overview, colWidths=[1.4 * inch, 1.5 * inch, 1.7 * inch, 1.4 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([Paragraph("Dataset overview", styles["Heading1"]), table, Spacer(1, 12)])

    schema = [["Column", "Inferred type", "Missing", "Unique"]]
    for column in frame.columns:
        schema.append([str(column)[:40], str(frame[column].dtype), str(int(frame[column].isna().sum())), str(int(frame[column].nunique()))])
    story.append(Paragraph("Column profile", styles["Heading1"]))
    schema_table = Table(schema, repeatRows=1, colWidths=[2.5 * inch, 1.4 * inch, 1.0 * inch, 1.0 * inch])
    schema_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 5)]))
    story.extend([schema_table, Spacer(1, 12)])

    story.append(Paragraph("Data preview", styles["Heading1"]))
    preview = [[str(value)[:70] for value in frame.columns[:8]]]
    preview.extend([[str(value)[:70] for value in row] for row in frame.iloc[:8, :8].itertuples(index=False, name=None)])
    preview_table = Table(preview, repeatRows=1, colWidths=[6.4 * inch / max(len(preview[0]), 1)] * len(preview[0]))
    preview_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 7), ("PADDING", (0, 0), (-1, -1), 4)]))
    story.extend([preview_table, Spacer(1, 12)])

    story.append(Paragraph("EDA and data quality", styles["Heading1"]))
    missing = frame.isna().sum()
    quality_rows = [["Check", "Result"]]
    quality_rows.append(["Rows containing missing values", str(int(frame.isna().any(axis=1).sum()))])
    quality_rows.append(["Exact duplicate rows", str(int(frame.duplicated().sum()))])
    quality_rows.append(["Columns with missing values", ", ".join(str(c) for c in missing[missing > 0].index) or "None"])
    mixed = [str(c) for c in frame.select_dtypes(include=["object"]).columns if frame[c].dropna().map(type).nunique() > 1]
    quality_rows.append(["Object columns with mixed Python value types", ", ".join(mixed) or "None detected"])
    quality_table = Table(quality_rows, colWidths=[2.6 * inch, 3.4 * inch])
    quality_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 5)]))
    story.extend([quality_table, Spacer(1, 12)])
    missing_summary = pd.DataFrame({"Column": missing.index.astype(str), "Missing": missing.values, "Missing %": (missing.values / max(len(frame), 1) * 100).round(2)})
    missing_table = Table([[str(v) for v in missing_summary.columns]] + [[str(v) for v in row] for row in missing_summary.itertuples(index=False, name=None)], repeatRows=1)
    missing_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 4)]))
    story.extend([Paragraph("Missing-value detail", styles["Heading2"]), missing_table, Spacer(1, 12)])

    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        story.append(Paragraph("Numeric summary", styles["Heading1"]))
        summary = numeric.describe().T.reset_index().rename(columns={"index": "Column"}).round(3)
        summary_data = [[str(value) for value in summary.columns]]
        summary_data.extend([[str(value) for value in row] for row in summary.itertuples(index=False, name=None)])
        summary_table = Table(summary_data, repeatRows=1, colWidths=[1.25 * inch] + [0.62 * inch] * (len(summary_data[0]) - 1))
        summary_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 7), ("PADDING", (0, 0), (-1, -1), 4)]))
        story.extend([summary_table, Spacer(1, 12)])

        outlier_rows = [["Numeric column", "IQR lower bound", "IQR upper bound", "Flagged rows"]]
        for column in numeric.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
            count = int(((values < low) | (values > high)).sum())
            outlier_rows.append([str(column), f"{low:.3g}", f"{high:.3g}", str(count)])
        story.append(Paragraph("Potential outliers (1.5 × IQR rule)", styles["Heading2"]))
        outlier_table = Table(outlier_rows, repeatRows=1)
        outlier_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 5)]))
        story.extend([outlier_table, Spacer(1, 12)])

        categorical_for_groups = frame.select_dtypes(include=["object", "category", "bool"]).columns
        if len(categorical_for_groups):
            group_col = categorical_for_groups[0]
            value_col = numeric.columns[0]
            grouped = frame.groupby(group_col, dropna=False)[value_col].agg(["count", "mean", "median"]).sort_values("mean", ascending=False).head(15).reset_index()
            story.append(Paragraph(f"Group analysis: {value_col} by {group_col}", styles["Heading2"]))
            grouped_data = [[str(v) for v in grouped.columns]] + [[str(v) for v in row] for row in grouped.itertuples(index=False, name=None)]
            grouped_table = Table(grouped_data, repeatRows=1)
            grouped_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 4)]))
            story.extend([grouped_table, Spacer(1, 10)])

        if len(numeric.columns) >= 2:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
            ax.set_title("Numeric correlation heatmap")
            fig.tight_layout()
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format="png", dpi=130)
            plt.close(fig)
            image_buffer.seek(0)
            story.extend([Paragraph("Visualizations", styles["Heading1"]), Image(image_buffer, width=6.3 * inch, height=3.8 * inch), Spacer(1, 8)])
        for column in numeric.columns[:4]:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            fig, (ax, box_ax) = plt.subplots(1, 2, figsize=(8, 2.8))
            ax.hist(values, bins="auto", color="#7064e8", edgecolor="white")
            ax.set(title=f"Distribution: {column}", xlabel=str(column), ylabel="Count")
            sns.boxplot(y=values, ax=box_ax, color="#55bdb4")
            box_ax.set(title=f"Box plot: {column}", ylabel=str(column))
            fig.tight_layout()
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format="png", dpi=130)
            plt.close(fig)
            image_buffer.seek(0)
            story.extend([Image(image_buffer, width=6.4 * inch, height=2.25 * inch), Spacer(1, 8)])

        categorical = frame.select_dtypes(include=["object", "category", "bool"])
        for column in categorical.columns[:3]:
            counts = categorical[column].astype("string").value_counts().head(15)
            if counts.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 3.3))
            sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="#55bdb4")
            ax.set(title=f"Category counts: {column}", xlabel="Count", ylabel=str(column))
            fig.tight_layout()
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format="png", dpi=130)
            plt.close(fig)
            image_buffer.seek(0)
            story.extend([Image(image_buffer, width=6.4 * inch, height=3.0 * inch), Spacer(1, 8)])

        if len(numeric.columns) >= 2:
            x_col, y_col = numeric.columns[:2]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.scatterplot(data=frame, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"Scatter plot: {x_col} vs {y_col}")
            fig.tight_layout()
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format="png", dpi=130)
            plt.close(fig)
            image_buffer.seek(0)
            story.extend([Image(image_buffer, width=6.4 * inch, height=3.0 * inch), Spacer(1, 8)])

    for column in frame.select_dtypes(include=["object", "category"]).columns[:3]:
        counts = frame[column].astype("string").value_counts(dropna=False).head(10)
        if counts.empty:
            continue
        story.append(Paragraph(f"Top values: {column}", styles["Heading2"]))
        values_table = Table([["Value", "Count"]] + [[str(value)[:70], str(int(count))] for value, count in counts.items()], colWidths=[4.8 * inch, 1.2 * inch])
        values_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 5)]))
        story.extend([values_table, Spacer(1, 8)])

    story.append(Paragraph("Statistical analysis", styles["Heading1"]))
    story.append(Paragraph("Includes descriptive summaries, missing-value counts, and IQR-based potential outlier flags above. Correlations describe association and do not establish causation.", styles["BodyText"]))

    story.append(Paragraph("Gemini AI analysis", styles["Heading1"]))
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-3.8-flash")
            prompt = ("Summarize this CSV dataset and provide concise observations, data-quality caveats, "
                      "and useful next analyses. Do not claim causation.\n\n"
                      f"Columns/types: {frame.dtypes.astype(str).to_dict()}\n"
                      f"Sample rows: {frame.head(5).to_dict(orient='records')}\n"
                      f"Numeric summary: {numeric.describe().to_dict() if not numeric.empty else 'No numeric columns'}")
            response = model.generate_content(prompt)
            story.append(Paragraph(html.escape(response.text or "Gemini returned an empty response.").replace("\n", "<br/>"), styles["BodyText"]))
        except Exception as exc:
            story.append(Paragraph(f"Gemini analysis was unavailable: {html.escape(str(exc))[:300]}", styles["BodyText"]))
    else:
        story.append(Paragraph("Not run. Gemini is optional and no API key was provided for this batch.", styles["BodyText"]))

    document.build(story)
    return output.getvalue()


def generate_batch_report_zip(uploaded_files, gemini_api_key: str | None = None) -> tuple[bytes, list[dict]]:
    """Generate a PDF report for each valid CSV and return a ZIP plus statuses."""
    output = io.BytesIO()
    results = []
    used_names = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for uploaded in uploaded_files:
            filename = str(getattr(uploaded, "name", "dataset.csv"))
            try:
                if not filename.lower().endswith(".csv"):
                    raise ValueError("Not a CSV file; report generation skipped")
                frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                if frame.empty or frame.shape[1] == 0:
                    raise ValueError("CSV contains no data rows or columns")
                pdf = _make_pdf(frame, filename, gemini_api_key=gemini_api_key)
                stem = _safe_stem(filename)
                report_name = f"{stem}_analysis_report.pdf"
                suffix = 2
                while report_name.casefold() in used_names:
                    report_name = f"{stem}_{suffix}_analysis_report.pdf"
                    suffix += 1
                used_names.add(report_name.casefold())
                archive.writestr(report_name, pdf)
                results.append({"file": filename, "status": "success", "report": report_name})
            except Exception as exc:
                results.append({"file": filename, "status": "failed", "error": str(exc)})

        manifest = ["CSV batch report results", ""]
        for result in results:
            if result["status"] == "success":
                manifest.append(f"SUCCESS: {result['file']} -> {result['report']}")
            else:
                manifest.append(f"FAILED: {result['file']} -> {result['error']}")
        archive.writestr("batch_report_manifest.txt", "\n".join(manifest) + "\n")
    return output.getvalue(), results
