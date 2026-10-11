# CSV Data Analyzer

A powerful data analysis application built with Python, Streamlit, and Google's Gemini AI that allows users to upload their own CSV files and get comprehensive analysis results.

## Features

- **Data Overview**: Quick summary of data structure, content, and basic statistics
- **Visualizations**: Interactive charts and plots including histograms, box plots, correlation heatmaps, and scatter plots
- **Statistical Analysis**: Detailed statistical analysis with descriptive statistics and group-by analysis
- **EDA & Data Quality**: Dataset and column profiles, missingness, duplicate checks, quality scoring, and generated reports
- **Advanced Statistics**: Pairwise tests, effect sizes, correlation methods, and multiple-comparison adjustment
- **Advanced Visualizations**: Distribution, residual, grouped, and missingness plots
- **Time-Series Analysis**: Date/value profiling, trend summaries, and optional resampling
- **Gemini AI Analysis**: AI-powered insights and recommendations (requires Gemini API key)

## Requirements

- Python 3.6+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Scikit-learn (used by the existing data-cleaning utilities)
- Google Generative AI Python SDK

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Upload your CSV file using the sidebar
3. Optionally, add your Gemini API key for AI-powered analysis
4. Explore EDA, quality, visualization, statistical, and time-series tabs to analyze your data

## Google sign-in setup

This app requires users to sign in with Google before they can access uploaded data or analysis tools.

1. In [Google Cloud Console](https://console.cloud.google.com/), create or select a project and configure the Google Auth Platform OAuth consent screen. For local testing, use the External audience and add the Google accounts that should be allowed as test users.
2. Create an OAuth client ID with application type **Web application**. Add `http://localhost:8501/oauth2callback` as an authorized redirect URI.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
4. Put your OAuth client ID and client secret in `.streamlit/secrets.toml`. Replace `cookie_secret` with a long random value. Keep this file private; it is excluded by `.gitignore`.
5. Install dependencies and start the app:

   ```powershell
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

6. Open `http://localhost:8501`, sign in with an allowed Google account, and use **Sign out** in the sidebar to end the app session.

For deployment, register the deployed app's `https://.../oauth2callback` URL with Google and update `redirect_uri` and secrets in the hosting platform's secrets manager. Never commit OAuth credentials.

## Sample Data

A sample dataset (`sample_data.csv`) is included for testing purposes.

[![Watch the Demo](./csv_analyzer_demo.png)](https://drive.google.com/file/d/1k-lLvTqXvn6jVPf7veY1Jd1Q4DgUGHo1/view?usp=sharing)


## Login history and activity audit

After Google sign-in, the app records a successful login and shows the user's previous login time in the sidebar. It records sign-in starts, sign-outs, CSV upload success/failure (row and column counts only), analysis-tab navigation, Gemini analysis runs, Gemini key configuration, and downloads. It does not store CSV rows, Gemini prompts/results, OAuth credentials, or the Gemini API key.

Events are stored locally in .streamlit/activity_log.sqlite3. Set CSV_ANALYSER_ACTIVITY_DB to a persistent database path when you need a different location, especially for deployments with ephemeral filesystems. The database is excluded from Git.

The app owner can review all events, while other signed-in users see only their own history. The current owner allowlist is configured in .streamlit/secrets.toml under [activity] admin_emails.
