# 📊 CSV DATA ANALYSIS REPORT

## Dataset Summary
- **Total Records**: 10
- **Total Columns**: 8
- **Data Quality**: 90-100% complete (1 missing phone value)

---

## 📈 Key Findings

### 1. **Demographics**
| Metric | Value |
|--------|-------|
| Age Mean | 36.2 years |
| Age Range | 27–52 years |
| Age Std Dev | 8.16 years |
| Median Age | 35 years |

**Age Distribution**: 
- 20s: 2 people (Jane, Alice)
- 30s: 5 people (John, Alice, David, Emma, Henry)
- 40s-50s: 3 people (Bob, Charlie, Grace)

### 2. **Geographic Distribution**
| Country | Count | Percentage |
|---------|-------|------------|
| United States | 6 | 60% |
| United Kingdom | 4 | 40% |

**US Cities**: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia
**UK Cities**: London, Manchester, Birmingham, Liverpool

### 3. **Data Quality Issues** ⚠️
| Issue | Count | Severity |
|-------|-------|----------|
| Missing Phone Numbers | 1 (10%) | ⚠️ Medium |
| Invalid Email Formats | 2 | ⚠️ Medium |
| Invalid Websites | 2 | ⚠️ Medium |

**Problematic Records**:
- **Bob Johnson**: Invalid email `bob.johnson@invalid`, malformed website `not-a-url`
- **Frank Miller**: Invalid email format, numeric phone, invalid website

### 4. **Categorical Breakdown**

**Email Validation**: 8/10 valid (80%)
- Invalid: bob.johnson@invalid, frank@invalid-email

**Website URLs**: 8/10 valid URLs (80%)
- Issues: "not-a-url", "invalid-url"

**Phone Coverage**: 9/10 records (90%)
- Missing: Charlie Brown

---

## 📊 Visualizations Generated

### Chart 1: **Numeric Distributions** (01_numeric_distributions.png)
- Histogram & box plot of age distribution
- Shows age clustering in 28-42 range
- Identifies outlier at age 52

### Chart 2: **Categorical Distributions** (02_categorical_distributions.png)
- Top values for all categorical fields
- Country, city, and domain distributions

### Chart 3: **Missing Data Analysis** (03_missing_data.png)
- Phone field has 10% missing rate
- All other fields 100% complete

### Chart 4: **Geographic Analysis** (05_geographic_analysis.png)
- Pie chart: US (60%) vs UK (40%)
- Bar chart: Top 10 cities represented

---

## 🔍 Data Quality Recommendations

1. **Phone Validation**: Add format validation (missing Charlie Brown's phone)
2. **Email Cleanup**: Fix or flag invalid emails (Bob Johnson, Frank Miller)
3. **Website Validation**: Implement URL scheme validation
4. **Standardization**: Ensure consistent format for international phone numbers

---

## 📁 Generated Files
- `csv_analysis.py` - Reusable analysis script
- `01_numeric_distributions.png` - Age distribution charts
- `02_categorical_distributions.png` - Categorical field breakdowns
- `03_missing_data.png` - Missing value analysis
- `05_geographic_analysis.png` - Geographic distribution

**Ready to run on any CSV!** Just modify the `CSV_FILE` variable in the script.
