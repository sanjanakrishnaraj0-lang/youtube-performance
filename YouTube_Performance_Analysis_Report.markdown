# Unlocking YouTube Channel Performance Secrets

**Prepared by**: GAUTHAM MADHUKAR\
**GitHub**: https://github.com/bunnycruz/YouTube-Performance-Analysis\
**Date**: May 15, 2025

## Table of Contents

- Executive Summary
- Introduction
- Methodology
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Visualization
- Predictive Modeling
- Recommendations
- Limitations
- Future Work
- Conclusion
- References

## Executive Summary

This project analyzes YouTube channel performance using a dataset of 364 videos to identify drivers of `Estimated Revenue (USD)`. Key findings:

- `Views` drive revenue (correlation).
- Engagement (`Likes`, `Shares`, `New Comments`) boosts earnings.
- 5–10 minute videos with high thumbnail click-through rates (CTR) perform best.

A Random Forest model predicts revenue with an R-squared. Recommendations include optimizing thumbnails and targeting high-CPM content. Access the analysis at https://github.com/bunnycruz/YouTube-Performance-Analysis.

## Introduction

With over 2 billion monthly users \[1\], YouTube is a competitive platform for creators. This project analyzes `youtube_channel_real_performance_analytics.csv` (364 videos, 70 features) to:

- Identify revenue drivers.
- Predict revenue.
- Recommend optimization strategies.

The analysis, conducted in Google Colab, showcases data science skills for a company audience.

## Methodology

The pipeline includes:

1. **Data Cleaning**: Remove invalid data.
2. **Exploratory Data Analysis (EDA)**: Examine correlations.
3. **Feature Engineering**: Create new features.
4. **Visualization**: Plot trends.
5. **Predictive Modeling**: Train a model.

### Tools

- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `isodate`, `joblib`.
- **Environment**: Google Colab.

## Data Cleaning

The dataset was cleaned:

- **Duplicates**: removed.
- **Invalid Values**: Filtered zero/negative `Views` or `Estimated Revenue (USD)`, yielding  rows.
- **Datetime**: Converted `Video Publish Time`.

```python
print(f"Duplicates: {data.duplicated().sum()}")
data = data.drop_duplicates()
data = data[data['Views'] > 0]
data = data[data['Estimated Revenue (USD)'] >= 0]
data['Video Publish Time'] = pd.to_datetime(data['Video Publish Time'])
```

## Exploratory Data Analysis (EDA)

EDA explored `Estimated Revenue (USD)` relationships.

### Correlation Heatmap

- `Views` vs. `Estimated Revenue (USD)`
- `Likes`, `Shares`, `New Comments`
- `Subscribers`, `Video Thumbnail CTR (%)`

### Top Videos

Top 10 videos showed:

- Max revenue
- Views
- Duration
- CTR

| ID | Revenue (USD) | Views | Duration (s) | CTR (%) |
| --- | --- | --- | --- | --- |
| \[ID1\] | \[5000\] | \[1000000\] | \[400\] | \[15.0\] |
| \[ID2\] | \[4500\] | \[900000\] | \[350\] | \[12.5\] |

### Insights

- `Views` drive revenue, amplified by engagement.
- 5–10 minute videos optimize ads.
- High CTR attracts views.

## Feature Engineering

Added features:

- **Revenue per View**: `Estimated Revenue (USD) / Views`.
- **Engagement Rate**: `(Likes + Shares + New Comments) / Views * 100`.

```python
data['Revenue per View'] = data['Estimated Revenue (USD)'] / data['Views']
data['Engagement Rate'] = (data['Likes'] + data['Shares'] + data['New Comments']) / data['Views'] * 100
```

## Data Visualization

### Revenue vs. Views

Shows a linear relationship.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('images', exist_ok=True)

# Heatmap
key_features = ['Views', 'Subscribers', 'Likes', 'Shares', 'New Comments', 'Estimated Revenue (USD)', 'Video Duration', 'Video Thumbnail CTR (%)']
plt.figure(figsize=(10, 8))
sns.heatmap(data[key_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig('images/heatmap.png')
plt.show()

# Revenue Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Revenue (USD)'], bins=50, kde=True, color='green')
plt.title("Revenue Distribution")
plt.xlabel("Revenue (USD)")
plt.ylabel("Frequency")
plt.savefig('images/revenue_distribution.png')
plt.show()

# Revenue vs Views
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Views'], y=data['Estimated Revenue (USD)'], alpha=0.7)
plt.title("Revenue vs Views")
plt.xlabel("Views")
plt.ylabel("Revenue (USD)")
plt.savefig('images/revenue_vs_views.png')
plt.show()
```

## Predictive Modeling

A Random Forest Regressor used features: `Views`, `Subscribers`, `Likes`, `Shares`, `New Comments`, `Engagement Rate`, `Video Duration`, `Video Thumbnail CTR (%)`.

### Performance

- **MSE**
- **R-squared**
- **Cross-Validation**

### Feature Importance

- `Views`
- `Engagement Rate`

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

features = ['Views', 'Subscribers', 'Likes', 'Shares', 'New Comments', 'Engagement Rate', 'Video Duration', 'Video Thumbnail CTR (%)']
X = data[features]
y = data['Estimated Revenue (USD)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"R-squared: {r2_score(y_test, y_pred):.2f}")
```

## Recommendations

1. **Optimize Thumbnails**: Aim for CTR.
2. **Boost Engagement**: Target `Engagement Rate`.
3. **Use 5–10 Minute Videos**: 300–600 seconds.
4. **Target High-CPM Content**: Focus on high `Revenue per 1000 Views`.

## Limitations

- No demographic or category data.
- Outliers skew predictions.
- Excludes algorithm changes.

## Future Work

- Use YouTube API for real-time data.
- Analyze video categories.
- Build a dashboard.

## Conclusion

`Views` and `Engagement Rate` drive revenue, with optimal video lengths and CTR as key factors. The model provides reliable predictions, and recommendations offer actionable strategies. See https://github.com/bunnycruz/YouTube-Performance-Analysis.

## References

\[1\] Brandwatch, "YouTube Statistics," https://www.brandwatch.com/blog/youtube-stats/, Accessed May 2025.