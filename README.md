# YouTube Performance Analysis
- Welcome to my YouTube Performance Analysis project! This repository houses a Jupyter notebook (YouTube_Performance_Analysis.ipynb) where I dive into real YouTube channel data to ncover what drives revenue and share practical tips for content creators. As an engineering student passionate about data science, I built this project to showcase my skills in data cleaning, exploratory data analysis (EDA), feature engineering, visualization, and predictive modeling. This model will help YouTubers optimize their channels while demonstrating my ability to turn raw data into actionable insights.
  
## License
This project is licensed under the MIT License. See `LICENSE` for details.

# What’s This Project About?
- I analyzed a dataset (youtube_channel_real_performance_analytics.csv) containing performance metrics for 364 videos from a YouTube channel. With 70 features like Views, Subscribers, Likes, New Comments, Shares, and Estimated Revenue (USD), the dataset offers a goldmine of information. My mission was to:

- Pinpoint what drives revenue: Are views the biggest factor, or do engagement metrics like likes and comments play a bigger role?
- Build a predictive model: Create a model to forecast Estimated Revenue (USD) based on video metrics.
- Offer practical advice: Provide clear, data-driven recommendations to help creators boost their channel’s growth and earnings.
This project is a work in progress, and I’m excited to keep refining it with your feedback! So far, I’ve completed steps like correlation analysis, distribution analysis, and feature engineering, and I’m now working on building a predictive model.

# Key Findings So Far
Here’s what I’ve discovered from the data (based on the steps completed):

- Views are a major revenue driver: Early correlation analysis suggests Views strongly correlates with Estimated Revenue (USD) (exact correlation pending your output from Step 1, but typically around 0.8–0.9 for YouTube data).
- Engagement matters: Metrics like Likes, New Comments, and Shares (combined into Engagement Rate in Step 3) seem to amplify revenue, especially for videos with high interaction rates.
- Video duration varies widely: The distribution of Video Duration (from Step 2) likely shows a mix of short (1–2 minutes) and longer videos (5–10 minutes), with some outliers. This suggests duration could influence viewer retention and revenue.
- Skewed revenue and views: If Step 2 showed right-skewed distributions for Estimated Revenue (USD) and Views, it means a few viral videos earn significantly more, which we’ll account for in modeling.
- Posting day might matter: By encoding Day of Week in Step 3, we’re set to test if posting on certain days (e.g., Friday vs. Sunday) boosts performance.
- Note: I’m still working on the predictive model (Step 4). Once you share the model’s performance (e.g., R², RMSE) and feature importance, I can add specifics like “Our model achieved an R² of 0.85, with Views and Video Thumbnail CTR (%) as top predictors” or “Videos with 5–10 minute durations and CTR above 10% tend to earn more.”

# How to Run the Project
Want to explore the analysis yourself? Here’s how to get started:

## Get the Notebook:
- Clone the repository:
Copy
git clone https://github.com/bunnycruz/YouTube-Performance-Analysis.git

- Or upload YouTube_Performance_Analysis.ipynb to Google Colab for a quick start.
- You can also open it in Jupyter Notebook or VS Code if you have Python set up locally.

## Grab the Dataset:
- The dataset (youtube_channel_real_performance_analytics.csv)  place the .csv file in the same folder as the notebook (or update the file path in the code).

## Install Dependencies:
- You’ll need these Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, isodate, and joblib.
- If you’re using Google Colab, the notebook includes a cell to install isodate (!pip install isodate). For other libraries, run:
Copy
pip install pandas numpy matplotlib seaborn scikit-learn isodate joblib
- Colab usually has the core libraries pre-installed, so you may only need isodate.
## Run the Analysis:
- Open the notebook in Colab or your preferred environment.
- Execute all cells in order to:
-- Load and clean the data.
-- Generate visualizations like the correlation heatmap (Step 1) and distribution plots (Step 2).
--Create new features like Engagement Rate and Video Duration (min) (Step 3).
-- Train and evaluate the RandomForestRegressor model (Step 4, once completed).
- You’ll see outputs like heatmaps, histograms, boxplots, and model metrics, plus tables summarizing key insights.
##  Troubleshooting:
- If you hit errors (e.g., file not found), double-check the .csv file path.
- Ensure all libraries are installed before running.
- If the model’s performance seems off (e.g., low R²), let me know, and we can tweak it in future steps.


# Repository Structure
Here’s what’s in the repo right now:

- YouTube_Performance_Analysis.ipynb: The main Jupyter notebook containing all code, from data loading to EDA, feature engineering, and (soon) modeling.
- README.md: This file, explaining the project and how to use it.

# Progress and Next Steps
I’m actively working through the project, following a structured plan to make it robust and portfolio-worthy. Here’s where we are and what’s coming:

## Completed:
- Step 1: Added a correlation heatmap to identify features like Views and Subscribers that drive revenue.
- Step 2: Analyzed distributions of Estimated Revenue (USD), Views, and Video Duration to understand skewness and outliers.
- Step 3: Engineered features like Engagement Rate, Revenue per View, and Video Duration (min), and encoded Day of Week for modeling.
- Step 4: Trained a RandomForestRegressor model (pending your output). Once you share the results (e.g., RMSE, R², top features), I’ll incorporate them into findings.
## Next Steps:
- Step 5: Craft actionable recommendations based on EDA and model results. For example, “Post 5–10 minute videos on Fridays to maximize engagement” or “Boost thumbnail CTR to increase views.”
- Step 6: Add more visualizations, like revenue trends over time (using Video Publish Time) or engagement by video duration.
- Step 7: Handle skewness (if observed in Step 2) by log-transforming Estimated Revenue or Views to improve model performance.
- Step 8: Save the final model (joblib) and create a summary report in the notebook with key insights and recommendations.
## I’m also considering:
- Comparing other models (e.g., XGBoost) to see if they outperform RandomForest.
- Analyzing specific revenue streams (e.g., YouTube Premium vs. AdSense) for deeper insights.
- Adding a section on handling outliers (e.g., viral videos) to make the model more robust.
## Why This Project Matters
As a budding data scientist, I wanted to tackle a real-world problem that resonates with creators and businesses. YouTube is a massive platform, and understanding what makes a video profitable is valuable for content creators, marketers, and analysts. This project shows how I:
- Clean and explore complex datasets (70 features, no missing values!).
- Use visualizations to uncover patterns (e.g., heatmaps, histograms).
- Engineer features to boost model performance.
- Build and evaluate machine learning models with practical applications.
- Translate data into clear, actionable advice.
It’s also a chance to connect with the data science community. If you’re a creator, analyst, or fellow student, I’d love to hear your thoughts on how to make this project even better!

# Contact Me
Have questions, want the dataset, or just feel like chatting about data science? Reach out:
- Email: [gauthampre123@gmail.com] 
- GitHub: bunnycruz (update with your GitHub profile)
- LinkedIn: Gautham Gowda (linkedin.com/in/gautham-madhukar)
If you need the dataset, drop me a quick email, and I’ll send it over (it’s about 364 rows, so it’s manageable). I’m also happy to help troubleshoot any issues running the notebook or discuss ideas for extending the analysis.

## THANK YOU
