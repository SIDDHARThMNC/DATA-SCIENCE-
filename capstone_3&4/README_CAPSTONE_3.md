# Capstone Project 3: Smart Marketing Prediction System

## 🎯 Business Problem

**ShopEasy E-commerce Company** is facing inefficient marketing campaigns:
- Thousands of daily visitors but low conversion rates
- Marketing budget wasted on users unlikely to purchase
- No way to identify high-value customers
- Need to predict purchase probability for targeted marketing

## 📊 Project Objective

Build an ML pipeline that predicts whether a customer will purchase (1) or not purchase (0) during a website session.

## 🔧 Technical Requirements

### ML Pipeline Components:
1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical features
   - Scale numerical features

2. **Feature Engineering**
   - Numerical features: session_duration, pages_viewed, time_on_site
   - Categorical features: device_type, traffic_source, user_location

3. **Model Training**
   - Multiple model comparison
   - Cross-validation
   - Performance metrics

4. **Model Selection**
   - Best model identification
   - Hyperparameter tuning

5. **Business Insights**
   - Customer segmentation
   - Marketing recommendations
   - ROI estimation

## 📁 Dataset

**Source:** https://github.com/himanshusar123/Datasets  
**Location:** DatasetCapstoneProject3

### Features:
- **Numerical:** Session duration, pages viewed, time on site, etc.
- **Categorical:** Device type, traffic source, location, etc.
- **Target:** Purchase (0 = No, 1 = Yes)

## 🚀 How to Run

1. Download dataset from GitHub repository
2. Place dataset in `capstone_3&4/` folder as `marketing_data.csv`
3. Run the script:
```bash
python capstone_3_smart_marketing_prediction.py
```

## 📈 Expected Outputs

1. **Model Performance Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC Score
   - Cross-validation scores

2. **Visualizations**
   - Target distribution
   - Model comparison
   - Confusion matrix
   - ROC curve
   - Probability distribution

3. **Business Insights**
   - Customer segmentation (High/Medium/Low probability)
   - Marketing recommendations
   - Expected ROI improvement

## 💡 Business Impact

### Marketing Strategy:
- **High Probability (>70%):** Premium discounts + personalized recommendations
- **Medium Probability (40-70%):** Targeted email campaigns
- **Low Probability (<40%):** Minimal spend, retargeting focus

### Expected Benefits:
- 40-60% improvement in marketing efficiency
- Reduced wasted ad spend
- Better customer targeting
- Increased conversion rates

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn:** ML pipeline, models, preprocessing
- **pandas:** Data manipulation
- **numpy:** Numerical operations
- **matplotlib/seaborn:** Visualization
- **GridSearchCV:** Hyperparameter tuning

## 📊 Models Tested

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)

## 🎓 Key Learnings

- Complete ML pipeline implementation
- Handling mixed data types (numerical + categorical)
- Model selection and comparison
- Hyperparameter optimization
- Business-focused ML solutions
- ROI-driven decision making
