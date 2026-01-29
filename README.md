# YelpR8: Business Recommendations System

![Yelp Review Project Banner](./src/docs/banner.png)


## Overview
YelpR8 is a hybrid recommendation system designed to predict user ratings for businesses on Yelp. This project leverages both model-based and item-based collaborative filtering approaches. The model-based component uses an XGBoost regressor with carefully engineered features extracted from Yelp data, while the item-based component computes Pearson similarities between businesses. Finally, a hybrid prediction is obtained by weighting the outputs of both methods. This integrated approach has allowed the system to achieve a competitive RMSE, placing it at rank 9 in the competition.

## Setup & Installation

### Environment Requirements
- **Python Version:** 3.6 (or later, ensuring compatibility with Spark 3.1.2)
- **Scala Version:** 2.12
- **JDK:** 1.8
- **Apache Spark:** 3.1.2 (using Spark RDDs – DataFrame and DataSet are not allowed)
- **External Libraries:**  
  - `xgboost`  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`

> The environment on Google Colab may differ slightly from a local setup, but there is no need to worry—it will still work as expected.

### Installation Steps
1. **Clone the Repository:**
   ```sh
   git clone git@github.com:KayvanShah1/YelpR8-Business-Recommendations-System.git
   cd YelpR8-Business-Recommendations-System
   ```
2. **Install Dependencies:**
   Create a virtual environment (optional) and install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```
> I recommend using Google Colab for this project, as it provides a convenient and hassle-free environment for running the code without requiring extensive local setup.

## Approach and Intuition

The main objective of YelpR8 is to improve recommendation accuracy by combining two complementary approaches:

1. **Model-Based Recommendation:**
   - **Feature Engineering:**  
     Data from multiple sources (user profiles, business details, reviews, tips, and photos) is aggregated. For each user-business pair, we extract features like average star ratings, review counts, user activity metrics, and business characteristics.
   - **XGBoost Regression:**  
     After feature extraction, the system normalizes the data using a min–max scaler. The XGBoost regressor is trained with hyperparameters optimized through local GridSearch. This model is able to capture non-linear relationships in the data, resulting in robust rating predictions.

2. **Item-Based Collaborative Filtering:**
   - **Pearson Similarity:**  
     We compute the Pearson similarity between businesses by comparing the ratings provided by common users. For pairs with few co-ratings, a fallback similarity based on average ratings difference is used.
   - **Neighbour Selection:**  
     For each target prediction, ratings from the top 15 most similar businesses are aggregated (weighted by similarity) to predict the rating.
  
3. **Hybrid Approach:**
   - **Fusion of Predictions:**  
     The final prediction is a weighted combination of the model-based and item-based predictions. A carefully chosen weight (e.g., FACTOR = 0.05222) helps balance the strengths of both methods, leading to improved accuracy overall.

This dual approach leverages the robustness of model-based predictions with the personalized similarity insights from collaborative filtering, resulting in improved RMSE performance.

> [!TIP]
> This repository is designed to provide guidance and insights into building a robust hybrid recommendation system using advanced data mining techniques. Use this project as a learning tool to deepen your understanding and enhance your skills. Always ensure that your work maintains academic integrity and originality—avoid plagiarism and give proper credit where it's due.

## Results

Below is a summary of the system's performance metrics on the validation and test datasets:

| Metric                          | Value (Validation) | Value (Test)    |
|---------------------------------|--------------------|-----------------|
| **RMSE**                        | 0.9773             | 0.9760          |
| **Error Distribution (Count)**  |                    |                 |
| &nbsp;&nbsp;>= 0 and < 1        | 102,076            | 102,162         |
| &nbsp;&nbsp;>= 1 and < 2        | 33,039             | 32,993          |
| &nbsp;&nbsp;>= 2 and < 3        | 6,156              | 6,116           |
| &nbsp;&nbsp;>= 3 and < 4        | 772                | 773             |
| &nbsp;&nbsp;>= 4                | 1                  | 0               |
| **Data Processing Time**        | ~179.5 seconds     | ~183.6 seconds  |
| **Model Training Time**         | ~70.4 seconds      | ~425.0 seconds  |
| **Total Execution Time**        | ~251.5 seconds     | ~622.1 seconds  |

*Note: The error distribution for the test set follows a similar pattern to the validation set. These metrics showcase that the system not only meets but slightly beats the TA’s RMSE threshold of 0.9800.*

> [!NOTE]
> Overall Rank: `12`

## Usage
### Command-Line Execution
The system is executed via Spark using the following command:
```sh
/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit file.py <folder_path> <test_file_name> <output_file_name>
```
- **folder_path:** Directory containing the Yelp datasets (e.g., `yelp_train.csv`, `review_train.json`, `user.json`, `business.json`, etc.)
- **test_file_name:** Path to the validation/test CSV file (e.g., `yelp_val.csv`)
- **output_file_name:** Desired output CSV file path for the predictions

### Example
```sh
spark-submit final.py ./YelpData ./YelpData/yelp_val.csv ./Output/predictions.csv
```

## Future Work

- **Model Improvements:**  
  Experiment with additional ensemble methods or deep learning approaches to further improve prediction accuracy.
- **Real-Time Recommendations:**  
  Extend the system to support near real-time predictions using streaming data.
- **Additional Features:**  
  Incorporate more sophisticated feature engineering (e.g., text mining from reviews) and user personalization.

## Authors
1. [Kayvan Shah](https://github.com/KayvanShah1) | `MS in Applied Data Science` | `University of Southern California`

#### LICENSE
This repository is licensed under the `MIT` License. See the [LICENSE](LICENSE) file for details.

#### Disclaimer

<sub>
The content and code provided in this repository are for educational and demonstrative purposes only. The project may contain experimental features, and the code might not be optimized for production environments. The authors and contributors are not liable for any misuse, damages, or risks associated with the use of this code. Users are advised to review, test, and modify the code to suit their specific use cases and requirements. By using any part of this project, you agree to these terms.
</sub>
