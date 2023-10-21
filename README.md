# Machine learning interview questions

**Q:** Explain the difference between supervised, unsupervised, and reinforcement learning. Can you provide real-world examples of each?

- *Supervised learning* involves training a model on a labeled dataset, meaning each training example is paired with an output label. A real-world example is spam classification for emails.
- *Unsupervised learning* finds hidden patterns or intrinsic structures in input data, which is unlabeled. An example is market basket analysis in retail, where you identify product items that frequently co-occur in transactions.
- *Reinforcement learning* is about taking suitable action to maximize reward in a particular situation; it is employed in various fields, especially in robotics for task automation, and in gaming, e.g., AlphaGo.

---

**Q:** How do you handle missing or corrupted data in a dataset?

You have several options: 
- Imputing data using methods like mean/mode/median imputation, predictive modeling, or using algorithms like k-Nearest Neighbors;
- Removing data, which involves deleting rows/columns with missing data; 
- Using algorithms that support missing values, like XGBoost.
It's also crucial to understand the source of missing or corrupt data and whether the data is missing at random or if it's a systematic loss.

---

**Q:** What is the bias-variance trade-off in machine learning?

Bias is the simplifying assumptions made by a model to make the target function easier to learn. Variance is the amount that the estimate of the target function will change given different training data. High bias can cause an algorithm to miss relevant relations between features and target outputs (underfitting); high variance can cause overfitting, which is modeling the random noise in the training data. The trade-off is about finding the right balance so that the model generalizes well to new data.

---

**Q:** Describe how a decision tree works and how you would prevent overfitting.

A decision tree makes decisions by splitting data based on certain conditions, creating a tree-like model of decisions. To prevent overfitting:
- Prune the tree by removing branches that use features with low importance;
- Set a minimum number of samples required at a leaf node or setting the maximum depth of the tree;
- Use ensemble methods like Random Forest or Gradient Boosting.

---

**Q:** Explain the purpose of a training set, a validation set, and a test set in a machine learning model.

- The *training set* is used to train the model's parameters.
- The *validation set* is used to provide an unbiased evaluation of the model fit during the tuning of the model's hyperparameters and can be used for regularization to avoid overfitting. 
- The *test set* is used to provide an unbiased evaluation of the final model fit. It's not used during training and thus serves as a final judge on the model's performance.

---

**Q:** What is cross-validation, and why is it useful?

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The most common method is k-fold cross-validation, where the dataset is divided into k subsets, and the model is trained k times, each time using a different subset as the test set and the remaining data as the training set. This method is useful because it provides a robust estimate of the model's performance.

---

**Q:** Can you explain what precision and recall are? In what scenarios might you prioritize one over the other?

- *Precision* is the ratio of correctly predicted positive observations to the total predicted positives. High precision relates to low false positive rate.
- *Recall* (Sensitivity) is the ratio of correctly predicted positive observations to all actual positives. 
You'd prioritize precision in scenarios where false positives are more concerning (e.g., email spam detection) and recall in scenarios where false negatives are more concerning (e.g., disease detection).

---

**Q:** What are some common feature selection methods used in machine learning?

Common methods include:
- Statistical methods (like Chi-Squared);
- Recursive Feature Elimination (RFE);
- Using feature importance from tree-based classifiers;
- L1 (Lasso) regularization;
- Correlation matrices with heatmap.
Feature selection helps in reducing overfitting, improving accuracy, and reducing training time.

---

**Q:** Explain what regularization is and why it is useful.

Regularization adds a penalty to the different parameters of the machine learning model to reduce the freedom of the model, hence avoiding overfitting. This is done by adding a complexity term to the objective function. L1 regularization (Lasso) adds a penalty equal to the absolute value of the magnitude of coefficients, and L2 regularization (Ridge) adds a penalty equal to the square of the magnitude of coefficients.

---

**Q:** How do ensemble methods work, and why might they be preferred over individual models?

Ensemble methods combine multiple machine learning algorithms to obtain better predictive performance. Techniques include:
- Bagging (e.g., Random Forest), which builds multiple models (typically of the same type) from different subsamples of the training dataset.
- Boosting (e.g., XGBoost, AdaBoost), which trains models sequentially, each trying to correct the errors from the previous one.
- Stacking, which involves training a learning algorithm to combine the predictions of several other learning algorithms.
Ensemble methods are often more robust and accurate as they combine the strengths of multiple models and can deliver superior results on complex problems.


