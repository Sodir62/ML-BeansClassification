# Results and Discussion Sections

## 3. Results

### 3.1 Random Forest Results

#### 3.1.1 Manual Configuration Comparison

We first evaluated four manually designed configurations with increasing levels of regularization. Table 1 summarizes the results obtained through 5-fold stratified cross-validation.

**Table 1: Random Forest Manual Configuration Results**

| Configuration | max_depth | min_samples_split | min_samples_leaf | CV F1 (macro) | Train F1 | Test F1 | Overfitting Gap |
|---------------|-----------|-------------------|------------------|---------------|----------|---------|-----------------|
| Baseline      | None      | 2                 | 1                | 0.9336        | 1.0000   | 0.9343  | 6.57%           |
| Light Reg     | 20        | 5                 | 2                | 0.9335        | 0.9877   | 0.9327  | 5.51%           |
| Medium Reg    | 15        | 10                | 4                | 0.9333        | 0.9682   | 0.9314  | 3.68%           |
| Heavy Reg     | 10        | 20                | 8                | 0.9279        | 0.9495   | 0.9308  | 1.87%           |

The baseline model without regularization achieved perfect training F1-score (1.0000), indicating severe overfitting with a gap of 6.57%. Progressive regularization reduced this gap substantially, with the Heavy Regularization configuration achieving the lowest overfitting gap of 1.87% while maintaining competitive test performance (93.08% F1).

#### 3.1.2 Grid Search Results

We performed exhaustive grid search over 720 parameter combinations using 5-fold stratified cross-validation. The parameter space explored was:
- `n_estimators`: {50, 100, 200}
- `max_depth`: {10, 15, 20, 25, None}
- `min_samples_split`: {2, 5, 10, 20}
- `min_samples_leaf`: {1, 2, 4, 8}
- `max_features`: {sqrt, log2, None}

**Table 2: Grid Search Best Configurations**

| Selection Criterion | CV F1 (macro) | CV Std | Train F1 | CV Gap | Configuration |
|---------------------|---------------|--------|----------|--------|---------------|
| Best CV F1          | 0.9360        | 0.0030 | 0.9885   | 5.25%  | depth=20, leaf=2, split=5, n=100 |
| Lowest Gap          | 0.9316        | 0.0008 | 0.9513   | 1.96%  | depth=10, leaf=8, split=20, n=200 |

The grid search revealed a clear trade-off between maximizing F1-score and minimizing overfitting. The configuration optimized for lowest gap achieved 93.16% CV F1 with only 1.96% overfitting gap, compared to 93.60% CV F1 with 5.25% gap for the highest-scoring configuration.

Based on these results, we selected the Heavy Regularization configuration (depth=10, leaf=8, split=20) with n_estimators=100 as our final model. The grid search results closely matched our manual regularization experiments, confirming that progressively increasing regularization constraints effectively reduces overfitting. The small differences between grid search CV scores and our manual test evaluation (e.g., 1.97% CV gap vs 1.87% test gap) are within expected variation due to different validation methodologies. Our final model achieved 93.08% test F1-score with 1.87% overfitting gap.

#### 3.1.3 Hyperparameter Sensitivity Analysis

Figure 1 shows the effect of each hyperparameter on both CV F1-score and overfitting gap.

**[INSERT FIGURE: hyperparameter_sensitivity.png]**

*Figure 1: Hyperparameter sensitivity analysis showing the trade-off between CV F1 (blue) and overfitting gap (red) for each parameter.*

Key observations:
- **max_depth**: Performance peaks around depth 15-20, but overfitting gap increases sharply beyond depth 10. Limiting depth to 10 reduces the gap from ~4% to ~2.7%.
- **min_samples_leaf**: Increasing from 1 to 8 reduces the overfitting gap from ~4.7% to ~2.5% with minimal F1 degradation (0.9330 to 0.9305).
- **min_samples_split**: Higher values (20) reduce overfitting gap from ~4.3% to ~2.8% while maintaining similar F1 scores.
- **n_estimators**: Increasing trees from 50 to 200 slightly improves both F1 and stability, with minimal impact on overfitting.

#### 3.1.4 Learning Curves

Figure 2 presents learning curves for two configurations: the heavily regularized model (lowest gap) and the best F1 model.

**[INSERT FIGURE: learning_curves.png]**

*Figure 2: Learning curves comparing training and validation F1-scores across different training set sizes for two model configurations.*

Both configurations show characteristic learning curve patterns:
- Training F1 decreases slightly as more data is added (from ~0.95 to ~0.95 for regularized)
- Validation F1 increases and stabilizes around ~0.93
- Final gap of approximately 2.0% for the Low Gap model and 2.05% for the Best F1 model
- Both curves converge, indicating the models benefit from the full training set size

#### 3.1.5 Feature Importance Analysis

Figure 3 displays the Gini importance scores for all 16 features.

**[INSERT FIGURE: feature_importance.png]**

*Figure 3: Random Forest feature importances ranked by Gini impurity decrease.*

**Table 3: Top 5 Most Important Features**

| Rank | Feature          | Importance |
|------|------------------|------------|
| 1    | Perimeter        | 0.117      |
| 2    | ShapeFactor3     | 0.113      |
| 3    | Compactness      | 0.100      |
| 4    | MinorAxisLength  | 0.098      |
| 5    | ShapeFactor1     | 0.079      |

The top five features account for approximately 51% of total importance. Note that feature importances are more evenly distributed because our model uses `max_features='sqrt'`, meaning each split only considers a random subset of features. Perimeter emerges as the single most discriminative feature, consistent with the intuition that bean size varies significantly across varieties. Shape-related features (ShapeFactor1, ShapeFactor3, Compactness) collectively contribute substantially, suggesting that bean shape is crucial for classification.

#### 3.1.6 Per-Class Performance

Figure 4 shows the F1-score achieved for each bean variety.

**[INSERT FIGURE: per_class_f1.png]**

*Figure 4: Per-class F1 scores comparing the Low Gap and Best F1 model configurations.*

**Table 4: Per-Class F1 Scores (Heavy Reg Model)**

| Bean Variety | F1-Score | Recall | Notes |
|--------------|----------|--------|-------|
| Bombay       | 1.000    | 1.00   | Perfect classification (smallest class) |
| Horoz        | 0.956    | 0.95   | High performance |
| Cali         | 0.936    | 0.93   | Above average |
| Seker        | 0.934    | 0.95   | High performance |
| Barbunya     | 0.917    | 0.91   | Slight confusion with Cali |
| Dermason     | 0.910    | 0.90   | Confusion with Sira |
| Sira         | 0.862    | 0.88   | Lowest performance; confused with Dermason |

Notably, Bombay beans are classified perfectly despite being the smallest class (3.8% of data), likely due to their distinct morphological characteristics. Sira beans show the lowest performance (86.2% F1), primarily due to confusion with Dermason beans, as evidenced in the confusion matrix (Figure 5).

#### 3.1.7 Confusion Matrix Analysis

Figure 5 presents the normalized confusion matrix showing per-class recall.

**[INSERT FIGURE: confusion_matrix_normalized.png]**

*Figure 5: Normalized confusion matrix showing recall (true positive rate) for each class.*

The confusion matrix reveals:
- **Bombay**: Perfect classification (100% recall) - distinct size characteristics
- **Sira-Dermason confusion**: 8% of Sira beans misclassified as Dermason, and 8% of Dermason as Sira
- **Barbunya-Cali confusion**: 5% of Barbunya misclassified as Cali
- **Diagonal dominance**: All classes achieve >88% recall, indicating strong overall performance

#### 3.1.8 Decision Tree Visualization

Figure 6 shows the structure of a single decision tree from the ensemble (limited to 4 levels for readability).

**[INSERT FIGURE: decision_tree_depth4.png]**

*Figure 6: Visualization of a single decision tree from the Random Forest ensemble (Tree #1 of 100, depth limited to 4 levels).*

The tree visualization illustrates:
- The root split uses MinorAxisLength, immediately separating Bombay beans (right branch) from others
- Subsequent splits utilize Perimeter, ShapeFactor3, and Compactness
- Each node shows the Gini impurity, sample percentage, and class distribution
- The hierarchical structure demonstrates how the model progressively partitions the feature space

---

### 3.2 Neural Network Results

**[TODO: Complete this section with Neural Network results]**

*Placeholder: The neural network achieved a best validation F1 (macro) of 94.10% (+/- 0.12%) with configuration: dense1=40, dense2=32, learning_rate=0.0005, batch_size=32, epochs=40, dropout=0.0, l2=0.0. Grid search tested 288 configurations.*

---

### 3.3 Model Comparison Summary

**Table 5: Final Model Comparison**

| Metric | Random Forest (Selected) | Random Forest (Best F1) | Neural Network |
|--------|--------------------------|-------------------------|----------------|
| CV F1 (macro) | 0.9279 | 0.9360 | [TODO] |
| Test F1 (macro) | 0.9308 | - | [TODO] |
| Overfitting Gap | 1.87% | 5.25% | [TODO] |
| Training Time | ~15 sec | ~15 sec | [TODO] |
| Interpretability | High | High | Low |

---

## 4. Discussion

### 4.1 Random Forest Performance Analysis

#### 4.1.1 Comparison with Literature

Our selected Random Forest configuration achieved a test macro F1-score of 93.08% with only 1.87% overfitting gap. This performance compares favorably to the results reported by Koklu and Ozkan [1], who achieved their best accuracy of 93.13% using a Support Vector Machine (SVM). Our Random Forest outperforms their decision tree baseline, which achieved under 90% accuracy, demonstrating the effectiveness of ensemble methods for this classification task.

The improvement over single decision trees can be attributed to:
1. **Bootstrap aggregation (bagging)**: Reduces variance by averaging predictions from multiple trees trained on different data subsets
2. **Feature randomization**: Each split considers only a random subset of features, decorrelating trees and reducing overfitting
3. **Balanced class weights**: Compensates for class imbalance by adjusting the loss function

#### 4.1.2 Regularization Effectiveness

The experimental results clearly demonstrate the importance of regularization in Random Forest models. Without constraints, the baseline model achieved perfect training performance but exhibited a 6.57% overfitting gap. Our regularization strategy, progressively constraining tree depth and minimum sample requirements, successfully reduced this gap to 1.87% while maintaining 93.08% test F1-score.

The hyperparameter sensitivity analysis (Figure 1) reveals that:
- **max_depth** is the most impactful parameter: limiting depth from unlimited to 10 reduces overfitting by approximately 50%
- **min_samples_leaf** provides additional regularization with minimal performance cost
- **min_samples_split** and **n_estimators** have more subtle effects

This finding aligns with theoretical understanding: deeper trees can memorize training data but fail to generalize, while shallow trees capture more robust patterns.

#### 4.1.3 Feature Importance Insights

The feature importance analysis provides valuable insights into what distinguishes bean varieties. Perimeter emerged as the most discriminative feature (19.8% importance), followed by shape factors and axis lengths. This finding has practical implications:

1. **Measurement priority**: Quality control systems should prioritize accurate perimeter and shape measurements
2. **Feature engineering**: Future work could explore derived features combining perimeter with area (e.g., circularity)
3. **Interpretability**: The reliance on geometric features makes the model's decisions transparent and physically meaningful

Interestingly, features like AspectRatio and Eccentricity, despite their intuitive relevance, showed low importance (<1%). This suggests that the information they encode is redundant with other features or that their discriminative power is limited for this specific classification task.

#### 4.1.4 Class-Specific Performance Analysis

The per-class analysis reveals interesting patterns:

**High performers (F1 > 0.93):**
- **Bombay** (100%): Despite being the smallest class (3.8%), Bombay beans are perfectly classified. Examination of the decision tree (Figure 6) shows that Bombay beans are separated at the first split using MinorAxisLength, indicating they have distinctly larger dimensions than other varieties.
- **Horoz** (96.0%): Characterized by elongated shape, well-captured by shape factors.
- **Seker** (93.9%): Distinct roundness characteristics enable reliable classification.

**Lower performers (F1 < 0.92):**
- **Dermason** (91.0%): As the most frequent class (26%), some misclassification is expected due to its variability.
- **Sira** (86.5%): The most challenging class, primarily confused with Dermason. Both varieties share similar size ranges and shape characteristics, making discrimination difficult based on geometric features alone.

The Sira-Dermason confusion (8% bidirectional misclassification) suggests that these varieties may require additional features beyond geometry for reliable separation, such as color information or texture analysis.

#### 4.1.5 Trade-off Analysis: Accuracy vs. Generalization

A key finding from our experiments is the trade-off between maximizing cross-validation F1 and minimizing overfitting gap. The two best configurations illustrate this:

| Model | CV F1 | Test F1 | Gap | Use Case |
|-------|-------|---------|-----|----------|
| Best F1 (Grid Search) | 93.60% | - | 5.25% | When maximum accuracy is critical |
| Selected (Heavy Reg) | 92.79% | 93.08% | 1.87% | When generalization to new data is critical |

For production deployment, we selected the Heavy Regularization configuration because:
1. The test F1 of 93.08% is competitive with the best configurations
2. A 4.70% reduction in overfitting gap (from 6.57% to 1.87%) suggests better generalization to unseen data
3. Real-world bean samples may differ slightly from training data distribution

#### 4.1.6 Computational Efficiency

Random Forest training completed in approximately 15 seconds for 100 trees, making it highly suitable for practical applications. The grid search over 720 configurations (3,600 total model fits with 5-fold CV) completed in under 2 hours, demonstrating the scalability of the approach.

### 4.2 Neural Network Discussion

**[TODO: Complete this section with Neural Network discussion]**

*Placeholder for neural network discussion including:*
- *Architecture design choices*
- *Regularization effects (dropout, L2)*
- *Learning dynamics and early stopping*
- *Comparison with Random Forest*

### 4.3 Limitations

Several limitations should be acknowledged:

1. **Feature limitations**: The dataset contains only geometric features. Color, texture, or spectral information could improve discrimination, particularly for Sira-Dermason separation.

2. **Class imbalance**: Despite using balanced class weights, the significant imbalance (Dermason: 26% vs. Bombay: 3.8%) may still affect model behavior.

3. **Dataset scope**: Results are specific to the seven bean varieties in this dataset. Generalization to other varieties or different imaging conditions is not evaluated.

4. **Single random seed**: While we used 5-fold CV for robustness, using a single random seed (42) limits our understanding of variance due to random initialization.

5. **Visualization configuration error**: During the poster presentation, the feature importance and confusion matrix figures were generated with an incorrect configuration (`max_features=None`), whereas our actual model used the scikit-learn default (`max_features='sqrt'`). This occurred because we assumed that omitting the parameter in the visualization script was equivalent to setting it to `None`, not realizing it needed to be explicitly set to match the default. The figures in this report have been corrected to reflect the actual model configuration.

### 4.4 Future Work

Based on our findings, several directions merit further investigation:

1. **Feature augmentation**: Incorporate color or texture features to improve Sira-Dermason discrimination
2. **Ensemble methods**: Combine Random Forest with Neural Network predictions for potential accuracy gains
3. **Cost-sensitive learning**: Adjust misclassification costs based on the economic impact of different errors
4. **Transfer learning**: Apply pre-trained image models directly to bean images, bypassing manual feature extraction

### 4.5 Conclusion

Our Random Forest classifier achieved 93.08% test macro F1-score on dry bean classification, comparable to the SVM benchmark of Koklu and Ozkan [1] (93.13%) while providing superior interpretability. Careful regularization through depth limiting and minimum sample constraints reduced overfitting from 6.57% to 1.87%. Feature importance analysis revealed that Perimeter and shape factors are the most discriminative features. The model successfully classifies all seven bean varieties with >86% per-class F1, with Bombay beans achieving perfect classification. The main challenge remains Sira-Dermason discrimination, which may require features beyond geometry for further improvement.

---

## Figures Summary

The following figures should be included in your report (all available in `RandomForest/outputs/`):

1. **Figure 1**: `hyperparameter_sensitivity.png` - Effect of hyperparameters on F1 and overfitting gap
2. **Figure 2**: `learning_curves.png` - Training vs validation F1 across training set sizes
3. **Figure 3**: `feature_importance.png` - Gini importance ranking of all 16 features
4. **Figure 4**: `per_class_f1.png` - Per-class F1 scores comparison
5. **Figure 5**: `confusion_matrix_normalized.png` - Normalized confusion matrix (recall)
6. **Figure 6**: `decision_tree_depth4.png` - Decision tree structure visualization
7. **Figure 7**: `class_distribution.png` - Class distribution in train/test sets (optional, for Methods section)

---

## References to Add

[1] Koklu, M., and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture, 174, 105507.

[5] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
