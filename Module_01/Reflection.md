# Project 1 Reflection: k-NN Credit Card Default Prediction

## Model Performance Summary

| k  | Accuracy | Precision | Recall | F1-Score |
|----|----------|-----------|--------|----------|
| 3  | 0.7758   | 0.4798    | 0.3545 | 0.4077   |
| 7  | 0.8045   | 0.5851    | 0.3499 | 0.4379   |
| 15 | 0.8125   | 0.6310    | 0.3338 | 0.4367   |

## Reflection

**Which value of k gave the best performance and why?**

k = 7 gave the best overall performance. It had the highest F1-score (0.438), which is a good way to measure performance when the data is unbalanced. In this dataset, only about 22% of people actually defaulted, so just looking at accuracy can be misleading. F1-score balances precision and recall together, and k = 7 did the best job of balancing both.

**How did the model's accuracy change as k increased?**

Accuracy went up as k increased, from 0.776 at k = 3, to 0.805 at k = 7, to 0.813 at k = 15. At first this looks like k = 15 is the best choice, but the accuracy increase mostly happened because the model got better at guessing "No Default," which is the more common outcome. It did not actually get much better at catching real defaults.

**Did any value of k seem to overfit or underfit the data?**

Yes. k = 3 showed signs of overfitting because it only looks at a few neighbors, so it can be thrown off by noisy or unusual data points. On the other hand, k = 15 showed signs of underfitting. Its recall was the lowest of the three (0.334), meaning it missed more actual defaulters because it leaned too heavily toward predicting the majority class.

**How would you approach selecting the best k in a real-world application?**

In a real-world setting, I would test many values of k using cross-validation instead of just picking a few numbers to compare. I would also choose the evaluation metric based on what matters most for the problem. For something like credit default prediction, missing a real defaulter (a false negative) is usually more costly than a false alarm, so I would focus on recall or F1-score rather than accuracy alone.
