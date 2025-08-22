# German Traffic Signs Classification
Task 6 @Elevvo ML Internship

## Instructions
1. Dataset (Recommended): GTSRB (Kaggle).
2. Classify traffic signs based on their image using deep learning.
3. Preprocess images (resizing, normalization).
4. Train a CNN model to recognize different traffic sign classes.
5. Evaluate performance using accuracy and confusion matrix.

### Bonus:
1. Add data augmentation to improve performance.
2. Compare custom CNN vs. pre-trained model (e.g., MobileNet).

## Dataset
- [German Traffic Signs Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Notebook
- [German Traffic Signs Classification](https://github.com/Asma-Nasr/German-Traffic-Signs-Classification/blob/main/german_traffic_signs.ipynb)

## Saved Model
- [Saved model](https://github.com/Asma-Nasr/German-Traffic-Signs-Classification/tree/main/Saved%20Model)

## Results
![Results](https://github.com/Asma-Nasr/German-Traffic-Signs-Classification/blob/main/output.png)

## Performance Comparison
| Models      | Accuracy | ROC Score |
|-------------|----------|-----------|
| Custom CNN  | 97.84%   | 0.9997    |
| VGG19       | 5.39%    | 0.5001    |
| MobileNet   | 99.29%   | 1.0000    |
