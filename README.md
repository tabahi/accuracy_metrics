# accuracy_metrics
Saving everyone the hassle. Precision, Recall, F1, unweighted, weighted accuracy, confusion matrix, all in one code. Requires ndarray type labels for numpy.

- Option to skip a class in weighting macro averages.
- 
## Use:
```python
import accuracy_metrics #put accuracy_metrics.py in your project dir.

y_true =['M', 'F', 'M', 'F', 'O', 'F', 'M', 'F', 'M', 'O', 'M', 'F', 'O', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'O', 'F', 'M', 'F', 'M', 'F', 'M', 'O', 'O', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'O']
y_pred =['F', 'F', 'M', 'O', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'O', 'F', 'F', 'M', 'O', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'O', 'F', 'F', 'M', 'O', 'F', 'F', 'M', 'M', 'M', 'F', 'O']


results = accuracy_metrics.generate_classification_metrics(y_true, y_pred, skip_label='O', confusion_csv="confusion.csv", precisions_csv="precisions.csv")
print(results)

'''
{'N_counts': [17, 13, 7], 'uar': 43.67, 'war': 43.33, 'precision': 0.405, 'recall': 0.405, 'f1_score': 0.405, 'precision_sk': 0.433, 'recall_sk': 0.419, 'f1_score_sk': 0.426}
'''

```
