import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, output_dir='outputs'):
  os.makedirs(output_dir, exist_ok=True)
  y_pred = model.predict(X_test)

  print('--- Classification Report ---')
  print(
      classification_report(
          y_test, y_pred, target_names=['Normal (0)', 'Port Scan (1)']
      )
  )

  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(6, 4))
  sns.heatmap(
      cm,
      annot=True,
      fmt='d',
      cmap='Blues',
      xticklabels=['Normal', 'Port Scan'],
      yticklabels=['Normal', 'Port Scan'],
  )
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('SVM Port Scan Detection - Confusion Matrix')
  plt.tight_layout()

  save_path = os.path.join(output_dir, 'confusion_matrix.png')
  plt.savefig(save_path, dpi=300)
  print(f'Saved confusion matrix to {save_path}')
  plt.show()