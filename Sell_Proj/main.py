from data_loader import load_network_data
from evaluate import evaluate_model
from preprocessing import preprocess_data, scale_features
from split_data import split_dataset
from svm_model import train_svm


def main():
  data_path = 'data_NSL_KDD/NSL-KDD_Security_Network/KDDTrain+_20Percent.txt'

  print('1. Loading dataset...')
  df = load_network_data(data_path)

  print('2. Preprocessing & Feature selection...')
  X, y = preprocess_data(df)

  print('3. Splitting dataset...')
  X_train, X_test, y_train, y_test = split_dataset(X, y)

  print('4. Scaling features...')
  X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

  print('5. Training SVM model...')
  model = train_svm(X_train_scaled, y_train)

  print('6. Evaluating model...')
  evaluate_model(model, X_test_scaled, y_test)


if __name__ == '__main__':
  main()