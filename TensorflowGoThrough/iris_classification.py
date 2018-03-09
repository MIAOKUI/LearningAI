import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris


TRAIN_PATH = 'iris_training.csv'
TEST_PATH = 'iris_test.csv'

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

BATCH_SIZE = 100
TRAINS_STEPS = 1000

def load_data(y_name='Species', train_path = TEST_PATH, test_path = TEST_PATH):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

## Create input function
def input_evaluation_set():
    features = {
        'SepalLength': np.array([6.4, 5.0]),
        'SepalWidth': np.array([2.8, 2.3]),
        'PetalLength': np.array([5.6, 3.3]),
        'PetalWidth': np.array([2.2, 1.0])
    }
    labels = np.array([2, 1])
    return features, labels

def input_fn(features, labels, batch_size):
    """
    Input size for training
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()





train, test = load_data(y_name='Species')
train_x, train_y = train
test_x, test_y = test

my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


## instantiated a Estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3
)

## Train the model
classifier.train(
    input_fn=lambda: input_fn(train_x, train_y, BATCH_SIZE),
    steps = TRAINS_STEPS)

## Evaluate the model
eval_result = classifier.evaluate(
    input_fn = lambda : input_fn(test_x, test_y, BATCH_SIZE)
)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
