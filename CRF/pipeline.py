
import CRF.feature_extraction as feature_extractor
import CRF.modeling as modeler
from pprint import  pprint


TRAIN_TEST_SPLIT = (0.8, 0.2)
DATA_PATH = "../Data/Test/data.json"

#Extract Features
X_train, X_test, y_train, y_test = feature_extractor.construct_train_test_dataset(TRAIN_TEST_SPLIT, DATA_PATH)

#Fit Model
eval_labels = ['PER', 'LOC']
crf = modeler.get_best_model(X_train, y_train, eval_labels)

#Evaluate
y_pred = crf.predict(X_test)
print (modeler.get_classification_report(y_test, y_pred, eval_labels))

#Inspect
modeler.print_state_features(crf)
modeler.print_transitions(crf)


