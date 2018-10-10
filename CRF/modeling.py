
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

def get_best_model(X_train, y_train, labels):
    '''

    :param X_train: Train features
    :param y_train: Train labels
    :param labels: list of all labels to be evaluated
    :return:
    '''
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    return rs.best_estimator_


def get_classification_report(y_test, y_pred, eval_labels):

    return metrics.flat_classification_report(y_test, y_pred, labels=eval_labels, digits=3)


def print_transitions(crf):
    trans_features = Counter(crf.transition_features_).most_common(20)
    print("Top likely transitions:")
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


    trans_features = Counter(crf.transition_features_).most_common()[-20:]
    print("\nTop unlikely transitions:")
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(crf):
    state_features = Counter(crf.state_features_).most_common(20)
    print("Top postive:")
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

    state_features = Counter(crf.state_features_).most_common()[-20:]
    print("\nTop negative:")
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
