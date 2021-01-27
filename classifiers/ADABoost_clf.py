import random
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def ADABoostParametersFeatures(number_of_features, icls):
    genome = list()

    n_estimator = random.randint(1, 50)
    genome.append(n_estimator)
    algorithm = ["SAMME", "SAMME.R"]
    genome.append(algorithm[random.randint(0, 1)])

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))
    return icls(genome)


def ADABoost_fitness(y, df, number_of_attributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(number_of_attributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - number_of_attributes)
    df_selected_features = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df_selected_features)
    estimator = AdaBoostClassifier(n_estimators=individual[0], algorithm=individual[1])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutation_ADABoost(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    if number_parameter == 0:
        n_estimator = random.randint(1, 50)
        individual[0] = n_estimator

    elif number_parameter == 1:
        algorithm = ["SAMME", "SAMME.R"]
        individual[1] = algorithm[random.randint(0, 1)]

    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0
