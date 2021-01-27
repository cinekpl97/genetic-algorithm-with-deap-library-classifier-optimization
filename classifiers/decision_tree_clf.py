import random
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def DecisionTreeParametersFeatures(number_of_features, icls):
    genome = list()

    criterion = ["gini", "entropy"]
    genome.append(criterion[random.randint(0, 1)])

    splitter = ["best", "random"]
    genome.append(splitter[random.randint(0, 1)])

    max_depth = random.randint(1, 50)
    genome.append(max_depth)

    for i in range(0, number_of_features):
        genome.append(random.randint(0, 1))
    return icls(genome)


def decision_tree_fitness(y, df, number_of_attributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(number_of_attributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - number_of_attributes)
    df_selected_features = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df_selected_features)
    estimator = DecisionTreeClassifier(criterion=individual[0], splitter=individual[1], max_depth=individual[2])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutation_decision_tree(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    if number_parameter == 0:
        criterion = ["gini", "entropy"]
        individual[0] = criterion[random.randint(0, 1)]

    elif number_parameter == 1:
        splitter = ["best", "random"]
        individual[1] = splitter[random.randint(0, 1)]

    elif number_parameter == 2:
        max_depth = random.randint(1, 50)
        individual[2] = max_depth

    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0
