import random
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def SVCParametersFeatures(number_features, icls):
    genome = list()

    # kernel
    list_kernel = ["linear", "rbf", "poly", "sigmoid"]
    genome.append(list_kernel[random.randint(0, 3)])

    # C
    k = random.uniform(0.1, 100)
    genome.append(k)

    # degree
    genome.append(random.uniform(0.1, 5))

    # gamma
    gamma = random.uniform(0.001, 5)
    genome.append(gamma)

    for i in range(0, number_features):
        genome.append(random.randint(0, 1))
    return icls(genome)


def SVC_parameters_fitness(y, df, number_of_attributes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    list_columns_to_drop = []
    for i in range(number_of_attributes, len(individual)):
        if individual[i] == 0:
            list_columns_to_drop.append(i - number_of_attributes)

    df_selected_features = df.drop(df.columns[list_columns_to_drop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df_selected_features)

    estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                    coef0=individual[4], random_state=101)

    result_sum = 0

    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)

        result_sum = result_sum + result

    return result_sum / split,


def mutation_svc(individual):
    number_parameter = random.randint(0, len(individual) - 1)
    if number_parameter == 0:
        # kernel
        list_kernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0] = list_kernel[random.randint(0, 3)]
    elif number_parameter == 1:
        # C
        k = random.uniform(0.1, 100)
        individual[1] = k

    elif number_parameter == 2:
        # degree
        individual[2] = random.uniform(0.1, 5)

    elif number_parameter == 3:
        # gamma
        gamma = random.uniform(0.01, 1)
        individual[3] = gamma

    elif number_parameter == 4:
        # coeff
        coeff = random.uniform(0.1, 1)
        individual[4] = coeff
    else:
        if individual[number_parameter] == 0:
            individual[number_parameter] = 1
        else:
            individual[number_parameter] = 0
