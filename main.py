import time

from deap import base
from deap import creator
from deap import tools
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from classifiers.SVC_clf import mutation_svc, SVC_parameters_fitness, SVCParametersFeatures
from classifiers.decision_tree_clf import decision_tree_fitness, DecisionTreeParametersFeatures, \
    mutation_decision_tree
from classifiers.random_forest_clf import mutation_random_forest, random_forest_fitness, \
    RandomForestParametersFeatures
from classifiers.k_neighbours_clf import KNeighborsParametersFeatureFitness, KNeighborsParametersFeatures, \
    mutationKNeighbors

from classifiers.ADABoost_clf import ADABoost_fitness, ADABoostParametersFeatures, mutation_ADABoost


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))
    return icls(genome)


def choose_classifier(toolbox, number_of_attributes, df, y, classifier_id):
    if classifier_id == 1:
        toolbox.register("individual", DecisionTreeParametersFeatures, number_of_attributes, creator.Individual)
        toolbox.register("evaluate", decision_tree_fitness, y, df, number_of_attributes)
        toolbox.register("mutate", mutation_decision_tree)

    elif classifier_id == 2:
        toolbox.register("individual", ADABoostParametersFeatures, number_of_attributes, creator.Individual)
        toolbox.register("evaluate", ADABoost_fitness, y, df, number_of_attributes)
        toolbox.register("mutate", mutation_ADABoost)

    elif classifier_id == 3:
        toolbox.register("individual", KNeighborsParametersFeatures, number_of_attributes, creator.Individual)
        toolbox.register("evaluate", KNeighborsParametersFeatureFitness, y, df, number_of_attributes)
        toolbox.register("mutate", mutationKNeighbors)

    elif classifier_id == 4:
        toolbox.register("individual", RandomForestParametersFeatures, number_of_attributes, creator.Individual)
        toolbox.register("evaluate", random_forest_fitness, y, df, number_of_attributes)
        toolbox.register("mutate", mutation_random_forest)

    elif classifier_id == 5:
        toolbox.register("individual", SVCParametersFeatures, number_of_attributes, creator.Individual)
        toolbox.register("evaluate", SVC_parameters_fitness, y, df, number_of_attributes)
        toolbox.register("mutate", mutation_svc)

    return toolbox


def main():
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("ReplicatedAcousticFeatures-ParkinsonDatabase.csv", sep=',')

    y = df['Status']
    df.drop('Status', axis=1, inplace=True)
    df.drop('ID', axis=1, inplace=True)
    df.drop('Recording', axis=1, inplace=True)
    number_of_attributes = len(df.columns)
    print(number_of_attributes)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    clf = SVC()
    scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(scores.mean())

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    print(
        "Choose classificator: \n 1: decision tree \n 2: ADABoost \n 3: k-neighbours \n 4: random forest \n 5: SVC \n")
    classifier = int(input())
    toolbox = choose_classifier(toolbox, number_of_attributes, df, y, classifier_id=classifier)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    plot_x, plot_value, plot_std, plot_avg = [], [], [], []
    print("It's a Genetic algorithm with couple of deap library examples")
    print("Please choose details:")

    print("Choose selection: \n 0: Tournament \n 1: Random \n 2: Best \n 3: Worst \n 4: Roulette \n 5: Blend")
    user_input_selection = input()
    if user_input_selection == '0':
        print("Choose tournament size: ")
        tournament_size = input()
        toolbox.register("select", tools.selTournament, tournsize=int(tournament_size))
    elif user_input_selection == '1':
        toolbox.register("select", tools.selRandom)
    elif user_input_selection == '2':
        toolbox.register("select", tools.selBest)
    elif user_input_selection == '3':
        toolbox.register("select", tools.selWorst)
    elif user_input_selection == '4':
        toolbox.register("select", tools.selRoulette)
    elif user_input_selection == '5':
        toolbox.register("select", tools.cxUniformPartialyMatched, indpb=0.9)
    else:
        toolbox.register("select", tools.selBest)

    print("Choose crossover method: \n 0: One point \n 1: Two point \n 2: uniform \n 3: Messy One point")
    user_input_mate = input()
    if user_input_mate == '0':
        toolbox.register("mate", tools.cxOnePoint)
    elif user_input_mate == '1':
        toolbox.register("mate", tools.cxTwoPoint)
    elif user_input_mate == '2':
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
    elif user_input_mate == '3':
        toolbox.register("mate", tools.cxMessyOnePoint)
    else:
        toolbox.register("mate", tools.cxOnePoint)

    print(
        "Choose mutation method: \n 0: Gaussian \n 1: Uniform Int \n 2: Shuffle Indexes\n 3: Custom algorithm mutation")
    user_input_mutation = input()
    if user_input_mutation == '0':
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.9)
    if user_input_mutation == '1':
        toolbox.register("mutate", tools.mutUniformInt, low=-10, up=10)
    if user_input_mutation == '2':
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.9)
    if user_input_mutation == '3':
        if classifier == 1:
            toolbox.register("mutate", mutation_decision_tree)
        elif classifier == 2:
            toolbox.register("mutate", mutation_ADABoost)
        elif classifier == 3:
            toolbox.register("mutate", mutationKNeighbors)
        elif classifier == 4:
            toolbox.register("mutate", mutation_random_forest)
        elif classifier == 5:
            toolbox.register("mutate", mutation_svc)
    else:
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.9)

    print("Set size of population: ")
    # user_input_population_size = input()
    # sizePopulation = int(user_input_population_size)
    sizePopulation = 50

    print("Set mutation probablility: ")
    # user_input_mutation_probability = input()
    # probabilityMutation = float(user_input_mutation_probability)
    probabilityMutation = 0.1

    print("Set crossover probability: ")
    # user_input_crossover_probability = input()
    # probabilityCrossover = float(user_input_crossover_probability)
    probabilityCrossover = 0.9
    print("Set number of generations: ")
    # user_input_number_of_generations = input()
    # numberIteration = int(user_input_number_of_generations)
    numberIteration = 50

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    numberElitism = 1
    start_time = time.clock()
    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print(" Min %s" % min(fits))
        print(" Max %s" % max(fits))
        print(" Avg %s" % mean)
        print(" Std %s" % std)
        best_ind = tools.selWorst(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind,
                                             best_ind.fitness.values))
        plot_x.append(g)
        plot_avg.append(mean)
        plot_std.append(std)
        plot_value.append(best_ind.fitness.values)
    #
    timeOfAll = time.clock() - start_time
    print(f'Time: {timeOfAll}')
    print("-- End of (successful) evolution --")

    plt.figure()
    plt.plot(plot_value)
    plt.ylabel('Best values')
    plt.show()
    plt.figure()
    plt.plot(plot_avg)
    plt.ylabel('Averages')
    plt.show()
    plt.figure()
    plt.plot(plot_std)
    plt.ylabel('Standard deviation')
    plt.show()


if __name__ == '__main__':
    main()
