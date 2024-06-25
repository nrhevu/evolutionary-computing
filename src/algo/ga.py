from random import random

from matplotlib import pyplot as plt

from src.utils.operator import crossover, mutation, new_generation_selection


def GA(population, n_gen, task, representation, prob_mutation=0.1, plot_progress=True):
    gen = 1
    progress = []
    n_individual = len(population.individuals)
    while gen <= n_gen:
        children = []
        for i in range(n_individual // 2):
            # select parent randomly
            a = random.randint(0, n_individual - 1)
            b = random.randint(0, n_individual - 1)
            while a == b:
                b = random.randint(0, n_individual - 1)
            individual_a = population.individuals[a]
            individual_b = population.individuals[b]
            # offspring
            rand = random.random()

            if rand > prob_mutation:
                children.extend(crossover(individual_a, individual_b, representation))
            else:
                children.append(mutation(individual_a, representation))
                children.append(mutation(individual_b, representation))
        # Calculate Fitness of children in population
        for i in range(len(children)):
            children[i].fitness = task.compute_fitness(children[i])
        population.add(children)
        population.individuals = new_generation_selection(
            population, population.n_individual
        )
        progress.append(population.get_best_individual().fitness)
        # update gen
        gen += 1
    population.individuals = sorted(
        population.individuals, key=lambda individual: individual.fitness, reverse=True
    )

    if plot_progress:
        plt.plot(progress)
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.show()

    return population
