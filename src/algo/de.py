from random import random

import numpy as np
from matplotlib import pyplot as plt

from src.modeling.representation import Individual, RealValueRepresentation
from src.utils.operator import crossover, mutation, new_generation_selection


def DE(
    population,
    n_gen,
    task,
    representation,
    type_de="DE/rand/l",
    prob_mutation=0.1,
    plot_progress=True,
):
    if representation != RealValueRepresentation:
        raise (Exception("No"))

    gen = 1
    progress = []
    n_individual = len(population.individuals)
    while gen <= n_gen:
        children = []

        for f in range(len(population.individuals) // 4):
            father = population.individuals[f]
            a = -1

            if type_de == "DE/rand/l":
                a = random.randint(0, n_individual - 1)
            elif type_de == "DE/best/l":
                a = 0

            if a == -1:
                raise (Exception("Error"))

            b = random.randint(0, n_individual - 1)
            while a == b:
                b = random.randint(0, n_individual - 1)
            c = random.randint(0, n_individual - 1)
            while c == b or c == a:
                c = random.randint(0, n_individual - 1)

            individual_a = population.individuals[a]
            individual_b = population.individuals[b]
            individual_c = population.individuals[c]

            g_a = np.array(individual_a.gene)
            g_b = np.array(individual_b.gene)
            g_c = np.array(individual_c.gene)

            g_d = list(g_a + prob_mutation * (g_b - g_c))
            individual_d = Individual(g_d, None)

            if random.random() > prob_mutation:
                children.extend(crossover(father, individual_d, representation))

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
