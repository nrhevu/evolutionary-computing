import random

import numpy as np

from src.modeling.representation import Individual

LIMIT = 9999999


def crossover(individual_a, individual_b, representation):
    children = list()

    child_1, child_2 = representation.crossover(individual_a, individual_b)
    ## offsprings
    # Evaluate the individuals in offspring-pop for selected optimization tasks only
    # produce child 1
    ind1 = Individual(child_1, LIMIT)
    children.append(ind1)
    # produce child 2
    ind2 = Individual(child_2, LIMIT)
    children.append(ind2)
    return children


def mutation(individual, representation):
    # make gene
    gene = representation.mutation(individual)
    # produce child 1
    ind = Individual(gene, LIMIT)

    return ind


def selection_operator(population):
    # rank-based selection
    sorted_population = sorted(
        population.individuals, key=lambda individual: individual.fitness, reverse=True
    )
    n = np.random.randint(0, random.randint(1, len(population.individuals)))
    return sorted_population[n]
    # proportional selection
    # max = sum([c.scalar_fitness for c in population.individuals])
    # selection_probs = [c.scalar_fitness/max for c in population.individuals]
    # return population.individuals[np.random.choice(len(population.individuals), p=selection_probs)]


def new_generation_selection(population, n_individual, n_elite=1):
    # select individual for next generation
    new_individual = []
    individuals = population.individuals.copy()
    sorted_population = sorted(
        individuals, key=lambda individual: individual.fitness, reverse=True
    )

    for i in range(n_elite):
        new_individual.append(sorted_population[0])
        sorted_population.pop(0)

    for i in range(n_individual - n_elite):
        n = np.random.randint(0, random.randint(1, len(sorted_population)))
        new_individual.append(sorted_population[n])
        sorted_population.pop(n)

    return new_individual
