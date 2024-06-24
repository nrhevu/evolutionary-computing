from typing import List

from src.representation import Individual, Representation
from src.task.task import Task


class Population:
    n_individual: int
    length_gene: int
    individuals: list
    task: Task

    def __init__(self, n_individual: int, task: Task, representation: Representation):
        self.n_individual = n_individual
        self.task = task
        self.representation = representation
        # get gene length of task
        self.length_gene = task.get_len_gene()

    def init_pop(self):
        individuals = list()

        for i in range(self.n_individual):
            ## random gene
            g = self.representation.random(self.length_gene)

            # compute fitness for task
            fitness = 0.0
            fitness = self.task.compute_fitness(g)

            individual = Individual(g, fitness)
            individuals.append(individual)
        # update individuals
        self.individuals = individuals

    def add(self, offsprings) -> List:
        # add offsprings to the population
        self.individuals.extend(offsprings)

    def get_best_individual(self):
        return sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)[0]

    def sort(self) -> List:
        list_individual_in_task = list()

        for individual in self.individuals:
            list_individual_in_task.append(
                Individual(individual.gene, individual.fitness)
            )

        return sorted(
            list_individual_in_task, key=lambda ind: ind.fitness, reverse=True
        )

    def __repr__(self) -> str:
        return (
            "Number of individuals: "
            + str(len(self.individuals))
            + ", Task"
            + str(self.task)
        )
