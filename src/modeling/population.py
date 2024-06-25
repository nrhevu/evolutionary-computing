from typing import List

from src.modeling.representation import Individual, Representation
from src.modeling.task import Task


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


class MFEAPopulation:
    n_individual: int
    n_task: int
    length_gene: int
    individuals: list
    tasks: list

    def __init__(self, n_individual: int, tasks: list, representation: Representation):
        self.n_individual = n_individual
        self.n_task = len(tasks)
        self.tasks = tasks
        self.representation = representation
        # get max gene length of all tasks
        max_len = 0
        for task in tasks:
            if task.get_len_gene() > max_len:
                max_len = task.get_len_gene()
        self.length_gene = max_len

    def init_pop(self):
        individuals = list()

        for i in range(self.n_individual):
            ## random gene
            g = self.representation.random(self.length_gene)

            # init individual
            if self.check_individual_valid(g):
                self.make_individual_valid(g)
            # compute fitness for all tasks
            fitness_tasks = list()
            for task in self.tasks:
                fitness_tasks.append(task.compute_fitness(g))

            individual = Individual(g, fitness_tasks)
            individuals.append(individual)
        # update individuals
        self.individuals = individuals

        self.update_rank_population()

    def check_individual_valid(self, individual):
        for task in self.tasks:
            if task.check_individual_valid(individual):
                return True

    def make_individual_valid(self, individual):
        i = 0
        xd = 0
        while True:
            task = self.tasks[i]
            if task.check_individual_valid(individual):
                xd = 0
                task.make_individual_valid(individual)
            else:
                xd += 1
            if xd >= len(self.tasks):
                break
            # loop through tasks list and turn around
            i = (i + 1) % len(self.tasks)

    def update_rank_population(self):
        rank_in_task = list()
        # make list individual in each task
        for i in range(self.n_task):
            list_individual_in_task = list()
            rank_in_task.append(list_individual_in_task)
        # sort individuals on thier fitness in each task
        ## Method 1
        # for i_ind in range(self.n_individual):
        #     individual = self.individuals[i_ind]
        # for i in range(self.n_task):
        #   list_individual_in_task = rank_in_task[i]
        #   check = True
        #
        #   for j in range(len(list_individual_in_task)):
        #       if list_individual_in_task[j].fitness_task[i] > \
        #           individual.fitness_task[i]:
        #           list_individual_in_task.append(j, individual)
        #           check = False
        #           break
        #
        #   if check :
        #       list_individual_in_task.append(individual)
        #
        #   rank_in_task[i] = list_individual_in_task

        ## Method 2
        individuals = self.individuals.copy()
        for i_ind in range(self.n_individual):
            individual = individuals[i_ind]
            for i in range(self.n_task):
                rank_in_task[i].append(individual)
        for i in range(self.n_task):
            list_individual_in_task = rank_in_task[i]
            rank_in_task[i] = sorted(
                list_individual_in_task,
                key=lambda ind: ind.fitness_task[i],
                reverse=True,
            )

        # set factorial_rank, skill_factor, scalar_fitness for each individual
        for i_ind in range(self.n_individual):
            individual = self.individuals[i_ind]
            factorial_rank = list()
            min_rank = self.n_individual + 2
            task_rank_min = -1
            for j in range(self.n_task):
                # set factorial rank for each task
                rank_j = rank_in_task[j].index(individual) + 1
                factorial_rank.append(rank_j)

                # find task with highest ranking
                if rank_j < min_rank:
                    min_rank = rank_j
                    task_rank_min = j
            # set properties for individual
            individual.factorial_rank = factorial_rank
            individual.skill_factor = task_rank_min
            individual.scalar_fitness = 1 / min_rank

    def add(self, offsprings) -> List:
        # add offsprings to the population
        individuals = self.individuals.copy()
        individuals.extend(offsprings)

        # Update the scalar fitness (φ) and skill factor (τ) of every individual in intermediate-pop
        for ind in range(len(offsprings)):
            # get skill factor of offsprings
            child = offsprings[ind]
            child_task = child.skill_factor
            # count rank
            list_individual_in_task = list()

            for individual in individuals:
                list_individual_in_task.append(individual)

            rank_in_task = sorted(
                list_individual_in_task,
                key=lambda ind: ind.fitness_task[child_task],
                reverse=True,
            )
            # get rank of individual
            index = -1
            for j in range(len(rank_in_task)):
                if (
                    rank_in_task[j].fitness_task[child_task]
                    < child.fitness_task[child_task]
                ):
                    index = j
                    break

            if index > -1:
                for j in range(index, len(rank_in_task)):
                    tmp = rank_in_task[j]
                    rank = tmp.factorial_rank
                    rank[child_task] = rank[child_task] + 1
                    tmp.factorial_rank = rank
            else:
                index = len(rank_in_task)

            fac_rank_ind = list()
            for i in range(self.n_task):
                fac_rank_ind.append(len(individuals) + 1)
            fac_rank_ind[child_task] = index + 1
            child.factorial_rank = fac_rank_ind
            offsprings[ind] = child

        for ind in offsprings:
            ind.scalar_fitness = 1 / ind.get_min_factorial_rank()

        self.individuals = individuals

    def count_rank(self, task) -> List:
        list_individual_in_task = list()

        for individual in self.individuals.copy():
            list_individual_in_task.append(individual)

        return sorted(
            list_individual_in_task,
            key=lambda ind: ind.fitness_task[task],
            reverse=True,
        )

    def get_individual_best_of_task(self, task):
        return sorted(
            self.individuals.copy(),
            key=lambda individual: individual.fitness_task[task],
            reverse=True,
        )[0]

    def __repr__(self) -> str:
        return "Number of individuals: " + len(self.individuals) + ", "
