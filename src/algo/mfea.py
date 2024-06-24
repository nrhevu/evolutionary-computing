from random import random
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.modeling.population import MFEAPopulation
from src.modeling.representation import Individual, Representation


class MFEA:
    population: MFEAPopulation
    n_indivudual: int
    time_reset_population: int
    prob_mutation: float
    tasks: List

    LIMIT = 100000000
    ITERATIONS = 1

    def __init__(
        self,
        tasks: List,
        n_individual: int,
        prob_mutation: float,
        representation: Representation,
        time_reset_population: int = 5,
    ):
        self.tasks = tasks
        self.n_individual = n_individual
        self.prob_mutation = prob_mutation
        self.representation = representation
        self.time_reset_population = time_reset_population

        self.population = MFEAPopulation(n_individual, tasks, representation)

    def run(self, n_generation):
        best_solution = list()

        self.population.init_pop()

        for i in range(len(self.tasks)):
            best_solution.append(self.population.individuals[i])

        # change_best = 0
        for iter in range(self.ITERATIONS):
            self.population.init_pop()

            gen = 1
            progress = []

            while gen <= n_generation:
                individuals = self.population.individuals.copy()
                children = list()
                # Genetic Algorithm
                for i in range(self.n_individual // 2):
                    # select parent randomly
                    a = random.randint(0, len(individuals) - 1)
                    b = random.randint(0, len(individuals) - 1)
                    while a == b:
                        b = random.randint(0, len(individuals) - 1)
                    individual_a = individuals[a]
                    individual_b = individuals[b]
                    # apply genetic operators on current-pop to generate an offspring-pop
                    t_a = individual_a.skill_factor
                    t_b = individual_b.skill_factor

                    rand = random.random()
                    if (t_a == t_b) or (rand > self.prob_mutation):
                        c = self.crossover(individual_a, individual_b)
                        children.append(c[0])
                        children.append(c[1])
                    else:
                        children.append(self.mutation(individual_a))
                        children.append(self.mutation(individual_b))

                # Concatenate offspring-pop and current-pop to form an intermediate-pop
                # Update the scalar fitness (φ) and skill factor (τ) of every individual in intermediate-pop
                self.population.add(children)
                # Select the fittest individuals from intermediate-pop to form the next current-pop (P).
                self.new_generation_selection()
                self.re_compute_fitness_task_for_chilren(children)
                # save progress
                progress.append(
                    [
                        self.population.get_individual_best_of_task(i).fitness_task[i]
                        for i in range(len(self.tasks))
                    ]
                )
                self.population.update_rank_population()
                # up generation
                gen += 1

            # update best solution
            print("--------ITER %d--------" % (iter + 1))
            for task in range(len(self.tasks)):
                ind = self.population.get_individual_best_of_task(task)
                if best_solution[task].fitness_task[task] < ind.fitness_task[task]:
                    # change_best = 0
                    best_solution[task] = ind

                print(
                    " Task %d (%s) : %s" % (task + 1, self.tasks[task].task_name, ind)
                )

            fig, ax = plt.subplots(1, len(self.tasks), figsize=(20, 5))
            # plt.plot(progress)
            # plt.ylabel('Fitness')
            # plt.xlabel('Generation')
            # plt.legend([task.task_name for task in self.tasks])
            # plt.show()
            for i in range(len(self.tasks)):
                print([p[i] for p in progress])
                ax[i].plot([p[i] for p in progress])
                ax[i].set_title(self.tasks[i].task_name)
            plt.show()

            # change_best += 1
            # if(change_best >= self.time_reset_population):
            #    change_best = 0

        # print best solution
        print("Solution: ")
        for task in range(len(self.tasks)):
            ind = self.population.get_individual_best_of_task(task)
            print(" Task %d (%s) : %s" % (task + 1, self.tasks[task].task_name, ind))
            self.tasks[task].show_result(ind)

    def re_compute_fitness_task_for_chilren(self, children):
        for child in children:
            for j in range(len(self.tasks)):
                if child.fitness_task[j] == self.LIMIT:
                    child.fitness_task[j] = self.tasks[j].compute_fitness(child.gene)

    def crossover(self, individual_a, individual_b):
        children = list()
        factorial_rank = list()
        for i in range(len(self.tasks)):
            factorial_rank.append(len(self.population.individuals) + 1)

        child_1, child_2 = self.representation.crossover(individual_a, individual_b)

        ## offsprings
        # Evaluate the individuals in offspring-pop for selected optimization tasks only

        # check child 1
        if self.population.check_individual_valid(child_1):
            self.population.make_individual_valid(child_1)
        # produce child 1
        ind = Individual(child_1, None)
        rand = random.random()

        # set skill factor for chil
        if rand < 0.5:
            ind.skill_factor = individual_a.skill_factor
        else:
            ind.skill_factor = individual_b.skill_factor
        # compute fitness each task for the child
        fitness_task = list()
        for i in range(len(self.tasks)):
            if i != ind.skill_factor:
                fitness_task.append(self.LIMIT)
            else:
                fitness_task.append(self.tasks[i].compute_fitness(child_1))
        ind.fitness_task = fitness_task
        ind.factorial_rank = factorial_rank
        children.append(ind)

        # check child 2
        if self.population.check_individual_valid(child_2):
            self.population.make_individual_valid(child_2)

        # produce child 2
        ind2 = Individual(child_2, None)
        rand = random.random()

        # set skill factor for chil
        if rand < 0.5:
            ind2.skill_factor = individual_a.skill_factor
        else:
            ind2.skill_factor = individual_b.skill_factor
        # compute fitness each task for the child
        fitness_task = list()
        for i in range(len(self.tasks)):
            if i != ind2.skill_factor:
                fitness_task.append(self.LIMIT)
            else:
                fitness_task.append(self.tasks[i].compute_fitness(child_2))
        ind2.fitness_task = fitness_task
        ind2.factorial_rank = factorial_rank
        children.append(ind2)

        return children

    def mutation(self, individual):
        factorial_rank = list()
        for i in range(len(self.tasks)):
            factorial_rank.append(len(self.population.individuals) + 1)
        # make gene
        gene = self.representation.mutation(individual)

        # make individual
        # check child 1
        if self.population.check_individual_valid(gene):
            self.population.make_individual_valid(gene)
        # produce child 1
        ind = Individual(gene, None)
        rand = random.random()

        # set skill factor
        ind.skill_factor = individual.skill_factor
        # compute fitness each task for the child
        fitness_task = list()
        for i in range(len(self.tasks)):
            if i != ind.skill_factor:
                fitness_task.append(self.LIMIT)
            else:
                fitness_task.append(self.tasks[i].compute_fitness(gene))
        ind.fitness_task = fitness_task
        ind.factorial_rank = factorial_rank

        return ind

    def selection_operator(self):
        # rank-based selection
        sorted_population = sorted(
            self.population.individuals,
            key=lambda individual: individual.scalar_fitness,
            reverse=True,
        )
        n = np.random.randint(0, random.randint(1, len(self.population.individuals)))
        return sorted_population[n]
        # proportional selection
        # max = sum([c.scalar_fitness for c in self.population.individuals])
        # selection_probs = [c.scalar_fitness/max for c in self.population.individuals]
        # return self.population.individuals[np.random.choice(len(self.population.individuals), p=selection_probs)]

    def new_generation_selection(self, n_elite=1):
        # select individual for next generation
        new_individual = []
        individuals = self.population.individuals.copy()
        for task in range(len(self.tasks)):
            individuals = sorted(
                individuals,
                key=lambda individual: individual.fitness_task[task],
                reverse=True,
            )
            for i in range(n_elite):
                new_individual.append(individuals[0])
                individuals.pop(0)

        sorted_population = sorted(
            individuals, key=lambda individual: individual.scalar_fitness, reverse=True
        )
        for i in range(self.n_individual - n_elite * len(self.tasks)):
            n = np.random.randint(0, random.randint(1, len(sorted_population)))
            new_individual.append(sorted_population[n])
            sorted_population.pop(n)

        self.population.individuals = new_individual
