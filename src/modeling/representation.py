import random
from abc import abstractmethod

import numpy as np


class Individual:
    gene: list
    fitness: float

    def __init__(self, gene, fitness):
        self.gene = gene
        self.fitness = fitness

    def __repr__(self) -> str:
        return (
            "Individual [gene="
            + str(self.gene)
            + ", fitness="
            + str(self.fitness)
            + "]"
        )


class Representation:
    @abstractmethod
    def random():
        pass

    @abstractmethod
    def crossover():
        pass

    @abstractmethod
    def mutation():
        pass


## PERMUTATION
class PermutationRepresentation(Representation):
    @staticmethod
    def random(length_gene):
        return random.sample([i for i in range(1, length_gene + 1)], length_gene)

    @staticmethod
    def crossover(individual_a, individual_b):
        ######## PERMUTATION CROSSOVER ##########
        # Crossover by selecting 2 points and copy gene of parent 1 between them and copy ordered gene from parent 2
        length_gene = len(individual_a.gene)
        child_1 = [-1] * length_gene
        child_2 = [-1] * length_gene
        start_point = int(np.random.randint(0, length_gene - 1))
        end_point = int(np.random.randint(start_point + 1, length_gene))
        # copy gene of the first parent to child
        child_1[start_point : end_point + 1] = individual_a.gene[
            start_point : end_point + 1
        ]
        child_2[start_point : end_point + 1] = individual_b.gene[
            start_point : end_point + 1
        ]
        # select ordered gene from the second parent
        childP1 = [item for item in individual_b.gene if item not in child_1]
        childP2 = [item for item in individual_a.gene if item not in child_2]
        # complete child_1 gene
        for i, item in enumerate(child_1):
            if item == -1:
                child_1[i] = childP1[0]
                childP1.pop(0)
        # complete child_2 gene
        for i, item in enumerate(child_2):
            if item == -1:
                child_2[i] = childP2[0]
                childP2.pop(0)

        return child_1, child_2

    @staticmethod
    def mutation(individual):
        ####### PERMUTATION MUTATION #########
        gene = individual.gene.copy()
        # mutate
        swapped = int(random.random() * len(gene))
        swapWith = int(random.random() * len(gene))

        city1 = gene[swapped]
        city2 = gene[swapWith]

        gene[swapped] = city2
        gene[swapWith] = city1

        return gene


# REAL-VALUE
class RealValueRepresentation(Representation):
    @staticmethod
    def random(length_gene):
        g = list()
        for i in range(length_gene):
            g.append(random.random())

        return g

    @staticmethod
    def crossover(individual_a, individual_b, nu=1):
        ######### REAL-VALUE REPRESENTATION ########
        length_gene = len(individual_a.gene)
        c1 = [-1] * length_gene
        c2 = [-1] * length_gene
        # calculate beta
        # mu = random.random()
        # beta = None
        # if mu < 0.5:
        #     beta = (2.0 * mu) ** (1.0 / (nu + 1))
        # elif mu >= 0.5:
        #     beta = (1.0 / (2.0 * (1 - mu))) ** (1.0 / (nu + 1))

        p1 = individual_a.gene
        p2 = individual_b.gene
        # calculate children points
        for i in range(length_gene):
            mu = random.random()
            beta = None
            if mu < 0.5:
                beta = (2.0 * mu) ** (1.0 / (nu + 1))
            elif mu >= 0.5:
                beta = (1.0 / (2.0 * (1 - mu))) ** (1.0 / (nu + 1))

            c1[i] = 0.5 * ((p1[i] + p2[i]) - beta * abs(p2[i] - p1[i]))
            c2[i] = 0.5 * ((p1[i] + p2[i]) + beta * abs(p2[i] - p1[i]))

        # c1 = list(np.max(np.vstack((c1, np.array([0.001] * length_gene))), 0))
        # c1 = list(np.min(np.vstack((c1, np.array([1] * length_gene))), 0))
        # c2 = list(np.max(np.vstack((c2, np.array([0.001] * length_gene))), 0))
        # c2 = list(np.min(np.vstack((c2, np.array([1] * length_gene))), 0))

        return c1, c2

    @staticmethod
    def mutation(individual, nu=20):
        ######### REAL-VALUE REPRESENTATION ########
        length_gene = len(individual.gene)
        child = [-1] * length_gene
        # assign range value
        xi_L = 0
        xi_U = 1
        p = individual.gene
        # the mutated solution p′
        # for a particular variable is created for a random number u created within [0, 1], as
        # follows
        for i in range(length_gene):
            u = random.random()
            if u <= 0.5:
                delta_L = (2 * u) ** (1 / (1 + nu)) - 1
                child[i] = p[i] + delta_L * (p[i] - xi_L)
            elif u > 0.5:
                delta_U = 1 - (2 * (1 - u)) ** (1 / (1 + nu))
                child[i] = p[i] + delta_U * (xi_U - p[i])

        return child


## IDPCDU
class IDPCDURepresentation(Representation):
    @staticmethod
    def random(length_gene):
        g = [[] for i in range(2)]
        for i in range(length_gene):
            g[0].append(random.random())
            g[1].append(random.randint(0, 1000))

        return g

    @staticmethod
    def crossover(individual_a, individual_b, nu=1):
        ######### REAL-VALUE REPRESENTATION ########

        ###############################################
        ####### Crossover 1st layer (realvalue) #######
        ###############################################

        length_gene = len(individual_a.gene[0])
        c1 = [-1] * length_gene
        c2 = [-1] * length_gene
        # calculate beta
        # mu = random.random()
        # beta = None
        # if mu < 0.5:
        #     beta = (2.0 * mu) ** (1.0 / (nu + 1))
        # elif mu >= 0.5:
        #     beta = (1.0 / (2.0 * (1 - mu))) ** (1.0 / (nu + 1))

        p1 = individual_a.gene[0]
        p2 = individual_b.gene[0]
        # calculate children points
        for i in range(length_gene):
            mu = random.random()
            beta = None
            if mu < 0.5:
                beta = (2.0 * mu) ** (1.0 / (nu + 1))
            elif mu >= 0.5:
                beta = (1.0 / (2.0 * (1 - mu))) ** (1.0 / (nu + 1))

            c1[i] = 0.5 * ((p1[i] + p2[i]) - beta * abs(p2[i] - p1[i]))
            c2[i] = 0.5 * ((p1[i] + p2[i]) + beta * abs(p2[i] - p1[i]))

        # c1 = list(np.max(np.vstack((c1, np.array([0.001] * length_gene))), 0))
        # c1 = list(np.min(np.vstack((c1, np.array([1] * length_gene))), 0))
        # c2 = list(np.max(np.vstack((c2, np.array([0.001] * length_gene))), 0))
        # c2 = list(np.min(np.vstack((c2, np.array([1] * length_gene))), 0))

        ###############################
        ##### Crossover 2nd layer #####
        ###############################
        d1 = [-1] * length_gene
        d2 = [-1] * length_gene

        p1 = individual_a.gene[1]
        p2 = individual_b.gene[1]
        for i in range(length_gene):
            rand = random.random()
            if rand < 0.5:
                d1[i] = p1[i]
                d2[i] = p2[i]
            else:
                d1[i] = p2[i]
                d2[i] = p1[i]

        return [c1, d1], [c2, d2]

    @staticmethod
    def mutation(individual, nu=20):
        ######### REAL-VALUE REPRESENTATION ########

        length_gene = len(individual.gene[0])

        ###############################################
        ####### Crossover 1st layer (realvalue) #######
        ###############################################

        c = [-1] * length_gene
        # assign range value
        xi_L = 0
        xi_U = 1
        p = individual.gene[0]
        # the mutated solution p′
        # for a particular variable is created for a random number u created within [0, 1], as
        # follows
        for i in range(length_gene):
            u = random.random()
            if u <= 0.5:
                delta_L = (2 * u) ** (1 / (1 + nu)) - 1
                c[i] = p[i] + delta_L * (p[i] - xi_L)
            elif u > 0.5:
                delta_U = 1 - (2 * (1 - u)) ** (1 / (1 + nu))
                c[i] = p[i] + delta_U * (xi_U - p[i])

        ###############################
        ##### Crossover 2nd layer #####
        ###############################

        d = [-1] * length_gene
        p = individual.gene[1]
        for i in range(length_gene):
            rand = random.random()
            if rand > 0.1:
                d[i] = p[i]
            else:
                d[i] = random.randint(0, 1000)

        return [c, d]
