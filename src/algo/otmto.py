import math
from random import random

import numpy as np
from scipy.stats import norm

from src.algo.de import DE
from src.modeling.population import Population
from src.modeling.representation import Individual, Real_value_representation
from src.utils import crossover, new_generation_selection
from src.utils.math import KL, norm_2

x = np.arange(-10, 10, 0.001)


class OTMTO:

    tasks: list
    # parameters
    p_OT_i_j: float
    p_CDT_i_j: float
    rOT = int
    max_FE: int

    def __init__(
        self,
        tasks,
        representation=Real_value_representation,
        p_OT_i_j=0.5,
        p_CDT_i_j=0.5,
        maxFE_n=5,
    ):
        self.tasks = tasks
        self.tasks_len = len(tasks)
        self.p_OT_i_j = p_OT_i_j
        self.p_OT = p_OT = [
            [p_OT_i_j for i in range(self.tasks_len)] for j in range(self.tasks_len)
        ]
        self.p_CDT_i_j = p_CDT_i_j
        self.p_CDT = p_CDT = [
            [p_CDT_i_j for i in range(self.tasks_len)] for j in range(self.tasks_len)
        ]
        self.rOT = 0
        self.representation = representation

        # Calculate total dimension -> maxFE
        total_dimension = 0
        for task in tasks:
            total_dimension += task.dimension
        self.maxFE = maxFE_n * total_dimension

    def CTM(self, task_i, task_j, nb, pop_i: Population, pop_j: Population):
        INFE = 0
        maxINFE = 10
        inps = 10

        # sort population from best to worst
        pop_i.individuals = pop_i.sort()
        pop_j.individuals = pop_j.sort()

        # choose x_gb_j
        x_gb_j = pop_j.individuals[0]

        # Initialize the population as the inps best individuals in pop_i;
        pop_map = Population(
            n_individual=inps, task=pop_i.task, representation=pop_i.representation
        )
        pop_map.individuals = pop_i.sort()[:inps]

        ####################
        # CALCULATE fvec_j #
        ####################

        # Calculate tvec_k_j = || x_gb_j - pop_k_j||_2
        tvec_j = []
        lenGene_j = len(x_gb_j.gene)
        for k in range(nb):
            tvec_j.append(norm_2(x_gb_j.gene, pop_j.individuals[k].gene))

        # Calculate ctr_i
        ps_i = len(pop_i.individuals)
        ctr_i = []
        for k in range(len(pop_i.individuals[0].gene)):
            ctr_k_i = 0.0
            for i in range(ps_i):
                ctr_k_i += pop_i.individuals[i].gene[k]
            ctr_k_i /= len(pop_i.individuals[0].gene)
            ctr_i.append(ctr_k_i)

        # Calculate ctr_j
        ps_j = len(pop_j.individuals)
        ctr_j = []
        for k in range(len(pop_j.individuals[0].gene)):
            ctr_k_j = 0.0
            for i in range(ps_j):
                ctr_k_j += pop_j.individuals[i].gene[k]
            ctr_k_j /= len(pop_j.individuals[0].gene)
            ctr_j.append(ctr_k_j)

        # Calculate rad_i
        rad_i = 0.0
        for k in range(ps_i):
            rad_i += norm_2(pop_i.individuals[k].gene, ctr_i)
        rad_i /= ps_i

        # Calculate rad_j
        rad_j = 0.0
        for k in range(ps_j):
            rad_j += norm_2(pop_j.individuals[k].gene, ctr_j)
        rad_j /= ps_j

        # Calculate fvec_j
        fvec_j = []
        for tvec_k_j in tvec_j:
            fvec_j.append(tvec_k_j * (rad_i / rad_j))

        ###################################################
        # RECOMPUTE FITNESS OF EACH INDIVIDUAL IN POP_MAP #
        ###################################################

        # For each inidividual in pop_map
        for i in range(inps):
            x_map = pop_map.individuals[i]
            # Calculate fvec_k_i = ||x_map - pop_k_j||_2
            fvec_i = []
            for k in range(nb):
                fvec_i.append(norm_2(x_map.gene, pop_i.individuals[k].gene))
            # Compute fitness of x_map
            x_map.fitness = norm_2(fvec_i, fvec_j) ** 2

        # loop
        prob_mutation = 0.1
        while INFE < maxINFE:
            # Evolve the population for one generation by using
            # DE/best/1 with the internal fitness function
            individuals = pop_map.individuals.copy()
            n_individuals = len(individuals)
            ##########################################
            ############ Genetic Algorithm ###########
            ##########################################
            #####children = list()
            #####for i in range(n_individuals // 2):
            #####    # select parent randomly
            #####    a = random.randint(0, len(individuals) - 1)
            #####    b = random.randint(0, len(individuals) - 1)
            #####    while(a == b):
            #####        b = random.randint(0, len(individuals) - 1)
            #####    individual_a = individuals[a]
            #####    individual_b = individuals[b]
            #####
            #####    rand = random.random()
            #####    if(rand > prob_mutation):
            #####        children.extend(crossover(individual_a, individual_b, self.representation))
            #####    else:
            #####        children.append(mutation(individual_a, self.representation))
            #####        children.append(mutation(individual_b, self.representation))

            ##########################################
            ############ Differential Evoluation #####
            ##########################################

            children = []

            for f in range(n_individuals // 4):
                father = individuals[f]
                ## DE/best/l
                a = 0

                b = random.randint(0, len(individuals) - 1)
                while a == b:
                    b = random.randint(0, len(individuals) - 1)
                c = random.randint(0, len(individuals) - 1)
                while c == b or c == a:
                    c = random.randint(0, len(individuals) - 1)

                individual_a = individuals[a]
                individual_b = individuals[b]
                individual_c = individuals[c]

                g_a = np.array(individual_a.gene)
                g_b = np.array(individual_b.gene)
                g_c = np.array(individual_c.gene)

                g_d = list(g_a + prob_mutation * (g_b - g_c))
                individual_d = Individual(g_d, None)

                if random.random() > prob_mutation:
                    children.extend(
                        crossover(father, individual_d, self.representation)
                    )

            # Calculate InternalFitness of children in pop_map
            for i in range(len(children)):
                x_map = children[i]
                # Calculate fvec_k_i = ||x_map - pop_k_j||_2
                fvec_i = []
                for k in range(nb):
                    fvec_i.append(norm_2(x_map.gene, pop_i.individuals[k].gene))
                # Compute fitness of x_map
                x_map.fitness = norm_2(fvec_i, fvec_j) ** 2

            pop_map.add(children)
            pop_map.individual = new_generation_selection(pop_map, pop_map.n_individual)
            # update gen
            INFE += 1
        # return x_mgb_j
        x_mgb_j = pop_map.sort()[0]
        return x_mgb_j

    def OT(self, task_i, task_j, D_i, pop_i, x_mgb_j, M=50):
        # Generate OA as LM(2Di);
        L_M_2Di = [[random.randint(1, 2) for i in range(D_i)] for j in range(M)]

        # Randomly select one individual xrk,i from popi;
        rand = random.randint(0, len(pop_i.individuals) - 1)
        x_rk_i = pop_i.individuals[rand]

        # Evaluate each solution on ith task and record the best solution xb;
        eval_ind = []
        max_fitness = 0.0
        factor_analysis = [[[0, 0] for i in range(D_i)] for j in range(2)]
        for i in range(M):
            g = [
                x_rk_i.gene[x] if L_M_2Di[i][x] == 1 else x_mgb_j.gene[x]
                for x in range(len(L_M_2Di[i]))
            ]
            fitness = task_i.compute_fitness(g)
            eval_ind.append(Individual(g, fitness))
        xb = sorted(eval_ind, key=lambda ind: ind.fitness, reverse=True)[0]

        # Use factor analysis to derive a predictive solution xp from the above M fitness;
        for i in range(M):
            fitness_i = eval_ind[i].fitness
            for a in range(D_i):
                if L_M_2Di[i][a] == 1:
                    factor_analysis[0][a][0] += fitness_i
                    factor_analysis[0][a][1] += 1
                elif L_M_2Di[i][a] == 2:
                    factor_analysis[1][a][0] += fitness_i
                    factor_analysis[1][a][1] += 1

        for i in range(2):
            for a in range(D_i):
                factor_analysis[i][a] = (
                    factor_analysis[i][a][0] / factor_analysis[i][a][1]
                )

        # Evaluate the fitness of xp on ith task;
        sol = []
        for i in range(D_i):
            if factor_analysis[0][a] > factor_analysis[1][a]:
                sol.append(1)
            elif factor_analysis[0][a] <= factor_analysis[1][a]:
                sol.append(2)

        g = [
            x_rk_i.gene[x] if sol[x] == 1 else x_mgb_j.gene[x] for x in range(len(sol))
        ]
        fitness = task_i.compute_fitness(g)
        xp = Individual(g, fitness)

        # Compare xb and xp, select the better one as the xOT;
        xOT = None
        if xb.fitness > xp.fitness:
            xOT = xb
        else:
            xOT = xp

        # If xOT is better than xrk,i, xOT replaces xrk,i and set rOT =1, else set rOT =0
        rOT = 0
        if xOT.fitness > x_rk_i.fitness:
            x_rk_i = xOT
            rOT = 1
        else:
            rOT = 0

        return rOT, pop_i

    def CDT(self, task_i, task_j, D_i, D_j, pop_i, pop_j):
        x_CDT_g = []

        # Calculate ctri, stdi of popi and ctrj, stdj of popj;
        # Calculate ctr_i, std_i
        ps_i = len(pop_i.individuals)
        ctr_i = []
        std_i = []
        for k in range(len(pop_i.individuals[0].gene)):
            ctr_k_i = 0.0
            std_k_i = 0.0
            for i in range(ps_i):
                ctr_k_i += pop_i.individuals[i].gene[k]
            ctr_k_i /= len(pop_i.individuals[0].gene)
            for i in range(ps_i):
                std_k_i += (pop_i.individuals[i].gene[k] - ctr_k_i) ** 2
            std_k_i /= len(pop_i.individuals[0].gene)
            std_k_i = math.sqrt(std_k_i)
            ctr_i.append(ctr_k_i)
            std_i.append(std_k_i)

        # Calculate ctr_j, std_j
        ps_j = len(pop_j.individuals)
        ctr_j = []
        std_j = []
        for k in range(len(pop_j.individuals[0].gene)):
            ctr_k_j = 0.0
            std_k_j = 0.0
            for i in range(ps_j):
                ctr_k_j += pop_j.individuals[i].gene[k]
            ctr_k_j /= len(pop_j.individuals[0].gene)
            for i in range(ps_j):
                std_k_j += (pop_j.individuals[i].gene[k] - ctr_k_j) ** 2
            std_k_j /= len(pop_j.individuals[0].gene)
            std_k_j = math.sqrt(std_k_j)
            ctr_j.append(ctr_k_j)
            std_j.append(std_k_j)

        epsilon = 1e-6
        for d in range(D_i):
            # Calculate the probability vector prs
            p_rs = []
            sim = []
            for k in range(D_j):
                p = norm.pdf(x, ctr_i[d], std_i[d])
                q = norm.pdf(x, ctr_j[k], std_j[k])
                sim_k = max(1 / (KL(p, q) + epsilon), 0.001)
                sim.append(sim_k)
            sum_sim = sum(sim)
            for k in range(D_j):
                p_rs.append(sim[k] / sum_sim)

            # Select a dimension sd from {1,...,Dj} according to the
            # probability vector prs in a roulette selection scheme
            sd = np.random.choice([i for i in range(D_j)], p=p_rs)

            # Sample xCDT,d= (ctrj,sd, stdj,sd);
            x_CDT_g.append(random.gauss(ctr_j[sd], std_j[sd]))

        # Evaluate xCDT and compare it with a randomly selected individual xrk,i from popi;
        xCDT = Individual(x_CDT_g, task_i.compute_fitness(x_CDT_g))
        # Randomly select one individual xrk,i from popi;
        rand = random.randint(0, len(pop_i.individuals) - 1)
        x_rk_i = pop_i.individuals[rand]

        # If xCDT is better than xrk,i, xOT replaces xrk,i and set rOT =1, else set rOT =0
        rCDT = 0
        if xCDT.fitness > x_rk_i.fitness:
            x_rk_i = xCDT
            rCDT = 1
        else:
            rCDT = 0

        return rCDT, pop_i

    def run(self, nb=5, n_individuals=100, n_gen=300):
        # Randomly initialize the population for each task
        pop = []
        for k in range(len(self.tasks)):
            pop.append(
                Population(
                    n_individuals,
                    self.tasks[k],
                    representation=Real_value_representation,
                )
            )
        for k in range(len(self.tasks)):
            pop[k].init_pop()
        # Loop
        FE = 0
        while FE < self.maxFE:
            for i in range(self.tasks_len):
                # Evolve pop_i, update FE
                # pop[i] = GA(pop[i], n_gen, self.tasks[i], self.representation, plot_progress=False)
                pop[i] = DE(
                    pop[i],
                    n_gen,
                    self.tasks[i],
                    self.representation,
                    type_de="DE/rand/l",
                    plot_progress=False,
                )
                FE += n_gen
                # Randomly select a source task j (j != i) in {1, ..., K}
                j = i
                while j == i:
                    j = random.randint(0, self.tasks_len - 1)
                task_i = self.tasks[i]
                task_j = self.tasks[j]
                # Execute CTM, OT, CDT methods
                rand = random.random()
                # CTM + OT
                if rand < self.p_OT[i][j]:
                    # Perform CTM strategy between i task and j task to obtain x_mgb_j
                    x_mgb_j = self.CTM(task_i, task_j, nb, pop[i], pop[j])

                    # Perform OT method between i task and j task to obtain r_OT and update FE
                    D_i = task_i.dimension
                    rOT, pop[i] = self.OT(task_i, task_j, D_i, pop[i], x_mgb_j)
                    FE += 2 ** math.ceil(math.log2(D_i + 1))
                    # Update p_OT_i_j with r_OT
                    self.p_OT[i][j] = 0.95 * self.p_OT[i][j] + 0.05 * rOT
                # CDT
                if rand < self.p_CDT[i][j]:
                    D_i = task_i.dimension
                    D_j = task_j.dimension
                    # Perform CDT method between i task and j task to obtain r_CDT and update FE
                    rCDT, pop[i] = self.CDT(task_i, task_j, D_i, D_j, pop[i], pop[j])

                    # Update p_CDT_i_j with r_CDT
                    FE += D_i * D_j
                    self.p_CDT[i][j] = 0.95 * self.p_CDT[i][j] + 0.05 * rCDT

        for i in range(len(self.tasks)):
            pop[i].individuals = sorted(
                pop[i].individuals,
                key=lambda individual: individual.fitness,
                reverse=True,
            )

        return pop
