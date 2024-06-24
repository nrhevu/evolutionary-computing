import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.representation import Individual, Permutation_representation
from src.modeling import Task


class City:
    """Class City to store city information"""

    def __init__(self, num: int, x: int, y: int):
        self.num = num
        self.x = x
        self.y = y

    def __repr__(self):
        return f"(City {str(self.num)} : ({str(self.x)}, {str(self.y)}))"

    @staticmethod
    def distance(city_1, city_2):
        return np.sqrt((city_1.x - city_2.x) ** 2 + (city_1.y - city_2.y) ** 2)


## TSP
class TSP(Task):

    def __init__(self, filename, representation):
        super(TSP, self).__init__(filename, representation)

        # Load file
        tsp = open(filename, "r")
        Lines = tsp.readlines()

        self.city_list = []

        for line in Lines[6:]:
            if line == "EOF":
                break
            city_num, city_x, city_y = line.split()
            self.city_list.append(City(int(city_num), int(city_x), int(city_y)))
        # make graph
        self.dimension = len(self.city_list)
        self.graph = np.zeros((self.dimension + 1, self.dimension + 1))
        for i, city_1 in enumerate(self.city_list):
            for city_2 in self.city_list[i:]:
                distance = City.distance(city_1, city_2)
                self.graph[city_1.num][city_2.num] = distance
                self.graph[city_2.num][city_1.num] = distance

    def decode(self, individual):
        # decode from numeric encode, the index of iterator in sorted array
        # by acsending order represents the solution

        if isinstance(individual, Individual):
            gene = individual.gene
        else:
            gene = individual

        if self.representation == Permutation_representation:
            return gene
        else:
            length_gene = len(gene)
            sorted_gene = sorted(gene, reverse=True)

            index_ordered_gene = []

            for i in range(length_gene):
                index_ordered_gene.append(gene.index(sorted_gene[i]) + 1)

            # decode bằng cách loại bỏ đi những thành phố thừa
            decoded_gene = []
            if len(index_ordered_gene) > self.dimension:
                for i in index_ordered_gene:
                    if i <= self.dimension:
                        decoded_gene.append(i)
            elif len(gene) == self.dimension:
                decoded_gene = index_ordered_gene
            else:
                raise Exception("Gene length smaller than task's dimension")

            return decoded_gene

    def print_graph(self):
        dimension = self.graph.shape[0]
        col = []
        for i in range(1, dimension):
            col.append(i)
        # store distance matrix in dataframe to print out nicely
        df = pd.DataFrame(self.graph[1:, 1:], columns=col, index=col)
        print("Distance of cites : \dimension" + str(df))

    # override
    def compute_fitness(self, individual):
        solution = self.decode(individual)

        route_distance = 0
        if len(solution) != self.dimension:
            raise Exception("Route length not equal to number of cities")
        for i in range(0, self.dimension):
            from_city = solution[i]
            to_city = None
            if i + 1 < len(solution):
                to_city = solution[i + 1]
            else:
                to_city = solution[0]
            route_distance += self.graph[from_city][to_city]

        return -float(route_distance)

    def compute_distance(self, individual):
        return -self.compute_fitness(individual)

    def get_len_gene(self):
        return self.dimension

    def plot_route(self, individual):
        solution = self.decode(individual.gene)

        plt.figure(figsize=(20, 10))
        x = [self.city_list[i - 1].x for i in solution]
        y = [self.city_list[i - 1].y for i in solution]
        x1 = [x[0], x[-1]]
        y1 = [y[0], y[-1]]
        plt.plot(x, y, "b", x1, y1, "b")
        plt.scatter(x, y)

        for i, txt in enumerate(solution):
            plt.annotate(txt, (x[i], y[i]), horizontalalignment="center")

        plt.show()
        return

    def show_result(self, individual):
        print("Route distance: %d" % (self.compute_distance(individual)))
        print("Route plot: ")
        self.plot_route(individual)
