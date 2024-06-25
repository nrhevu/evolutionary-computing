from src.modeling.representation import Individual, MFEAIndividual
from src.modeling.task import Task


class Item:
    def __init__(self, id, weight, value):
        self.id = id
        self.weight = weight
        self.value = value

    def __repr__(self):
        return "item id: %d - weight: %d, value: %d" % (
            self.id,
            self.weight,
            self.value,
        )


class Knapsack(Task):
    def __init__(self, filename, representation):
        super(Knapsack, self).__init__(filename, representation)

        knapsack = open(filename, "r")
        Lines = knapsack.readlines()
        # define dimension and capacity
        self.dimension = int(Lines[1])
        self.capacity = int(Lines[2])

        # create items list
        self.items_list = []
        for id, line in enumerate(Lines[4:]):
            weight, value = line.split()
            weight, value = int(weight), int(value)
            self.items_list.append(Item(id + 1, weight, value))

    def decode(self, individual):
        if isinstance(individual, Individual) or isinstance(individual, MFEAIndividual):
            gene = individual.gene
        else:
            gene = individual

        length_gene = len(gene)
        avg_value = sum(gene) / length_gene

        decoded_gene = [-1 for i in range(self.dimension)]

        # take items whose gene value higher than the avg value
        for i in range(self.dimension):
            decoded_gene[i] = 1 if (gene[i] > avg_value) else 0

        return decoded_gene

    def compute_fitness(self, individual):
        solution = self.decode(individual)
        result = 0
        # if exceed capacity, then fitness = 0
        if self.get_weight(individual) > self.capacity:
            return 0
        for i in range(len(solution)):
            result += self.items_list[i].value * solution[i]

        return result

    def get_weight(self, individual):
        solution = self.decode(individual)
        result = 0
        for i in range(len(solution)):
            result += self.items_list[i].weight * solution[i]

        return result

    def get_value(self, individual):
        solution = self.decode(individual)
        result = 0
        for i in range(len(solution)):
            result += self.items_list[i].value * solution[i]

        return result

    def get_len_gene(self):
        return self.dimension
    
    def check_individual_valid(self, gene):
        # solution = self.decode(individual)
        # result = 0
        # for i in range(len(solution)):
        #     result +=  self.items_list[i].weight * solution[i]
        #
        # return not (result <= capacity)

        return False

    def make_individual_valid(self, gene):
        pass

    def show_result(self, individual):
        solution = self.decode(individual)
        solution_items = []
        # get list of items
        for i in range(self.dimension):
            if solution[i] == 1:
                solution_items.append(self.items_list[i])

        print("items id :" + str([item.id for item in solution_items]), end=", ")
        print(
            "total weight: %d, total value: %d"
            % (self.get_weight(individual), self.get_value(individual))
        )
