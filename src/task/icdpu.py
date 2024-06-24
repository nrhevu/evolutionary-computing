from src.representation import Individual
from src.task import Task


class Edge:
    def __init__(self, start_v, end_v, weight, domain):
        self.start_v = start_v
        self.end_v = end_v
        self.weight = weight
        self.domain = domain

    def __repr__(self):
        return "Edge: (%d -> %d), weight: %d, domain: %d" % (
            self.start_v,
            self.end_v,
            self.weight,
            self.domain,
        )


class IDPCDU(Task):
    def __init__(self, filename, representation):
        super(IDPCDU, self).__init__(filename, representation)

        idpcdu = open(filename, "r")
        Lines = idpcdu.readlines()
        # define start and end vertex, dimension and domain
        self.start = int(Lines[1].split()[0])
        self.end = int(Lines[1].split()[1])

        self.dimension = int(Lines[0].split()[0])
        self.domain = int(Lines[0].split()[1])

        # create items list
        self.graph = [
            [[] for j in range(self.dimension + 1)] for i in range(self.dimension + 1)
        ]
        for id, line in enumerate(Lines[2:]):
            start_v, end_v, weight, domain = line.split()
            start_v, end_v, weight, domain = (
                int(start_v),
                int(end_v),
                int(weight),
                int(domain),
            )
            self.graph[start_v][end_v].append(Edge(start_v, end_v, weight, domain))
        # remove redundant edge, only keep edge with the smallest weight of each domain
        graph_d = [
            [[] for j in range(self.dimension + 1)] for i in range(self.dimension + 1)
        ]
        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                domain_id = [-1] * ((self.domain) + 1)
                domain_weight = [99999] * ((self.domain) + 1)
                for k in range(len(self.graph[i][j])):
                    w = self.graph[i][j][k].weight
                    d = self.graph[i][j][k].domain
                    if w < domain_weight[d]:
                        domain_id[d] = k
                        domain_weight[d] = w
                # add smallest domain weight edge to new graph
                domain_id = sorted(domain_id)
                for z in range(self.domain + 1):
                    if domain_id[z] == -1:
                        continue
                    else:
                        graph_d[i][j].append(self.graph[i][j][domain_id[z]])

        self.graph = graph_d

    def decode(self, individual):
        if isinstance(individual, Individual):
            gene = individual.gene
        else:
            gene = individual
        length_gene = len(gene[0])

        route = []

        passed_point = [False] * (self.dimension + 1)
        passed_point[0] = True
        passed_domain = [False] * (self.domain + 1)

        # find the order of priority vertex
        priority_point = []
        sorted_gene = [(gene[0][i], i) for i in range(length_gene)]
        sorted_gene = sorted(sorted_gene, key=lambda a: a[0], reverse=True)
        for i in range(len(sorted_gene)):
            priority_point.append(sorted_gene[i][1] + 1)

        current_v = self.start
        current_d = -1
        passed_point[current_v] = True
        path = []
        while current_v != self.end:
            # Find next v according to priority_point
            to = -1
            possible_path = []
            for i in priority_point:
                if passed_point[i] == False:
                    to = i
                # Choose path
                # find possible path
                for k in range(len(self.graph[current_v][to])):
                    if passed_domain[self.graph[current_v][to][k].domain]:
                        continue
                    else:
                        possible_path.append(self.graph[current_v][to][k])
                if len(possible_path) == 0:
                    continue
                break
            if len(possible_path) == 0 or i == -1:
                break
            # Choose path by taking modulo of gene and length of possible path
            edge = possible_path[gene[1][to - 1] % len(possible_path)]
            path.append(edge)

            # update current pos and passed vertex and domain
            current_v = to
            passed_point[current_v] = True
            if edge.domain != current_d:
                passed_domain[current_d] = True
            current_d = edge.domain
        return path

    def compute_fitness(self, individual):
        solution = self.decode(individual)
        result = 0
        # if can't reach the end point fitness = 0
        if solution[-1].end_v != self.end:
            result = 999999999
        else:
            # else sum all the weight and take the inverse
            for i in range(len(solution)):
                result += solution[i].weight

        return -result

    def get_len_gene(self):
        return self.dimension

    def show_result(self):
        pass
