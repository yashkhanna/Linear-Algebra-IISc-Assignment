import networkx
import numpy
import matplotlib.pyplot
import sys
import math
import os
import warnings


##
## Utility Functions
##
def matrix_add(a, b):
    size = len(a)
    c = [[float(0) for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(0, size):
            c[i][j] = a[i][j] + b[i][j]
    return c


# Compute (B^t x A)
def matrix_vector_mult(a, b):
    size = len(a)
    c = init_vector(size)
    for i in range(0, size):
        for j in range(0, size):
            c[i] += a[i][j] * float(b[j])
    return c


def matrix_multiply(a, b):
    size = len(a)
    c = init_square_matrix(size)
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                c[i][j] += a[i][k] * float(b[k][j])
    return c


# Compute (V x V^T)
def vector_vectortranspose_multiply(a, b):
    size = len(a)
    c = init_square_matrix(size)
    for i in range(0, size):
        for j in range(0, size):
            c[i][j] += a[i] * float(b[j])
    return c


def vector_scalar_mult(a, scalar):
    size = len(a)
    c = init_vector(size)
    for i in range(0, size):
        if a[i] != 0:
            c[i] = a[i] * float(scalar)
    return c


def vector_vector_mult(a, b):
    size = len(a)
    c = float(0)
    for i in range(0, size):
        c += a[i] * float(b[i])
    return c


def matrix_scalar_mult(a, scalar):
    size = len(a)
    c = [[0.0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(0, size):
            if (a[i][j] != 0):
                c[i][j] = a[i][j] * float(scalar)
    return c


def calculate_vertex_degrees(adj_mat, size):
    deg = [0] * size
    for i in range(0, size):
        for j in range(0, size):
            deg[i] += adj_mat[i][j]
    return deg


def trace(adj_mat):
    size = len(adj_mat)
    ans = float(0)
    for i in range(0, size):
        ans += adj_mat[i][i]
    return ans


# Find number of zeroes in list from [st, end]
def is_list_zero(list, st, end):
    ans = True
    for i in range(st, end + 1):
        if abs(list[i]) > 0.0001:
            ans = False
    return ans


# Initialise a vector of "size"
def init_vector(size):
    return [float(0)] * size


def length_vector(vec):
    size = len(vec)
    ans = float(0)
    for i in range(0, size):
        ans += vec[i] ** 2
    return math.sqrt(ans)


def identity(size):
    I = init_square_matrix(size)
    for i in range(0, size):
        I[i][i] = float(1)
    return I


# Initialise a matrix of "size X size"
def init_square_matrix(size):
    return [[float(0) for x in range(size)] for y in range(size)]


def compare(a, b):
    return abs(a - b) < 0.00001


# Print Matrix in row-major form using in the given indices
def print_matrix(matrix, row_start, row_end, col_start, col_end, opfile):
    for i in range(row_start, row_end + 1):
        for j in range(col_start, col_end + 1):
            opfile.write(str(round(matrix[i][j], 5)) + " ")
            print round(matrix[i][j], 5),
        opfile.write("\n")
        print ""


########################################################################################################################


# Return the laplacian and normalized laplacian
def laplacian(adj_mat, size, opfile):
    lap = init_square_matrix(size)
    lap_norm = init_square_matrix(size)
    deg = calculate_vertex_degrees(adj_mat, size)

    for i in range(0, size):
        for j in range(0, size):
            if adj_mat[i][j] == 1:
                lap[i][j] = float(-1)
            elif i == j:
                lap[i][j] = float(deg[i])

    for i in range(0, size):
        for j in range(0, size):
            if adj_mat[i][j] == 1:
                lap_norm[i][j] = float(-1.0) / float(math.sqrt(float(deg[i]) * float(deg[j])))
            elif i == j:
                if deg[i] == 0:
                    lap_norm[i][j] = float(0)
                else:
                    lap_norm[i][j] = float(1)

    print "Adjacency Matrix : "
    opfile.write("Adjacency Matrix : \n")
    print_matrix(adj_mat, 0, size - 1, 0, size - 1, opfile)
    print ""
    opfile.write("\n")

    print "Laplacian Matrix : "
    opfile.write("Laplacian Matrix : \n")
    print_matrix(lap, 0, size - 1, 0, size - 1, opfile)
    print ""
    opfile.write("\n")

    print "Normalized Laplacian Matrix : "
    opfile.write("Normalized Laplacian Matrix : \n")
    print_matrix(lap_norm, 0, size - 1, 0, size - 1, opfile)
    print ""
    opfile.write("\n")

    return lap, lap_norm


# def characteristic_polynomial(mat):
#     size = len(mat)
#     p = [1] * (size + 1)
#     p[0] = float(1)
#     p[1] = trace(mat)
#     temp = init_square_matrix(size)
#
#     for i in range(0, size):
#         for j in range(0, size):
#             temp[i][j] = mat[i][j]
#
#     for i in range(2, size + 1):
#         temp = matrix_multiply(mat, matrix_add(temp, matrix_scalar_mult(identity(size), -p[i - 1])))
#         p[i] = float(1.0 / float(i)) * trace(temp)
#
#     for i in range(1, size + 1):
#         if (p[i] != 0):
#             p[i] = float(-p[i])
#         else:
#             p[i] = 0.0
#     return p


# def eigenvalues(mat):
#     p = characteristic_polynomial(mat)
#     y = numpy.roots(p)
#     x = [float(0)] * len(y)
#     # for i in range(0, len(y)):
#     #     x[i] = float(y[i])
#     real = 0
#     # for i in range(0, len(x)):
#     #     if numpy.isreal(x[i]):
#     #         real += 1
#     return x, real


# def eigenvectormatrix(mat, eigenval):
#     size = len(mat)
#     minLambdaI = matrix_scalar_mult(identity(size), -1.0 * eigenval)
#     mat = matrix_add(mat, minLambdaI)
#     return mat


# def verify_eigen_pair(mat, eigenvalue, eigenvector):
#     size = len(mat)
#     one = matrix_vector_mult(mat, eigenvector)
#     two = vector_scalar_mult(eigenvector, eigenvalue)
#     hits = 0
#     for i in range(0, size):
#         if abs(one[i] - two[i]) < 0.0001:
#             hits = hits + 1
#     return hits


def task4(mat, opfile):
    print "\nTask 4: "
    opfile.write("\nTask 4: \n")
    size = len(mat)
    lap, lap_norm = laplacian(mat, size, opfile)
    # eigen_values, real_ct = eigenvalues(lap)

    # eigen_vectors = []
    # for i in range(0, len(eigen_values)):
    #     eigen_vectors.append(eigenvectors(lap, eigen_values[i]))
    #
    # norm_eigen_vectors = []
    # norm_eigen_vectors = normalize_eigen_vectors(eigen_vectors)

    # print real_ct
    # print ""
    # for i in range(0, len(eigen_vectors)):
    #     print eigen_values[i]
    #     print eigen_vectors[i]
    #     # print norm_eigen_vectors[i]
    #     print ""

    eigen_values = []
    eigen_vectors = []

    temp = init_square_matrix(size)
    for i in range(0, size):
        for j in range(0, size):
            temp[i][j] = lap[i][j]

    for i in range(0, size):
        vec = power_method(temp)
        B = vector_vectortranspose_multiply(vec, vec)
        # l2 = numpy.dot(vec, numpy.dot(temp, vec))
        l = vector_vector_mult(vec, matrix_vector_mult(temp, vec))
        eigen_values.append(l)
        eigen_vectors.append(vec)
        print "Eigen-value #" + str(i + 1) + " : " + str(l)
        opfile.write("Eigen-value #" + str(i + 1) + " : " + str(l) + "\n")
        # print vec
        print "(",
        opfile.write("( ")
        for i in range(0, size):
            print str(vec[i]) + " ",
            opfile.write(str(vec[i]) + "  ")
        opfile.write(")")
        print " )",
        # print_matrix(vec, 0, 0, 0, size - 1, opfile)
        # print matrix_vector_mult(lap, vec)
        # print vector_scalar_mult(vec, l)
        # temp = temp - B * l
        temp = matrix_add(temp, matrix_scalar_mult(B, -l))
        print ""
        opfile.write("\n")

    real_ct = 0
    complex_ct = 0
    for i in range(0, size):
        if numpy.isreal(eigen_values[i]):
            real_ct += 1
        else:
            complex_ct += 1

    print "# of real eigen-values : " + str(real_ct)
    opfile.write("# of real eigen-values : " + str(real_ct) + "\n")
    print "# of complex eigen-values : " + str(complex_ct)
    opfile.write("# of complex eigen-values : " + str(complex_ct) + "\n")
    print "# of real eigen-values : " + str(real_ct + complex_ct)
    opfile.write("# of real eigen-values : " + str(real_ct + complex_ct) + "\n")
    return lap, lap_norm, eigen_values, eigen_vectors, real_ct, complex_ct


def task5(eigen_values, eigen_vectors, opfile):
    print "\nTask 5: "
    opfile.write("\nTask 5: \n")
    size = len(eigen_values)

    smallest = size - 1
    sec_smallest = size - 2

    for i in range(size - 2, -1, -1):
        if not compare(round(abs(eigen_values[i]), 5), round(abs(eigen_values[smallest]), 5)):
            sec_smallest = i
            break

    if smallest >= 0:
        print "Smallest Eigen-value : " + str(eigen_values[smallest])
        opfile.write("Smallest Eigen-value : " + str(eigen_values[smallest]) + "\n")

    if sec_smallest >= 0:
        print "Second Smallest Eigen-value : " + str(eigen_values[sec_smallest])
        opfile.write("Second Smallest Eigen-value : " + str(eigen_values[sec_smallest]) + "\n")

    if smallest >= 0:
        vec = vector_scalar_mult(eigen_vectors[smallest], 1.0 / length_vector(eigen_vectors[smallest]))
        val = eigen_values[smallest]
        print "Absolute difference between eigen-value and zero : " + str(val)
        opfile.write("Absolute difference between eigen-value and zero : " + str(val) + "\n")
        diff = 0
        const = 1.0 / math.sqrt(size)
        for i in range(0, size):
            diff += float(abs(const - abs(vec[i])))
        print "Absolute difference (component wise) between the smallest eigen-vector and vector of ones (Both normalized) : " + str(
            diff)
        opfile.write(
            "Absolute difference (component wise) between the smallest eigen-vector and vector of ones (Both normalized) : " + str(
                diff) + "\n")
    return sec_smallest


def task6(eigen_values, eigen_vectors, graph, labels, second_smallest_idx, opfile):
    print "\nTask 6: "
    opfile.write("\nTask 6: \n")
    print "Clustering w.r.t second smallest eigen value : "
    opfile.write("Clustering w.r.t second smallest eigen value : \n")
    print eigen_values[second_smallest_idx]
    opfile.write(str(eigen_values[second_smallest_idx]) + "\n")
    # print eigen_vectors[second_smallest_idx]
    # opfile.write(str(eigen_vectors[second_smallest_idx]) + "\n")
    size = len(eigen_values)

    print "(",
    opfile.write("( ")
    for i in range(0, size):
        print str(eigen_vectors[second_smallest_idx][i]) + " ",
        opfile.write(str(eigen_vectors[second_smallest_idx][i]) + "  ")
    opfile.write(")")
    print " )",

    pos = []
    neg = []
    # print eigen_vectors2[size - 2]
    for i in range(0, size):
        if eigen_vectors[second_smallest_idx][i] >= 0:
            pos.append(unicode(i + 1))
        else:
            neg.append(unicode(i + 1))

    print "\nHouse 1: ",
    opfile.write("\nHouse 1: ")
    for i in range(0, len(pos)):
        # print pos[i]
        print labels.get(pos[i]) + ", ",
        opfile.write(labels.get(pos[i]) + ", ")

    print ""
    opfile.write("\n")
    print "House 2: ",
    opfile.write("House 2: ")
    for i in range(0, len(neg)):
        # print neg[i]
        print labels.get(neg[i]) + ", ",
        opfile.write(labels.get(neg[i]) + ", ")
    fig3 = matplotlib.pyplot.figure()
    # position = networkx.spring_layout(graph, k=200.0, iterations=10000)
    # position = networkx.circular_layout(graph)
    # position = networkx.fruchterman_reingold_layout(graph, scale=100)
    position = networkx.kamada_kawai_layout(graph, scale=150)
    networkx.draw_networkx_nodes(graph, pos=position, node_size=1200, nodelist=pos, node_color='r', node_shape='s')
    networkx.draw_networkx_nodes(graph, pos=position, node_size=1200, nodelist=neg, node_color='g', node_shape='o')
    networkx.draw_networkx_edges(graph, pos=position, alpha=0.5, width=3)
    # networkx.draw_networkx_edges(graph, pos=networkx.spring_layout(graph),
    #                              edgelist=[(1, 2)], width=8, alpha=0.5, edge_color='r')
    networkx.draw_networkx_labels(graph, pos=position, labels=labels, font_size=8)
    matplotlib.pyplot.axis('off')
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('output_plots/problem_1_task_6.png')
    matplotlib.pyplot.close(fig3)
    print ""
    opfile.write("\n")
    return


def bonus3(eigen_values, eigen_vectors, graph, labels, opfile):
    print "\nBonus 3: "
    opfile.write("\nBonus 3: \n")
    print "Clustering w.r.t second largest eigen value : "
    opfile.write("Clustering w.r.t second largest eigen value : \n")

    size = len(eigen_values)
    pos = []
    neg = []

    largest = 0
    sec_largest = 1

    for i in range(1, size):
        if not compare(round(abs(eigen_values[i]), 5), round(abs(eigen_values[largest]), 5)):
            sec_largest = i
            break

    print eigen_values[sec_largest]
    opfile.write(str(eigen_values[sec_largest]) + "\n")
    # print eigen_vectors[sec_largest]
    print "(",
    opfile.write("( ")
    for i in range(0, size):
        print str(eigen_vectors[sec_largest][i]) + " ",
        opfile.write(str(eigen_vectors[sec_largest][i]) + "  ")
    opfile.write(")")
    print " )",

    for i in range(0, size):
        if eigen_vectors[sec_largest][i] >= 0:
            pos.append(unicode(i + 1))
        else:
            neg.append(unicode(i + 1))

    print "\nHouse 1: ",
    opfile.write("\nHouse 1: ")
    for i in range(0, len(pos)):
        # print pos[i]
        print labels.get(pos[i]) + ", ",
        opfile.write(labels.get(pos[i]) + ", ")

    print ""
    opfile.write("\n")
    print "House 2: ",
    opfile.write("House 2: ")
    for i in range(0, len(neg)):
        # print neg[i]
        print labels.get(neg[i]) + ", ",
        opfile.write(labels.get(neg[i]) + ", ")
    fig4 = matplotlib.pyplot.figure()
    # position = networkx.spring_layout(graph, k=200.0, iterations=10000)
    # position = networkx.circular_layout(graph)
    position = networkx.kamada_kawai_layout(graph)
    networkx.draw_networkx_nodes(graph, pos=position, node_size=1200, nodelist=pos, node_color='r', node_shape='s')
    networkx.draw_networkx_nodes(graph, pos=position, node_size=1200, nodelist=neg, node_color='g', node_shape='o')
    networkx.draw_networkx_edges(graph, pos=position, alpha=0.5, width=3)
    # networkx.draw_networkx_edges(graph, pos=networkx.spring_layout(graph),
    #                              edgelist=[(1, 2)], width=8, alpha=0.5, edge_color='r')
    networkx.draw_networkx_labels(graph, pos=position, labels=labels, font_size=8)
    matplotlib.pyplot.axis('off')
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('output_plots/problem_1_task_bonus_3.png')
    matplotlib.pyplot.close(fig4)
    return


# Read GML file and outputs
def parse_input(ipfilepath):
    graph = networkx.read_gml(ipfilepath)
    size = graph.number_of_nodes()
    e = graph.size()
    list = networkx.to_dict_of_dicts(graph)
    adj_mat = [[0 for x in range(size)] for y in range(size)]
    for i in list:
        for j in list[i]:
            adj_mat[int(i) - 1][int(j) - 1] = 1
            adj_mat[int(j) - 1][int(i) - 1] = 1
    labels = networkx.get_node_attributes(graph, 'name')

    return graph, adj_mat, labels, size, e


def power_method(mat):
    size = len(mat)
    iterations = 10000
    vec = numpy.random.rand(size)

    for _ in range(iterations):
        vec = matrix_vector_mult(mat, vec)
        vec = vector_scalar_mult(vec, 1.0 / length_vector(vec))

    return vec


# Solves all pair shortest path via Floyd Warshall Algorithm
def floydWarshall(mat):
    size = len(mat)
    dist = init_square_matrix(size)

    for i in range(0, size):
        for j in range(0, size):
            if mat[i][j] == 0:
                if i == j:
                    dist[i][j] = 0
                else:
                    dist[i][j] = sys.maxint
            else:
                dist[i][j] = mat[i][j]

    for k in range(0, size):
        for i in range(0, size):
            for j in range(0, size):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # print_matrix(dist, 0, size - 1, 0, size - 1)
    return dist


# Calculate Harmonic Centrality of the graph
# https://en.wikipedia.org/wiki/Centrality#Harmonic_centrality
def vertex_centrality(mat, dist):
    size = len(mat)
    vec = init_vector(size)
    for i in range(0, size):
        temp = float(0)
        for j in range(0, size):
            if i == j or dist[i][j] == sys.maxint or dist[i][j] == float(0):
                continue
            else:
                temp += 1.0 / dist[j][i]
        vec[i] = temp

    for i in range(0, size):
        vec[i] /= (size - 1)

    return vec


def task3(mat, labels, opfile):
    print "\nTask 3: "
    opfile.write("\nTask 3: \n")
    size = len(mat)
    betweenness = init_square_matrix(size)

    for i in range(0, size):
        S, P, sigma = bfs(mat, i)
        betweenness = update_weights(mat, betweenness, S, P, sigma, i)

    if size > 1:
        for i in range(0, size):
            for j in range(0, size):
                betweenness[i][j] /= (size) * (size - 1)

    high_i = 0
    high_j = 0
    for i in range(0, size):
        for j in range(0, size):
            if betweenness[i][j] > betweenness[high_i][high_j]:
                high_i = i
                high_j = j

    print "The most central edge is between : " + "Node " + str(high_i) + " (" + str(
        labels.get(unicode(high_i + 1))) + ")" + " and " + "Node " + str(high_j) + " (" + str(
        labels.get(unicode(high_j + 1))) + ")"
    opfile.write("The most central edge is between : " + "Node " + str(high_i) + " (" + str(
        labels.get(unicode(high_i + 1))) + ")" + " and " + "Node " + str(high_j) + " (" + str(
        labels.get(unicode(high_j + 1))) + ")\n")

    return high_i, high_j


def update_weights(mat, betweenness, S, P, sigma, s):
    size = len(mat)
    delta = init_vector(size)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            betweenness[v][w] += c
            betweenness[w][v] += c
            delta[v] += c
    return betweenness


def bfs(mat, src):
    S = []
    parents = {}
    size = len(mat)
    for i in range(0, size):
        parents[i] = []

    visited = []

    for i in range(0, size):
        visited.append(False)

    dist = init_vector(size)
    sigma = init_vector(size)
    sigma[src] = 1.0
    queue = [src]

    while len(queue) > 0:
        v = queue.pop(0)
        visited[v] = True
        S.append(v)
        for w in range(0, size):
            if mat[v][w] == 0 or v == w:
                continue
            break_loop = False
            for x in parents[v]:
                if x == w:
                    break_loop = True
            if break_loop:
                continue
            else:
                if dist[w] == 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                    sigma[w] += sigma[v]
                    parents[w].append(v)
                else:
                    sigma[w] += sigma[v]
                    parents[w].append(v)
    return S, parents, sigma


# node centrality using any one from the different standard node centrality measures.
def task2(adj_mat, labels, opfile):
    print "\nTask 2: "
    opfile.write("\nTask 2: \n")
    size = len(adj_mat)
    dist = floydWarshall(adj_mat)
    node_centrality = vertex_centrality(adj_mat, dist)

    sorted_node_centrality = sorted(node_centrality, reverse=True)
    if len(sorted_node_centrality) >= 2:
        val1 = sorted_node_centrality[0]
        val2 = sorted_node_centrality[1]
    elif len(sorted_node_centrality) == 1:
        val1 = sorted_node_centrality[0]
        val2 = -1
    else:
        val1 = -1
        val2 = -1

    index1 = -1
    index2 = -1
    for i in range(0, size):
        if index1 == -1 and node_centrality[i] == val1:
            index1 = i
        elif index2 == -1 and node_centrality[i] == val2:
            index2 = i

    # print node_centrality
    # print sorted_node_centrality

    # print index1, index2
    if index1 != -1:
        print "Central Node #1 (Highest) : " + " Node " + str(index1) + " (" + str(
            labels.get(unicode(index1 + 1))) + ")"
        opfile.write("Central Node #1 (Highest) : " + " Node " + str(index1) + " (" + str(
            labels.get(unicode(index1 + 1))) + ")\n")
    if index2 != -1:
        print "Central Node #2 (Second-Highest) : " + " Node " + str(index2) + " (" + str(
            labels.get(unicode(index2 + 1))) + ")"
        opfile.write("Central Node #2 (Second-Highest) : " + " Node " + str(index2) + " (" + str(
            labels.get(unicode(index2 + 1))) + ")\n")

    fig2 = matplotlib.pyplot.figure()
    matplotlib.pyplot.title("Figure 2: Node Centrality Histogram")
    matplotlib.pyplot.ylabel("Count of Vertices")
    matplotlib.pyplot.xlabel("Node Centrality")
    matplotlib.pyplot.hist(node_centrality, bins=numpy.arange(min(node_centrality), max(node_centrality) + 0.01, 0.01),
                           facecolor='blue', alpha=0.5, rwidth=0.5)
    matplotlib.pyplot.savefig('output_plots/problem_1_task_2.png')
    matplotlib.pyplot.close(fig2)
    return index1, index2


# To obtain the degree distribution of the nodes.
def task1(mat, labels, opfile):
    print "Task 1: "
    opfile.write("Task 1: \n")
    size = len(mat)

    deg = calculate_vertex_degrees(mat, size)
    length = max(deg) + 1
    degrees = [0] * (length)
    count = [float(0)] * (length)

    for i in range(0, length):
        degrees[i] = i

    for i in range(0, size):
        count[deg[i]] += 1

    # sum = float(0)
    for i in range(0, length):
        count[i] /= size
        # sum += count[i]
    # print sum

    print "Degree Distribution : "
    opfile.write("Degree Distribution : \n")
    for i in range(0, size):
        print "Node " + str(i) + " (" + str(labels.get(unicode(i + 1))) + ") : " + str(deg[i])
        opfile.write("Node " + str(i) + " (" + str(labels.get(unicode(i + 1))) + ") : " + str(deg[i]) + "\n")

    fig1 = matplotlib.pyplot.figure()
    matplotlib.pyplot.stem(degrees, count, '--r')
    matplotlib.pyplot.yticks(numpy.arange(0, 1.2, 0.1))
    matplotlib.pyplot.xticks(numpy.arange(0, length + 2, 1))
    matplotlib.pyplot.xlabel('Degree')
    matplotlib.pyplot.ylabel('Fraction of Vertices')
    matplotlib.pyplot.title('Figure 1: Degree Distribution')
    matplotlib.pyplot.savefig('output_plots/problem_1_task_1.png')
    matplotlib.pyplot.close(fig1)
    return


def mean(data):
    ans = float(0)
    size = len(data)
    for i in range(0, size):
        ans += data[i]
    ans /= size
    return ans


def variance(data):
    size = len(data)
    data_squared = []
    for i in range(0, size):
        data_squared.append(data[i] ** 2)

    return mean(data_squared) - mean(data) ** 2


def task7(eigen_values, eigen_vectors, labels, second_smallest_idx, opfile):
    print "\nTask 7: "
    opfile.write("\nTask 7: \n")
    print "Clustering w.r.t second smallest eigen value : "
    opfile.write("Clustering w.r.t second smallest eigen value : \n")
    size = len(eigen_values)
    pos = []
    eigen_pos = []
    neg = []
    eigen_neg = []
    for i in range(0, size):
        if eigen_vectors[second_smallest_idx][i] >= 0:
            pos.append(unicode(i + 1))
            eigen_pos.append(eigen_vectors[size - 2][i])
        else:
            neg.append(unicode(i + 1))
            eigen_neg.append(eigen_vectors[size - 2][i])

    temp = True
    if variance(eigen_pos) > variance(eigen_neg):
        temp = True
    else:
        temp = False

    print "House 1: ",
    opfile.write("House 1: ")
    for i in range(0, len(pos)):
        # print pos[i]
        print labels.get(pos[i]) + ", ",
        opfile.write(labels.get(pos[i]) + ", ")

    print ""
    opfile.write("\n")
    print "House 2: ",
    opfile.write("House 2: ")
    for i in range(0, len(neg)):
        # print neg[i]
        print labels.get(neg[i]) + ", ",
        opfile.write(labels.get(neg[i]) + ", ")

    print "\nResult: (Based on the variability measure)",
    opfile.write("\nResult: (Based on the variability measure)")
    if temp:
        print " House of Algebrician should join House 1"
        opfile.write(" House of Algebrician should join House 1\n")
    else:
        print " House of Algebrician should join House 2"
        opfile.write(" House of Algebrician should join House 2\n")
    return


def bonus1(lap, graph, labels, opfile):
    print "\nBonus 1: "
    opfile.write("\nBonus 1: \n")
    print "Clustering w.r.t second smallest eigen value (USING NUMPY): "
    opfile.write("Clustering w.r.t second smallest eigen value (USING NUMPY): \n")
    size = len(lap)
    eigen_values_numpy, eigen_vectors_numpy = numpy.linalg.eig(lap)
    sorted = numpy.argsort(eigen_values_numpy)
    # print eigen_vectors_numpy[:, sorted[0]]
    # print eigen_vectors_numpy[:, sorted[1]]

    smallest = sorted[0]
    sec_smallest = sorted[1]

    for i in range(1, size):
        if not compare(round(abs(eigen_values_numpy[sorted[i]]), 5), round(abs(eigen_values_numpy[smallest]), 5)):
            sec_smallest = i
            break

    print eigen_values_numpy[sorted[sec_smallest]]
    opfile.write(str(eigen_values_numpy[sorted[sec_smallest]]) + "\n")
    # print eigen_vectors_numpy[sorted[sec_smallest]]
    print "(",
    opfile.write("( ")
    for i in range(0, size):
        print str(eigen_vectors_numpy[sorted[sec_smallest]][i]) + " ",
        opfile.write(str(eigen_vectors_numpy[sorted[sec_smallest]][i]) + "  ")
    opfile.write(")")
    print " )",

    pos = []
    neg = []
    for i in range(0, size):
        if eigen_vectors_numpy[:, sorted[sec_smallest]][i] >= 0:
            pos.append(unicode(i + 1))
        else:
            neg.append(unicode(i + 1))

    print "\nHouse 1: ",
    opfile.write("\nHouse 1: ")
    for i in range(0, len(pos)):
        # print pos[i]
        print labels.get(pos[i]) + ", ",
        opfile.write(labels.get(pos[i]) + ", ")

    print ""
    opfile.write("\n")
    print "House 2: ",
    opfile.write("House 2: ")
    for i in range(0, len(neg)):
        # print neg[i]
        print labels.get(neg[i]) + ", ",
        opfile.write(labels.get(neg[i]) + ", ")
    fig5 = matplotlib.pyplot.figure()
    # position = networkx.spring_layout(graph, k=200.0, iterations=10000)
    # position = networkx.circular_layout(graph)
    position = networkx.kamada_kawai_layout(graph)
    networkx.draw_networkx_nodes(graph, pos=position, node_size=1200, nodelist=pos, node_color='r', node_shape='s')
    networkx.draw_networkx_nodes(graph, pos=position, node_size=1200, nodelist=neg, node_color='g', node_shape='o')
    networkx.draw_networkx_edges(graph, pos=position, alpha=0.5, width=3)
    # networkx.draw_networkx_edges(graph, pos=networkx.spring_layout(graph),
    #                              edgelist=[(1, 2)], width=8, alpha=0.5, edge_color='r')
    networkx.draw_networkx_labels(graph, pos=position, labels=labels, font_size=8)
    matplotlib.pyplot.axis('off')
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('output_plots/problem_1_task_bonus_1.png')
    matplotlib.pyplot.close(fig5)
    print ""
    opfile.write("\n")
    return


# Main function
def main():
    warnings.filterwarnings("ignore")
    ipfilepath = sys.argv[1]
    plots_path = "./output_plots"
    data_path = "./output_data"
    opfile_path = "./output_data/output_problem1.txt"
    access_rights = 0o755

    try:
        if not os.path.exists(plots_path):
            os.mkdir(plots_path, access_rights)
    except OSError:
        exit(1)

    try:
        if not os.path.exists(data_path):
            os.mkdir(data_path, access_rights)
    except OSError:
        exit(1)

    opfile = open(opfile_path, "w")
    graph, adj_mat, labels, size, edges = parse_input(ipfilepath)

    task1(adj_mat, labels, opfile)
    task2(adj_mat, labels, opfile)
    task3(adj_mat, labels, opfile)

    lap, lap_norm, eigen_values, eigen_vectors, real_ct, complex_ct = task4(adj_mat, opfile)

    second_smallest_idx = task5(eigen_values, eigen_vectors, opfile)

    task6(eigen_values, eigen_vectors, graph, labels, second_smallest_idx, opfile)
    task7(eigen_values, eigen_vectors, labels, second_smallest_idx, opfile)
    bonus1(lap, graph, labels, opfile)
    bonus3(eigen_values, eigen_vectors, graph, labels, opfile)
    opfile.close()
    return 0


if __name__ == "__main__": main()
