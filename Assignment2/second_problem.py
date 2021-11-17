import numpy
import matplotlib.pyplot
import sys
import math
import os
import sklearn
import warnings

if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier


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


def matrix_multiply2(a, b):
    rows = len(a)
    common = len(a[0])
    cols = len(b[0])
    c = [[float(0) for x in range(cols)] for y in range(rows)]
    for i in range(0, rows):
        for j in range(0, cols):
            for k in range(0, common):
                c[i][j] += a[i][k] * b[k][j]
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
        if (a[i] != 0):
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


def degree(adj_mat, size):
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


def vector_vector_dist_squared(vec1, vec2):
    size = len(vec1)
    ans = float(0)
    for i in range(0, size):
        ans += (vec1[i] - vec2[i]) ** 2
    return ans


def euclidean_dist(vec1, vec2):
    return math.sqrt(vector_vector_dist_squared(vec1, vec2))


def vector_vector_add(a, b):
    size = len(a)
    c = init_vector(size)
    for i in range(0, size):
        c[i] += a[i] + b[i]
    return c


def identity(size):
    I = init_square_matrix(size)
    for i in range(0, size):
        I[i][i] = float(1)
    return I


# Initialise a matrix of "size X size"
def init_square_matrix(size):
    return [[float(0) for x in range(size)] for y in range(size)]


def compare(a, b, error):
    return abs(a - b) < error


# Print Matrix in row-major form using in the given indices
def print_matrix(matrix, row_start, row_end, col_start, col_end, opfile):
    for i in range(row_start, row_end + 1):
        for j in range(col_start, col_end + 1):
            if round(matrix[i][j], 3) == -0.0:
                matrix[i][j] = 0.0
            opfile.write(str(round(matrix[i][j], 5)) + " ")
            print round(matrix[i][j], 5),
        opfile.write("\n")
        print ""


def is_zero_vector(vec):
    return length_vector(vec) == 0


def transpose(mat):
    rows = len(mat)
    cols = len(mat[0])
    temp = [[float(0) for x in range(rows)] for y in range(cols)]

    for i in range(0, rows):
        for j in range(0, cols):
            temp[j][i] = mat[i][j]

    return temp


########################################################################################################################

# Return projection of y on u
def projection(y, u):
    alpha = vector_vector_mult(y, u) / vector_vector_mult(u, u)
    p1 = vector_scalar_mult(u, alpha)
    p2 = vector_vector_add(y, vector_scalar_mult(u, -1))
    return p1, p2


def gram_schmidt(vectors, count, size, contains_zero_vector):
    ortho_vectors = []
    for i in range(0, count):
        temp2 = vectors[i]
        for j in range(0, i):
            proj, _ = projection(vectors[i], ortho_vectors[j])
            proj = vector_scalar_mult(proj, -1)
            temp2 = vector_vector_add(temp2, proj)
        ortho_vectors.append(temp2)

    if contains_zero_vector == True:
        ortho_vectors.append(init_vector(size))

    return ortho_vectors


def clean_input_gram_schmidt(vectors, count, size):
    unit_vectors = [-1] * count
    contains_zero_vector = False
    for i in range(0, count):
        if not is_zero_vector(vectors[i]):
            unit_vectors[i] = vector_scalar_mult(vectors[i], 1.0 / length_vector(vectors[i]))
        else:
            contains_zero_vector = True

    unique = [False] * count

    for i in range(0, count):
        if unit_vectors[i] == -1:
            continue
        temp = True
        for j in range(i + 1, count):
            if unit_vectors[j] == -1:
                continue
            else:
                count2 = 0
                for k in range(0, size):
                    if compare(unit_vectors[i][k], unit_vectors[j][k], 0.0001):
                        count2 += 1
                if count2 == size:
                    temp = False
        unique[i] = temp

    reduced_set = []

    for i in range(0, count):
        if unique[i] == True:
            reduced_set.append(vectors[i])

    return reduced_set, len(reduced_set), contains_zero_vector


def parse_input_matrix_form(filepath):
    vectors = []
    count = 0
    size = 0
    with open(filepath, "r") as f:
        for line in f:
            count += 1
            temp = map(float, line.split())
            size = len(temp)
            vectors.append(temp)
    return vectors, count, size


def parse_input(filepath):
    data = []
    labels = []
    count = 0
    dimension = 0
    with open(filepath, "r") as f:
        for line in f:
            labels.append(int(line.split(',')[0]))
            temp = map(float, line.split(',')[1:])
            dimension = len(temp)
            data.append(temp)
            count = count + 1

    return data, labels, count, dimension


# Find the mean vector and covariance of the data
def task1(data, count, dimension):
    mean = init_vector(dimension)
    for i in range(0, dimension):
        temp = 0.0
        for j in range(0, count):
            temp += data[j][i]
        temp = temp / count
        mean[i] = temp

    cov = init_square_matrix(dimension)
    for i in range(0, dimension):
        for j in range(0, dimension):
            # print (i, j)
            temp = 0.0

            for k in range(0, count):
                temp += (data[k][i] - mean[i]) * (data[k][j] - mean[j])

            cov[i][j] = temp / (count - 1)

    return mean, cov


def power_method(mat):
    size = len(mat)
    iterations = 200
    vec = numpy.random.rand(size)

    for i in range(iterations):
        vec = matrix_vector_mult(mat, vec)
        vec = vector_scalar_mult(vec, 1.0 / length_vector(vec))

    return vec


def task2(eigen_values, eigen_vectors, count, dimension):
    clusters = []
    marked = [False] * dimension

    for i in range(0, dimension):
        if not marked[i]:
            temp = []
            temp.append(i)
            marked[i] = True
            for j in range(i + 1, dimension):
                if compare(abs(eigen_values[i][0]), abs(eigen_values[j][0]), 0.01):
                    temp.append(j)
                    marked[j] = True
            clusters.append(temp)

    print len(clusters)
    # for i in range(0, len(clusters)):
    #     print clusters[i]

    # for i in range(0, dimension):
    #     for j in range(i + 1, dimension):
    #         print vector_vector_mult(eigen_vectors[i], eigen_vectors[j])

    return


def task4(eigen_vectors, count, features, data, mean, M):
    data2 = mean_normalized_data(data, mean, features, count, -1)
    eigen_reduced = []
    for i in range(0, M):
        eigen_reduced.append(eigen_vectors[i])

    eigen_reduced_transpose = transpose(eigen_reduced)

    reduced_data = matrix_multiply2(data2, eigen_reduced_transpose)
    # print len(reduced_data)
    # print len(reduced_data[0])
    # print_matrix(reduced_data, 0, len(reduced_data) - 1, 0, len(reduced_data[0]) - 1)

    # reconstructed_data = mean_normalized_data(, mean, features, count, 1)
    # reconstructed_data = matrix_multiply2(reduced_data, eigen_reduced)
    # print len(reconstructed_data)
    # print len(reconstructed_data[0])
    # # gen_image(data[0])
    # gen_image(reconstructed_data[0])
    return reduced_data


def mean_normalized_data(data, mean, features, count, weight):
    data2 = []
    for i in range(0, count):
        data2.append(init_vector(features))

    for j in range(0, features):
        for i in range(0, count):
            data2[i][j] = data[i][j] + weight * mean[0][j]

    return data2


def eigen_values_vectors(mat):
    size = len(mat)

    eigen_values = []
    eigen_vectors = []

    temp = init_square_matrix(size)
    for i in range(0, size):
        for j in range(0, size):
            temp[i][j] = mat[i][j]

    opfile1 = open("./input_data/eigenvalues.txt", "w")
    opfile2 = open("./input_data/eigenvectors.txt", "w")

    for i in range(0, size):
        print "Eigen-value # " + str(i)
        vec = power_method(temp)
        B = vector_vectortranspose_multiply(vec, vec)
        l = vector_vector_mult(vec, matrix_vector_mult(temp, vec))
        eigen_values.append(l)
        eigen_vectors.append(vec)
        opfile1.write(str(l) + "\n")
        for i in range(0, len(vec)):
            opfile2.write(str(vec[i]) + " ")
        opfile2.write("\n")
        print l
        print vec
        temp = matrix_add(temp, matrix_scalar_mult(B, -l))
        print ""

    real_ct = 0
    complex_ct = 0
    for i in range(0, size):
        if numpy.isreal(eigen_values[i]):
            real_ct += 1
        else:
            complex_ct += 1

    print "# of real eigen-values : " + str(real_ct)
    print "# of complex eigen-values : " + str(complex_ct)
    opfile1.close()
    opfile2.close()
    return eigen_values, eigen_vectors, real_ct, complex_ct


def bonus1(arr):
    two_d = (numpy.reshape(arr, (28, 28)) * 255).astype(numpy.uint8)
    matplotlib.pyplot.imshow(two_d, cmap='gray')
    matplotlib.pyplot.show()
    return -1


def task5(data, count, features, eigen_vectors, mean):
    x = [10, 50, 65, 85, 100, 125, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 784]
    y = []

    eigen_vectors_transpose = transpose(eigen_vectors)

    for i in range(0, len(x)):
        print x[i]
        reduced_data = task4(eigen_vectors, count, features, data, mean, x[i])
        data_recovered = matrix_multiply2(reduced_data, eigen_vectors_transpose)
        data_recovered += mean
        data += mean

        error = float(0)
        for j in range(0, count):
            error += vector_vector_dist_squared(data[j], data_recovered[j])

        y.append(error)

    fig5 = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.xlabel('Number of Eigen-vectors used for Reconstruction (M)')
    matplotlib.pyplot.ylabel('Error Value')
    matplotlib.pyplot.title('\"RECONSTRUCTION ERROR\"')
    matplotlib.pyplot.savefig('output_plots/problem_2_task_5.png')
    matplotlib.pyplot.close(fig5)
    return


def bonus2(data, labels, testData, testLabels, M):
    # handle older versions of sklearn
    # Split Data
    (trainData, valData, trainLabels, valLabels) = train_test_split(data, labels, test_size=0.1)

    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k
    kVals = range(1, 30, 2)
    accuracies = []
    scores = []
    print "M = " + str(M)
    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 30, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        model.fit(trainData, trainLabels)

        # evaluate the model and update the accuracies list
        score = model.score(valData, valLabels)
        print("K=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)
        scores.append(score * 100)

    fig5 = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(kVals, scores)
    matplotlib.pyplot.xlabel('K values')
    matplotlib.pyplot.ylabel('Accuracy Percentage')
    title = "K-NN Accuracy for M = " + str(M) + " (Number of Eigen Vectors)"
    matplotlib.pyplot.title(title)
    plot_path = "output_plots/problem_2_task_" + str(M) + ".png"
    matplotlib.pyplot.savefig(plot_path)
    matplotlib.pyplot.close(fig5)

    # find the value of k that has the largest accuracy
    i = int(numpy.argmax(accuracies))
    print("K=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))
    # re-train our classifier using the best k value and predict the labels of the
    # test data
    # model = KNeighborsClassifier(n_neighbors=kVals[i])
    # model.fit(trainData, trainLabels)
    # predictions = model.predict(testData)

    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits
    # print("EVALUATION ON TESTING DATA")
    # print(classification_report(testLabels, predictions))
    return


# def knn_classify_single(train_data, train_labels, test_data, k):
#     distances = []
#     for i in range(len(train_data)):
#         dist = euclidean_dist(test_data, train_data[i])
#         distances.append((i, dist))
#     distances = sorted(distances, key=lambda a_entry: a_entry[1])
#
#     classes = [0] * 15
#     for i in range(0, k):
#         classes[train_labels[distances[i][0]]] += 1
#
#     ans = -1
#     for i in range(0, 15):
#         ans = max(ans, classes[i])
#
#     return ans


def knn_classify(train_data, train_labels, test_data, k):
    predicted_labels = []
    for j in range(0, len(test_data)):
        distances = []
        for i in range(len(train_data)):
            dist = euclidean_dist(test_data[j], train_data[i])
            distances.append((i, dist))
        distances = sorted(distances, key=lambda a_entry: a_entry[1])

        classes = [0] * 10
        for i in range(0, k):
            classes[train_labels[distances[i][0]]] += 1

        ans = 0
        for i in range(0, 10):
            if classes[i] > classes[ans]:
                ans = i

        predicted_labels.append(ans)

        if j % 10 == 0:
            print "Processed " + str((100.0 * j) / len(test_data)) + "% Data"

    return predicted_labels


def knn_accuracy(test_data, test_labels, predicted_labels, opfile):
    correct = [0] * 10
    incorrect = [0] * 10
    total = [0] * 10
    correct_sum = 0
    incorrect_sum = 0
    total_sum = 0
    for i in range(0, len(test_data)):
        if test_labels[i] == predicted_labels[i]:
            correct[test_labels[i]] += 1
            total[test_labels[i]] += 1
            correct_sum += 1
            total_sum += 1
        else:
            incorrect[test_labels[i]] += 1
            total[test_labels[i]] += 1
            incorrect_sum += 1
            total_sum += 1

    print "\nClass-wise Results-\n"
    opfile.write("\n\nClass-wise Results-\n\n")
    for i in range(0, 10):
        print "Class " + str(i)
        print "Correct : " + str(correct[i])
        print "Incorrect : " + str(incorrect[i])
        print "Total : " + str(total[i])
        if total[i] != 0:
            print "Accuracy Percentage : " + str(correct[i] / float(total[i]) * 100)
            print "Error Percentage : " + str(100 - correct[i] / float(total[i]) * 100)
        print ""
        opfile.write("Class " + str(i) + "\n")
        opfile.write("Correct : " + str(correct[i]) + "\n")
        opfile.write("Incorrect : " + str(incorrect[i]) + "\n")
        opfile.write("Total : " + str(total[i]) + "\n")
        if total[i] != 0:
            opfile.write("Accuracy Percentage : " + str(correct[i] / float(total[i]) * 100) + "\n")
            opfile.write("Error Percentage : " + str(100 - correct[i] / float(total[i]) * 100) + "\n")
        opfile.write("\n")

    print "Aggregate Results-\n"
    print "Correct : " + str(correct_sum)
    print "Incorrect : " + str(incorrect_sum)
    print "Total : " + str(total_sum)
    print "Accuracy Percentage : " + str(correct_sum / float(total_sum) * 100)
    print "Error Percentage : " + str(100 - correct_sum / float(total_sum) * 100)
    opfile.write("\n\nAggregate Results-\n")
    opfile.write("Correct : " + str(correct_sum) + "\n")
    opfile.write("Incorrect : " + str(incorrect_sum) + "\n")
    opfile.write("Total : " + str(total_sum) + "\n")
    opfile.write("Accuracy Percentage : " + str(correct_sum / float(total_sum) * 100) + "\n")
    opfile.write("Error Percentage : " + str(100 - correct_sum / float(total_sum) * 100) + "\n")
    opfile.write("\n")
    return


def task6(train_data, train_labels, test_data, test_labels, k, opfile):
    predicted_labels = knn_classify(train_data, train_labels, test_data, k)
    knn_accuracy(test_data, test_labels, predicted_labels, opfile)
    return


# /Users/yash/Downloads/data/mnist_train.csv
# dummy.txt
def main():
    warnings.filterwarnings("ignore")
    plots_path = "./output_plots"
    data_path = "./output_data"
    opfile_path = "./output_data/output_problem2.txt"
    opfile_path_gram_schmidt = "./output_data/output_problem2_gram_schmidt.txt"
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

    if str(sys.argv[1]) == "-type=gram-schimdt":
        # TASK 3
        # GRAM_SCHMIDT_ALGORITHM
        ipfilepath = sys.argv[2]
        vectors, count, size = parse_input_matrix_form(ipfilepath)
        opfile_gram_schmidt = open(opfile_path_gram_schmidt, "w")
        reduced_set, count_reduced_set, contains_zero_vector = clean_input_gram_schmidt(vectors, count, size)
        ortho_vectors = gram_schmidt(reduced_set, count_reduced_set, size, contains_zero_vector)
        if contains_zero_vector:
            print_matrix(ortho_vectors, 0, count_reduced_set, 0, size - 1, opfile_gram_schmidt)
        else:
            print_matrix(ortho_vectors, 0, count_reduced_set - 1, 0, size - 1, opfile_gram_schmidt)
        opfile_gram_schmidt.close()
    else:
        training_data_path = "./input_data/mnist_train.csv"
        data, labels, count, dimension = parse_input(training_data_path)
        # reduced_model_path = "./input_data/reduced_model.txt"
        reduced_eigen_vectors_path = "./input_data/eigenvectors.txt"
        # reduced_model_labels = "./input_data/reduced_model_labels.txt"

        testing_data_path = str(sys.argv[1])
        test_data, test_labels, test_count, test_dimension = parse_input(testing_data_path)

        # TASK 1
        # DONE THIS OFFLINE
        # Find the mean vector and covariance of the data
        # mean, cov = task1(data, count, dimension)

        # TASK 2
        # DONE THIS OFFLINE
        # Find the eigen-values and vectors of the covariance matrix
        # eigen_values, eigen_vectors, real_ct, complex_ct = eigen_values_vectors(cov)

        # TASK 2
        # DONE THIS OFFLINE
        # Find the number of distinct eigen-values
        # task2(eigen_values, eigen_vectors, count_eigen_vectors, size_eigen_vectors)

        # TASK 4
        # DONE THIS OFFLINE
        # Dimensionality Reduction
        # eigen_values, count_eigen_values, size_eigen_values = parse_input_matrix_form("./input_data/eigenvalues.txt")
        # eigen_vectors, count_eigen_vectors, size_eigen_vectors = parse_input_matrix_form("./input_data/eigenvectors.txt")
        # mean, count_mean, size_mean = parse_input_matrix_form("./input_data/mean.txt")
        # cov, count_cov, size_cov = parse_input_matrix_form("./input_data/covariance.txt")
        # reduced_set = task4(eigen_vectors, count, dimension, data, mean, 150)

        # TASK 5
        # DONE THIS OFFLINE
        # Reconstruction Error
        # task5(data, count, dimension, eigen_vectors, mean)

        # TASK 6
        # DONE THIS OFFLINE
        # Parameter Fixing
        # M = [50, 150, 250, 350, 450, 550, 650, 750]
        # for i in range(0, len(M)):
        #     # evecs = parse_input_matrix_form(reduced_eigen_vectors_path)
        # data_temp, _, evecs, _ = task4(eigen_vectors, count, dimension, data, mean, M[i])
        #     bonus2(numpy.dot(data, evecs), labels, numpy.dot(test_data, evecs), test_labels, M[i])

        # exit(0)
        # TASK 6
        # ONLINE TASK!
        # Classification

        opfile = open(opfile_path, "w")
        #### USING K = 3 and M = 250
        reduced_eigen_vectors = parse_input_matrix_form(reduced_eigen_vectors_path)
        task6(numpy.dot(data, reduced_eigen_vectors[0]), labels, numpy.dot(test_data, reduced_eigen_vectors[0]),
              test_labels, 3, opfile)
        opfile.close()
        # exit(0)

        # BONUS 1
        # DONE THIS OFFLINE
        # Projection of Data Vectors on 2D space
        # X = [32, 432, 4571, 1, 9999]
        # for i in range(0, len(X)):
        #     bonus1(data[X[i]])

        # BONUS 2
        # DONE THIS OFFLINE
        # Using SKLearn
        # M = [50, 150, 250, 350, 450, 550, 650, 750]
        # for i in range(0, len(M)):
        # evecs = parse_input_matrix_form(reduced_eigen_vectors_path)
        # data_temp, _, evecs, _ = task4(eigen_vectors, count, dimension, data, mean, M[i])
        # bonus2(numpy.dot(data, evecs), labels, numpy.dot(test_data, evecs), test_labels, M[i])

        return 0


if __name__ == "__main__": main()
