import sys
import numpy
import time
import random


# Yash Khanna
# MTech (Research) - CSA
# 15563


# Find number of zeroes in list from [st, end]
def is_list_zero(list, st, end):
    ans = True
    for i in range(st, end + 1):
        if list[i] != 0:
            ans = False
    return ans


# Print Matrix in row-major form using in the given indices
def print_matrix(matrix, row_start, row_end, col_start, col_end, opfile):
    for i in range(row_start, row_end + 1):
        for j in range(col_start, col_end + 1):
            opfile.write(str(round(matrix[i][j], 3)) + " ")
            print round(matrix[i][j], 3),
        opfile.write("\n")
        print ""


# Print steps
def print_steps(steps, opfile):
    for i in range(len(steps)):
        print steps[i]
        opfile.write(steps[i] + "\n")


# Given a matrix, find its inverse and if it does not exist, terminate with a RREF form
def invert_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    col_range = cols / 2 - 1
    last_non_zero_row = -1
    last_pivot_col = -1
    pivot_cols = []
    steps = []

    # Check the last non_zero_row
    lo = 0
    hi = rows - 1
    while lo < hi:
        while not is_list_zero(matrix[lo], 0, col_range) and lo < hi:
            lo += 1

        while is_list_zero(matrix[hi], 0, col_range) and lo < hi:
            hi -= 1

        if lo < hi:
            matrix[lo], matrix[hi] = matrix[hi], matrix[lo]
            # print "SWITCH " + str(lo + 1) + " " + str(hi + 1)
            steps.append("SWITCH " + str(lo + 1) + " " + str(hi + 1))
            lo += 1
            hi -= 1

    for i in reversed(range(rows)):
        if not is_list_zero(matrix[i], 0, col_range):
            last_non_zero_row = i
            break

    for i in range(rows):
        if is_list_zero(matrix[i], 0, col_range):
            if last_non_zero_row > i:
                k = -1
                diff = 0
                for j in range(last_non_zero_row, i, -1):
                    if not is_list_zero(matrix[j], 0, col_range):
                        k = j
                        break
                    else:
                        diff += 1

                last_non_zero_row -= diff
                if k == -1:
                    break
                else:
                    matrix[k], matrix[i] = matrix[i], matrix[k]
                    # print "SWITCH " + str(i + 1) + " " + str(k + 1)
                    steps.append("SWITCH " + str(i + 1) + " " + str(k + 1))
            else:
                last_non_zero_row -= 1
                break

        pivot_row = 0
        pivot_col = 0
        pivot_val = 0
        break_loop = False

        for j in range(last_pivot_col + 1, cols):
            for k in range(i, last_non_zero_row + 1):
                if not break_loop and matrix[k][j] != 0:
                    pivot_row = k
                    pivot_col = j
                    pivot_val = matrix[k][j]
                    break_loop = True

        pivot_cols.append(pivot_col)
        if i != pivot_row:
            matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
            # print "SWITCH " + str(i + 1) + " " + str(pivot_row + 1)
            steps.append("SWITCH " + str(i + 1) + " " + str(pivot_row + 1))

        if pivot_val != 1:
            # print "MULTIPLY " + str(round(1 / (pivot_val * 1.0), 3)) + " " + str(i + 1)
            steps.append("MULTIPLY " + str(round(1 / (pivot_val * 1.0), 3)) + " " + str(i + 1))
            for j in range(cols):
                matrix[i][j] /= (pivot_val * 1.0)

        for j in range(rows):
            if j == i:
                continue
            foo = matrix[j][pivot_col]
            if foo != 0:
                # print "MULTIPLY&ADD " + str(-foo * 1.0) + " " + str(i + 1) + " " + str(j + 1)
                steps.append("MULTIPLY&ADD " + str(round(-foo * 1.0, 3)) + " " + str(i + 1) + " " + str(j + 1))
                for k in range(cols):
                    matrix[j][k] = matrix[j][k] - foo * matrix[i][k]

        # print_matrix(matrix, 0, rows - 1, 0, 2 * rows - 1)

    # Rounding Error + Output Formatting
    for i in range(rows):
        for j in range(cols):
            matrix[i][j] = round(matrix[i][j], 3)
            if matrix[i][j] == -0.0:
                matrix[i][j] = 0.0

    return matrix, pivot_cols, steps


# Augment our square matrix with Identity Matrix (of size n x n)
def augment_with_identity(matrix, n):
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                matrix[i].append(float(1))
            else:
                matrix[i].append(float(0))
    return matrix


# Fetch Input for problem 2, as given in the question
def parse_input(filepath):
    file = open(filepath, "r")
    n = int(file.readline())
    matrix = []
    for i in range(0, n):
        matrix.append(map(float, file.readline().split()))

    file.close()

    augmented_matrix = [row[:] for row in matrix]
    augmented_matrix = augment_with_identity(augmented_matrix, n)

    return matrix, augmented_matrix, n


# Extract the Inverse from matrix (2nd half from the augmented matrix)
def inverse_from_rref(matrix, n):
    inverse = [[0 for x in range(n)] for y in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            inverse[i][j] = matrix[i][j + n]
    return inverse


# Solve Problem 2 and report the answer in correct format
def solve(matrix, n, opfile):
    start_time = time.clock()
    rref_matrix, pivot_cols, steps = invert_matrix(matrix)
    end_time = time.clock()
    # print pivot_cols

    if len(pivot_cols) != n:
        opfile.write("ALAS! DIDN'T FIND ONE!\n")
        print "ALAS! DIDN'T FIND ONE!"
        # print_matrix(rref_matrix, 0, n - 1, 0, 2 * n - 1)
        print_steps(steps, opfile)
    else:
        opfile.write("YAAY! FOUND ONE!\n")
        print "YAAY! FOUND ONE!"
        # print_matrix(rref_matrix, 0, n - 1, 0, 2 * n - 1)
        print_matrix(rref_matrix, 0, n - 1, n, 2 * n - 1, opfile)
        print_steps(steps, opfile)
        print_steps(steps, opfile)
    return rref_matrix, end_time - start_time


# Solve Problem 2 using numpy (ONLY FOR PERFORMANCE COMPARISON)
def solve_numpy(matrix, n):
    start_time = time.clock()
    end_time = 0
    try:
        inverse = numpy.linalg.inv(matrix)
        end_time = time.clock()
        # print_matrix(inverse, 0, n - 1, 0, n - 1)
    except numpy.linalg.LinAlgError:
        # Not invertible. Skip this one.
        print "Error: Not Invertible"
    return inverse, end_time - start_time


# Convert numpy matrix to 2D array (ONLY FOR PERFORMANCE COMPARISON)
def numpy_to_array(matrix, n):
    inverse = [[0 for x in range(n)] for y in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            inverse[i][j] = matrix[i][j]
    return inverse


# Compare 2 matrices of same size and report the number of hits/misses (ONLY FOR PERFORMANCE COMPARISON)
def compare(mat1, mat2, n):
    hits = 0
    misses = 0
    for i in range(0, n):
        for j in range(0, n):
            if abs(mat1[i][j] - mat2[i][j]) < 0.001:
                hits = hits + 1
            else:
                misses = misses + 1
    print hits
    print misses
    return misses == 0


# Generate a random matrix and compare our algorithm with Numpy inverse (ONLY FOR PERFORMANCE COMPARISON)
def compare_with_numpy(opfile):
    sz = 10
    iterations = 50
    time1_sum = 0
    time2_sum = 0
    for k in range(0, iterations, 1):
        mat2 = [[0 for x in range(sz)] for y in range(sz)]
        for i in range(0, sz):
            for j in range(0, sz):
                mat2[i][j] = random.uniform(1, sz + 1)

        mat3 = [row[:] for row in mat2]
        mat3 = augment_with_identity(mat3, sz)
        inverse2, time2 = solve_numpy(mat2, sz)
        inverse1, time1 = solve(mat3, sz, opfile)
        time1_sum += time1
        time2_sum += time2
        inverse1 = inverse_from_rref(inverse1, sz)
        inverse2 = numpy_to_array(inverse2, sz)
        compare(inverse1, inverse2, sz)
    print time1_sum / iterations
    print time2_sum / iterations
    print time1_sum / iterations - time2_sum / iterations


# Main function
def main():
    # Command Line Arguments
    ipfilepath = sys.argv[1]
    matrix, augmented_matrix, n = parse_input(ipfilepath)
    opfile = open("./output_problem2.txt", "w")
    solve(augmented_matrix, n, opfile)
    # compare_with_numpy(opfile) (ONLY FOR PERFORMANCE COMPARISON)
    opfile.close()


if __name__ == "__main__": main()
