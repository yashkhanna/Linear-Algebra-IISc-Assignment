import sys
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
def print_matrix(matrix, row_start, row_end, col_start, col_end):
    for i in range(row_start, row_end + 1):
        for j in range(col_start, col_end + 1):
            print round(matrix[i][j], 3),
        print ""


# Given a matrix, find its RREF
def rref(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    col_range = cols - 2
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


# Fetch Input for problem 1, as given in the question
def parse_input(filepath, is_part_two):
    file = open(filepath, "r")
    if is_part_two:
        n, k = map(int, file.readline().split())
    else:
        n, k = 4, 4

    target_potion = map(int, file.readline().split())
    matrix = []
    for i in range(0, n):
        matrix.append(map(float, file.readline().split()))
    max_ingredients = map(int, file.readline().split())

    file.close()

    for i in range(0, n):
        matrix[i].append(float(target_potion[i]))

    return n, k, target_potion, matrix, max_ingredients


# In case of "More than one solution", generate a single solution satisfying the boundary conditions and the equation
def generate_single_soln(free_cols, pivot_cols, n, k, matrix, max_ingredients):
    single_sol = [0] * k
    len_free = len(free_cols)
    len_pivot = len(pivot_cols)
    iterations = 10000
    # print max_ingredients
    # print free_cols
    for iteration in range(0, iterations, 1):
        for i in range(0, len_free):
            idx = free_cols[i]
            single_sol[idx] = round(random.uniform(0, max_ingredients[idx]), 3)

        for i in range(0, n):
            pivot = -1
            val = 0
            for j in range(0, k):
                if matrix[i][j] != 0:
                    if j in pivot_cols:
                        pivot = j
                    else:
                        val += -single_sol[j] * matrix[i][j]
            if pivot != -1:
                single_sol[pivot] = matrix[i][k] + val
        ct = 0
        # print single_sol
        for i in range(0, k, 1):
            if single_sol[i] >= 0 and single_sol[i] <= max_ingredients[i]:
                ct += 1
        if ct == k:
            # print single_sol
            # print iteration
            return single_sol, iteration + 1
        # else:
        # print ct

    return [], iterations


# In case of "More than one solution", generate the generic solution
def generate_generic_soln(free_cols, pivot_cols, n, k, matrix):
    gen_sol = [""] * k

    for i in range(0, n):
        temp = ""
        pivot_var = -1
        for j in range(0, k):
            if matrix[i][j] != 0:
                if j in pivot_cols:
                    pivot_var = j
                else:
                    if temp == "":
                        if matrix[i][j] == 1:
                            temp += "X" + str(j)
                        else:
                            temp += str(matrix[i][j]) + "*X" + str(j)
                    else:
                        if matrix[i][j] == 1:
                            temp += "+" + "X" + str(j)
                        else:
                            temp += "+" + str(matrix[i][j]) + "*X" + str(j)
        if pivot_var != -1:
            if matrix[i][k] != 0:
                if temp == "":
                    gen_sol[pivot_var] = str(matrix[i][k])
                else:
                    gen_sol[pivot_var] = str(matrix[i][k]) + "-(" + temp + ")"
            else:
                if temp == "":
                    gen_sol[pivot_var] = str(matrix[i][k])
                else:
                    gen_sol[pivot_var] = "-(" + temp + ")"

    for i in range(0, k):
        if i in free_cols:
            gen_sol[i] = "X" + str(i)

    return gen_sol


# Solve Problem 1 and report the answer in correct format
def solve(input, opfile):
    (n, k, target_potion, matrix, max_ingredients) = input
    # print_matrix(matrix, 0, n - 1, 0, k)
    rref_matrix, pivot_cols, steps = rref(matrix)
    # print n
    # print k
    # print pivot_cols
    # print_matrix(rref_matrix, 0, n - 1, 0, k)
    # print steps

    inconsistent = False
    for i in range(0, n):
        if is_list_zero(matrix[i], 0, k - 1) and not is_list_zero(matrix[i], 0, k):
            inconsistent = True

    if inconsistent:
        opfile.write("NOT POSSIBLE, SNAPE IS WICKED!\n")
        print "NOT POSSIBLE, SNAPE IS WICKED!"
    else:
        free_cols = []
        for i in range(0, k, 1):
            if i not in pivot_cols:
                free_cols.append(i)
        # print free_cols

        if len(free_cols) == 0:
            for i in range(0, k):
                if matrix[i][k] < 0 or matrix[i][k] > max_ingredients[i]:
                    inconsistent = True

            if inconsistent:
                opfile.write("NOT POSSIBLE, SNAPE IS WICKED!\n")
                print "NOT POSSIBLE, SNAPE IS WICKED!"
            else:
                opfile.write("EXACTLY ONE!\n")
                print "EXACTLY ONE!"
                for i in range(0, k):
                    print round(matrix[i][k], 3),
                    opfile.write(str(round(matrix[i][k], 3)) + " ")
        else:
            single_sol, iterations = generate_single_soln(free_cols, pivot_cols, n, k, matrix, max_ingredients)
            # print iterations
            if len(single_sol) == 0:
                opfile.write("NOT POSSIBLE, SNAPE IS WICKED!\n")
                print "NOT POSSIBLE, SNAPE IS WICKED!"
            else:
                gen_sol = generate_generic_soln(free_cols, pivot_cols, n, k, matrix)
                opfile.write("MORE THAN ONE!\n")
                print "MORE THAN ONE!"
                for i in range(0, k):
                    print round(single_sol[i], 3),
                    opfile.write(str(round(single_sol[i], 3)) + " ")
                print ""
                opfile.write("\n")
                for i in range(0, k):
                    print gen_sol[i],
                    opfile.write(str(gen_sol[i]) + " ")


# Main function
def main():
    # Command Line Arguments
    subproblem = str(sys.argv[1]).split("=")[1]
    ipfilepath = sys.argv[2]
    if subproblem == "one":
        opfilename = "./output_problem1_part1.txt"
        input = parse_input(ipfilepath, False)
    elif subproblem == "two":
        opfilename = "./output_problem1_part2.txt"
        input = parse_input(ipfilepath, True)
    else:
        return
    opfile = open(opfilename, "w")
    solve(input, opfile)
    opfile.close()


if __name__ == "__main__": main()
