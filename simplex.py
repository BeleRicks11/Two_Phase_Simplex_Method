import numpy as np
import pandas as pd

"""
- A matrix of the coefficient of the original problem
- b vector of coefficient (base) of the original problem
- c vector of the costs/profit to minimize/maximize

   Ind    z    x1   x2   x3   x4
          0    c    c    c    c
    i     b    A    A    A    A
    i     b    A    A    A    A
    i     b    A    A    A    A
    i     b    A    A    A    A
"""


class Linear_Model():
    def __init__(self, A, b, c, min_or_max, constr_signs, rule, verbose):
        self.A = A
        self.b = b
        self.c = c
        self.min_or_max = min_or_max
        self.optimal_value = None
        self.coefficients = {}
        self.feasible = True
        self.bounded = False
        self.constr_signs = constr_signs
        self.num_slacks_var = 0
        self.num_artificial = 0
        self.rule = rule
        self.verbose = verbose

    def set_A(self, A):
        self.A = A

    def set_B(self, b):
        self.b = b

    def set_C(self, c):
        self.c = c

    def get_coefficients(self):
        return self.coefficients

    def get_optimal(self):
        return round(self.optimal_value, 2)

    def set_objective(self, min_or_max):
        if(min_or_max == "MIN" or min_or_max == "MAX"):
            self.min_or_max = min_or_max
        else:
            print("Invalid objective.")

    def printTableau(self, tableau):
        print("ind \t\t", end="")
        for i in range(2, 2 + len(self.c)):
            print("x_" + str(i-1), end="\t")
        for i in range(2 + len(self.c), 2 + len(self.c) + self.num_slacks_var):
            print("s_" + str(i-1), end="\t")
        for i in range(2 + len(self.c) + self.num_slacks_var, len(tableau[0])):
            print("a_" + str(i-1), end="\t")
        print()
        for j in range(0, len(tableau)):
            for i in range(0, len(tableau[0])):
                if(not np.isnan(tableau[j, i])):
                    if(i == 0):
                        print(int(tableau[j, i]-1), end="\t")
                    else:
                        print(round(tableau[j, i], 2), end="\t")
                else:
                    print(end="\t")
            print()

    def get_Tableau_Phase1(self):

        def add_variables(self, system, type):
            for i in range(len(self.constr_signs)):
                var_col = np.zeros(len(self.b))
                # For ≥ constraint, we add a surplus and an artificial variable
                if self.constr_signs[i] == 1:
                    if type == 'standard':
                        var_col[i] = -1
                    else:
                        var_col[i] = 1
                    system = np.hstack((system, np.transpose([var_col])))

                # For ≤ constraint, we add a slack variable
                elif self.constr_signs[i] == -1:
                    if type == 'standard':
                        var_col[i] = 1
                        system = np.hstack((system, np.transpose([var_col])))

                # For = constraint, we add an artificial variable
                elif self.constr_signs[i] == 0:
                    if type == 'auxiliary':
                        var_col[i] = 1
                        system = np.hstack((system, np.transpose([var_col])))
            return system

        # Costruct starting Tableau
        system = np.hstack((np.transpose([self.b]), self.A))

        # Check if there are negative coefficients (if yes multiply row by -1 and change the sign of the inequality)
        for i in range(len(self.b)):
            if self.b[i] < 0:
                system[i, :] = -system[i, :]
                self.constr_signs[i] = -self.constr_signs[i]

        # Passing to standard form
        system = add_variables(self, system, type="standard")

        # Passing to auxiliary problem
        system = add_variables(self, system, type="auxiliary")

        # Builiding the tableau
        self.num_artificial = len(self.constr_signs[self.constr_signs >= 0])
        self.num_slacks_var = len(
            system[0]) - len(self.A[0]) - self.num_artificial - 1

        cost_row = np.hstack(
            ([None], [0], [0]*len(self.A[0]), [0] * self.num_slacks_var, [1]*self.num_artificial))

        # Initializing the indexes of variables that are in the base
        basis = np.array([0]*len(system))
        slack_index = 2 + len(self.c)
        artif_index = 2 + len(self.c) + self.num_slacks_var

        for i in range(0, len(basis)):
            if self.constr_signs[i] == -1:
                basis[i] = slack_index
                slack_index += 1
            else:
                basis[i] = artif_index
                artif_index += 1

        block = np.hstack(
            (np.transpose([basis]), system))

        tableau = np.vstack((cost_row, block))

        # The coefficients for all the basic variables in the objective must be zero (AV have a coefficient equal to 1)
        # We subtract to row 0 the row that correspond to the basic variable that have a non zero value in the objective
        for i in range(1, len(tableau)):
            for j in range(len(tableau[0]) - self.num_artificial, len(tableau[0])):
                if tableau[i, j] == 1:
                    tableau[0, 1:] = tableau[0, 1:] - tableau[i, 1:]

        tableau = np.array(tableau, dtype='float')
        return tableau

    def pivoting_operation(self, tableau, row_index, col_index):
        # The element at index (r,j) will be the pivot element and row r will be the pivot row.

        # Divide the r-th row by pivot to make it 1. And subtract c*(rth row) from other
        # rows to make them 0, where c is the coefficient required to make that row 0.

        pivot = tableau[row_index, col_index]

        # Divide the pivoting row by the pivot element
        tableau[row_index, 1:] = tableau[row_index, 1:] / pivot

        # Pivot the other rows
        for row in range(0, len(tableau)):
            if row != row_index:
                c = tableau[row, col_index] / tableau[row_index, col_index]
                tableau[row, 1:] = tableau[row, 1:] - \
                    c * tableau[row_index, 1:]

        tableau = np.around(tableau, 14)
        return tableau

    def simplex(self, tableau):
        # Number of iterations
        iter = 0

        while True:
            # assume initial basis is optimal, problem is feasible
            self.optimal = True
            self.feasible = True
            self.bounded = True

            # check for optimality
            for cost in tableau[0, 2:]:
                if cost < 0:
                    self.optimal = False
                    self.bounded = False
                    break

            if self.optimal == True:
                break

            # Blands anticycle rule
            if self.rule == "blands":
                for i in range(2, len(tableau[0])):
                    if tableau[0, i] < 0:
                        col_index = i
                        break
            else:
                # Standard rule
                # Find the column corresponding to min relative cost
                # Let column j have the min relative cost: xj will enter the basis
                col_index = np.argmin(tableau[0, 2:]) + 2

            # Check if the problem is bounded (if in the pivoting column there are elements > 0)
            row_index = -1
            for i in range(1, len(tableau)):
                if tableau[i, col_index] > 0:
                    row_index = i
                    self.bounded = True
                    break

            if (not self.bounded):
                print("Unbounded; No solution.")
                return

            # Min ratio test
            min_ratio = 99999
            for row in range(row_index, len(tableau)):
                if tableau[row, col_index] > 0:
                    ratio = tableau[row, 1] / tableau[row, col_index]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        row_index = row
                    elif ratio == min_ratio and self.rule == "blands":
                        if tableau[row, 0] < tableau[row_index, 0]:
                            row_index = row

            pivot = tableau[row_index, col_index]

            if self.verbose:
                print("Pivot Row:", row_index)
                print("Pivot Column:", col_index)
                print("Pivot Element: ", pivot)
                print("Row of the new Basis: " +
                      str(int(tableau[row_index, 0])-1))

            # Applying the pivoting
            tableau = self.pivoting_operation(tableau, row_index, col_index)

            # new basic variable
            tableau[row_index, 0] = col_index

            if self.verbose:
                self.printTableau(tableau)
                print("-"*70)

            iter += 1

        if self.verbose:
            print("Final Tableau reached in", iter, "iterations:\n")
            self.printTableau(tableau)
            print("-"*70)
        return tableau

    def remove_artificial_variables(self, tableau):
        # Check if all artificial variables are out of base
        # If yes we can simply remove the artificial columns and substitute the objective function
        # If no we have to remove them from the base with pivoting operations

        # If there are not artificial variables return the original tableau
        if self.num_artificial == 0:
            return tableau

        # Computing the ids of the artificial variables
        artif_var_ids = []
        for i in range(len(tableau[0]) - self.num_artificial, len(tableau[0])):
            artif_var_ids.append(i)

        # Check if there are artificial variables in the basis
        for i in range(1, len(tableau)):
            if tableau[i, 0] in artif_var_ids:
                # If all values in the row of the AV are equal to zero we can simply remove that row
                # Otherwise we have to apply the pivot operation in the element of the row different from zero
                zero_row = True
                for j in range(2, len(tableau[0])):
                    if (tableau[i, j] != 0):
                        zero_row = False
                        tableau = self.pivoting_operation(tableau, i, j)
                        break
                if zero_row:
                    tableau = np.delete(tableau, i, axis=0)

        # Once removed the AV in the basis we can delete the AV columns from the tableau
        tableau = tableau[:, :artif_var_ids[0]]

        if self.verbose:
            print("\nArtificial variables removed")
            self.printTableau(tableau)
            print("-"*70)
        return tableau

    def get_Tableau_Phase2(self, tableau):

        # If we have a Maximization problem we transform it into MIN by changing the sign of the objective function
        if self.min_or_max == "MAX":
            self.c = -self.c

        # Initializing the new tableau for phase 2 with the original objective function
        tableau[0] = np.hstack(([None], [0], self.c, [0]*self.num_slacks_var))

        # Passing to the canonic form
        # I have to make sure that all the basic variables have coefficient equal to 0 in the first row
        for i in range(1, len(tableau)):
            if tableau[0, int(tableau[i, 0])] != 0:
                tableau[0, 1:] = tableau[0, 1:] - \
                    tableau[0, int(tableau[i, 0])] * tableau[i, 1:]

        if self.verbose:
            print("\nTableau ready for the Phase 2\n")
            self.printTableau(tableau)
            print("-"*70)
        return tableau

    def optimization(self):

        tableau = self.get_Tableau_Phase1()

        if self.verbose:
            print("Phase 1")
            print("Intitial tableau:")
            self.printTableau(tableau)
            print("-"*70)

        tableau = self.simplex(tableau)

        if(not self.bounded):
            return

        # Check for feasibility
        if(int(tableau[0, 1]) != 0):
            self.feasible = False
            print("Problem Infeasible; No Solution")
            return

        tableau = self.remove_artificial_variables(tableau)

        tableau = self.get_Tableau_Phase2(tableau)

        tableau = self.simplex(tableau)

        if(not self.bounded):
            return

        # Saving coefficients
        for row in range(1, (len(tableau))):
            if int(tableau[row, 0])-1 <= len(self.c):
                self.coefficients["x_" +
                                  str(int(tableau[row, 0])-1)] = np.around(tableau[row, 1], 3)

        # Adding coefficients equal to 0 in the dict
        for i in range(1, len(self.c) + 1):
            if not("x_" + str(i) in self.coefficients):
                self.coefficients["x_" + str(i)] = 0.0

        # Saving optimal value
        if self.min_or_max == "MAX":
            self.optimal_value = np.around(tableau[0, 1], 3)
        else:
            self.optimal_value = np.around(-tableau[0, 1], 3)
        self.printSoln()

    def printSoln(self):
        if(self.feasible):
            if(self.bounded):
                if self.verbose:
                    print("Coefficients: ")
                    print(self.coefficients)
                    print("Optimal value: ")
                    print(self.optimal_value)
            else:
                print("Problem Unbounded; No Solution")
        else:
            print("Problem Infeasible; No Solution")
