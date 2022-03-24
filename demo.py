import numpy as np
import pandas as pd
import json
from simplex import Linear_Model


"""
The algorithm must receive in input the following data:

- A: matrix of the coefficient
- b: vector of base coefficients
- c: vector of the costs/profit to minimize/maximize
- "Min" or "Max": in order to specify if we have a minimization or maximization problem
- signs: vector of signes of the (in)equalities (1 means ?; -1 means ?; = means =)



TEST the 2 PHASES SIMPLEX ALGORITM:

There are 11 different examples that can be choose in order to verify the proper operation of the code.
The input data are imported from the test_data.json file.

"""


def demo_examples(n_example):

    def parse_json(file_name, n_example):
        with open(file_name, "r") as read_file:
            decodedArray = json.load(read_file)
            A = np.asarray(decodedArray["A" + n_example])
            b = np.asarray(decodedArray["b" + n_example])
            c = np.asarray(decodedArray["c" + n_example])
            min_max = decodedArray["Min_or_Max" + n_example]
            constr_signs = np.asarray(decodedArray["signs" + n_example])
        return A, b, c, min_max, constr_signs

    A, b, c, min_or_max, constr_signs = parse_json(
        "test_data.json", n_example)
    model = Linear_Model(A, b, c, min_or_max, constr_signs, verbose=True)
    model.optimization()


"""
DIET PROBLEM:

- I estimated 14 meals a week including only lunches and dinners.
- Every meal is composed by first and second courses, so the total number of courses in a week is 28
- In the dataset there are 40 different foods with their nutritional values

There are 2 possibilities:
1) Minimize the total cost of the food:
- Below are listed the constraints to satisfy the nutritional needs of 7 lunches and 7 dinners:

    * Min/Max kcal: 8000-11500
    * Min/Max grams of Fats: 260-370
    * Min/Max grams of Carbohydrates: 1100-1810
    * Min/Max grams of Proteins: 480-660
    * Each food type can be chosen up to 2 times in a week in order to varying the meals

2) Maximize the proteins for a high-protein diet:
- Below are listed the constraints to satisfy the nutritional needs of 7 lunches and 7 dinners:

    * Min/Max kcal: 8000-11500
    * Min/Max grams of Fats: 260-370
    * Min/Max grams of Carbohydrates: 1100-1810
    * Each food type can be chosen up to 2 times in a week in order to varying the meals
    * Total cost have to be less than 45 euros
    * Total number of courses equal to 28


Put objective equal to 1 to minimize cost, equal to 2 to maximize proteins
"""


def demo_diet(objective):

    def minimize_costs(min_cal, min_fat, min_car, min_pr, max_cal, max_fat, max_car, max_pr, max_times):
        df = pd.read_csv("my_diet_data.csv", nrows=25)
        num_meals = len(df)
        food_list = df["Food"].to_list()
        c = df["Price/Serving"].to_numpy()
        x = np.transpose(
            df.filter(items=["Calories", "Fat", "Carbohydrates", "Protein"]).to_numpy())
        A = np.vstack((x, x,  np.identity(num_meals)))
        b = np.hstack(([min_cal, min_fat, min_car, min_pr, max_cal, max_fat,
                        max_car, max_pr], [max_times]*num_meals))
        constr_signs = np.hstack(
            ([1]*4, [-1]*4, [-1]*num_meals))
        return A, b, c, constr_signs, food_list

    def maximize_proteins(min_cal, min_fat, min_car, max_cal, max_fat, max_car, max_cost, max_times, total_courses):
        df = pd.read_csv("my_diet_data.csv")
        num_meals = len(df)
        food_list = df["Food"].to_list()
        c = df["Protein"].to_numpy()
        y = np.transpose(df["Price/Serving"].to_numpy())
        x = np.transpose(
            df.filter(items=["Calories", "Fat", "Carbohydrates"]).to_numpy())
        A = np.vstack((x, x, y, np.identity(num_meals), [1]*num_meals))
        b = np.hstack(([min_cal, min_fat, min_car, max_cal, max_fat,
                        max_car, max_cost], [max_times]*num_meals, [total_courses]))
        constr_signs = np.hstack(
            ([1]*3, [-1]*3, [-1], [-1]*num_meals, [0]))
        return A, b, c, constr_signs, food_list

    if objective == 1:
        A, b, c, constr_signs, food_list = minimize_costs(min_cal=8000, min_fat=260, min_car=1100, min_pr=510,
                                                          max_cal=11500, max_fat=350, max_car=1810, max_pr=660, max_times=2)
        model = Linear_Model(A, b, c, "MIN", constr_signs, verbose=False)
        model.optimization()

        # Printing the results
        if model.get_feasible() and model.get_bounded():
            print(
                f"The minimum cost to buy a weeks worth of food while meeting all nutritional requirements and sufficiently varying the meals is {model.get_optimal()}€\n")
            print("List of optimal number of each meal that should be purchased:\n")
            for i in range(len(model.get_coefficients().keys())):
                if model.get_coefficients()['x_'+str(i+1)] != 0:
                    print(
                        f"{food_list[i]}: {model.get_coefficients()['x_'+str(i+1)]}")

    elif objective == 2:
        A, b, c, constr_signs, food_list = maximize_proteins(min_cal=8000, min_fat=260, min_car=1100, max_cal=11500,
                                                             max_fat=370, max_car=1810, max_cost=45, max_times=2, total_courses=28)
        model = Linear_Model(A, b, c, "MAX", constr_signs, verbose=False)
        model.optimization()

        # Printing the results
        if model.get_feasible() and model.get_bounded():
            print(
                f"The maximum quantity of proteins that can be archived considering all the cost and nutritional constraints is {model.get_optimal()}g\n")
            print("List of optimal number of each meal that should be purchased:\n")
            for i in range(len(model.get_coefficients().keys())):
                if model.get_coefficients()['x_'+str(i+1)] != 0:
                    print(
                        f"{food_list[i]}: {model.get_coefficients()['x_'+str(i+1)]}")
    else:
        print("Invalid number! Insert 1 or 2 to show the demo of the correspondent diet example")


# Select the example inserting a number between 1 and 11
# demo_examples(n_example="11")


# Put objective equal to 1 to minimize cost, equal to 2 to maximize proteins
demo_diet(objective=1)
