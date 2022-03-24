# Two Phase Simplex Method 
Implementation of the Two Phase Simplex Algorithm with pratical application to a diet problem.

## Diet Problem 🍕🍔

The goal of the diet problem is to select a set of foods that will satisfy a set of nutritional requirement at minimum cost. The problem is formulated as a linear program where the objective is to minimize cost and the constraints are to satisfy the specified nutritional requirements. 
The diet problem constraints regulate the number of calories and the amount of carbohydrates, fats, proteins in the diet.

- I estimated 14 meals a week including only lunches and dinners.
- Every meal is composed by first and second courses, so the total number of courses in a week is 28
- In the dataset there are 40 different foods with their nutritional values

There are 2 possibilities:
1) Minimize the total cost of the food:
- Below are listed the constraints to satisfy the nutritional needs of 7 lunches and 7 dinners:

    * Min/Max kcal: 8000-11500 grams
    * Min/Max grams of Fats: 260-370 grams
    * Min/Max grams of Carbohydrates: 1100-1810 grams
    * Min/Max grams of Proteins: 480-660 grams
    * Each food type can be chosen up to 2 times in a week in order to varying the meals

2) Maximize the proteins for a high-protein diet:
- Below are listed the constraints to satisfy the nutritional needs of 7 lunches and 7 dinners:

    * Min/Max kcal: 8000-11500 grams
    * Min/Max grams of Fats: 260-370 grams
    * Min/Max grams of Carbohydrates: 1100-1810 grams
    * Each food type can be chosen up to 2 times in a week in order to varying the meals
    * Total cost have to be less than 45 euros
    * Total number of courses equal to 28

## Repository Files
* **Simplex.py**: Implementation of the algorithm
* **Demo.py**: Demo to test algorithm
* **test_data.json**: File containing the data of some example to test the algorithm
* **my_diet_data.csv**: File containg the data of the foods and their nutritional requirements


## Libraries
* Numpy
* Pandas
