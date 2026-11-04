
foods = [
    {"name": "Milk", "category": "Dairy", "ingredients": 1},
    {"name": "Bread", "category": "Bakery", "ingredients": 5},
    {"name": "Cola", "category": "Drinks", "ingredients": 6},
    {"name": "Cheese", "category": "Dairy", "ingredients": 4},
    {"name": "Juice", "category": "Drinks", "ingredients": 3}
]

print("FOOD DATA ANALYSIS")
print("------------------")

print("Total products:", len(foods))

for food in foods:
    print(food["name"], "-", food["ingredients"], "ingredients")

print("\nProducts with more than 3 ingredients:")

for food in foods:
    if food["ingredients"] > 3:
        print(food["name"])

total = sum(food["ingredients"] for food in foods)
average = total / len(foods)

print("\nAverage ingredient count:", average)
