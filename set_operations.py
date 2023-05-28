set1_list = input("Enter elements for Set 1 (space-separated): ").split()
set1 = set(set1_list)

set2_list = input("Enter elements for Set 2 (space-separated): ").split()
set2 = set(set2_list)

union = set1.union(set2)
intersection = set1.intersection(set2)
difference = set1.difference(set2)
symmetric_difference = set1.symmetric_difference(set2)

# Print the results
print("Union of Set 1 and Set 2 is", union)
print("Intersection of Set 1 and Set 2 is", intersection)
print("Difference of Set 1 and Set 2 is", difference)
print("Symmetric difference of Set 1 and Set 2 is", symmetric_difference)