input_list = input("Enter a list of integers separated by spaces: ").split()

numbers = [int(i) for i in input_list]

positive_numbers = [j for j in numbers if j > 0]
print("Positive numbers:", positive_numbers)