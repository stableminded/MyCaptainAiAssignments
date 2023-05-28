num_terms = int(input("Enter the number of Fibonacci terms: "))

num1 = 0
num2 = 1

print(num1, end="")

if num_terms > 1:
    
    print(", " + str(num2), end="")

    for i in range(2, num_terms):
        num3 = num1 + num2
        print(", " + str(num3), end="")
        num1 = num2
        num2=  num3
