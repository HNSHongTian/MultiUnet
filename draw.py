import matplotlib.pyplot as plt
import numpy as np

filename = 'dice.txt'
X, Y = [], []
with open(filename, 'r') as f:  # 1
    lines = f.readlines()  # 2
    for line in lines:  # 3
        value = [float(s) for s in line.split()]  # 4
        X.append(value[0])  # 5
        Y.append(value[1])

print(X)
print(Y)

plt.plot(X, Y)
plt.savefig("dice.jpg")
plt.show()





