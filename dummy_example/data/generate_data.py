# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

num_samples = 1000
x_range = (0, 10)
threshold = np.mean(x_range)

X = list()
for i in range(num_samples):
    x_min, x_max = x_range
    X.append([np.random.uniform(x_min, x_max), np.random.uniform(x_min, x_max)])

x1 = [i for i,j in X]
x2 = [j for i,j in X]

Y1 = list()
Y2 = list()

for i,j in X:
    if j > threshold:
        Y1.append(0)
    else:
        Y1.append(1)
    if i > j:
        Y2.append(0)
    else:
        Y2.append(1)

class_0_y1 = list()
class_1_y1 = list()

class_0_y2 = list()
class_1_y2 = list()

for i in range(len(Y1)):
    if Y1[i] == 0:
        class_0_y1.append(X[i])
    else:
        class_1_y1.append(X[i])
    if Y2[i] == 0:
        class_0_y2.append(X[i])
    else:
        class_1_y2.append(X[i])
        
x1_0_y1 = [i for i,j in class_0_y1]
x2_0_y1 = [j for i,j in class_0_y1]
x1_1_y1 = [i for i,j in class_1_y1]
x2_1_y1 = [j for i,j in class_1_y1]

x1_0_y2 = [i for i,j in class_0_y2]
x2_0_y2 = [j for i,j in class_0_y2]
x1_1_y2 = [i for i,j in class_1_y2]
x2_1_y2 = [j for i,j in class_1_y2]

plt.scatter(x1_0_y1, x2_0_y1, c='g')
plt.scatter(x1_1_y1, x2_1_y1, c='b')
plt.show()

plt.scatter(x1_0_y2, x2_0_y2, c='g')
plt.scatter(x1_1_y2, x2_1_y2, c='b')
plt.show()

f = open('data1', 'w')
for i in range(len(x1_0_y1)):
    f.write(str(x1_0_y1[i]) + '\t' + str(x2_0_y1[i]) + '\t' + str(0) + '\n')
for i in range(len(x1_1_y1)):
    f.write(str(x1_1_y1[i]) + '\t' + str(x2_1_y1[i]) + '\t' + str(1) + '\n')
f.close()

f = open('data2', 'w')
for i in range(len(x1_0_y2)):
    f.write(str(x1_0_y2[i]) + '\t' + str(x2_0_y2[i]) + '\t' + str(0) + '\n')
for i in range(len(x1_1_y2)):
    f.write(str(x1_1_y2[i]) + '\t' + str(x2_1_y2[i]) + '\t' + str(1) + '\n')
f.close()
    