# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
from model.model import *

data_dir = os.path.join(os.getcwd(), 'data', 'data2')

model = Model(data_dir)

x1 = [i for i,j in model.x]
x2 = [j for i,j in model.x]

x_range = (-100, 50)
num_samples = 10000
random_points = list()
for i in range(num_samples):
    x_min, x_max = x_range
    random_points.append([np.random.uniform(x_min, x_max), np.random.uniform(x_min, x_max)])

y = model.model.predict(model.x)
random_y = model.model.predict(random_points)

y = [i[0] for i in y]
random_y = [i[0] for i in random_y]

threshold = 0.5

class_0_x = list()
class_1_x = list()

for i in range(len(y)):
    if y[i] < threshold:
        class_0_x.append(model.x[i])
    else:
        class_1_x.append(model.x[i])
for i in range(len(random_y)):
    if random_y[i] < threshold:
        class_0_x.append(random_points[i])
    else:
        class_1_x.append(random_points[i])

x1_class_0 = [i for i,j in class_0_x]
x2_class_0 = [j for i,j in class_0_x]

x1_class_1 = [i for i,j in class_1_x]
x2_class_1 = [j for i,j in class_1_x]

plt.scatter(x1_class_0, x2_class_0, c = 'g')
plt.scatter(x1_class_1, x2_class_1, c = 'b')
plt.show()
