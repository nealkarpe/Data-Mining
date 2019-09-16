from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import numpy as np

data = []
with open("iris.csv", "r") as f:
	reader = list(csv.reader(f))
	attributes_row = list(reader[0])
	for i in range(1,len(reader)):
		row = list(reader[i])
		row = list(map(float, row[:-1]))
		data.append(row)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [row[0] for row in data]
y = [row[1] for row in data]
z = [row[2] for row in data]
c = [row[3] for row in data]

ax.set_xlabel("sepal_length", {'size':18})
ax.set_ylabel("sepal_width", {'size':18})
ax.set_zlabel("petal_length", {'size':18})

p = ax.scatter(x, y, z, c=c, s=100, cmap=plt.hot())
cb = fig.colorbar(p, orientation='vertical', label="petal_width", shrink=0.5)
cb.set_label(label="petal_width", size=18)
plt.show()
