from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import numpy as np

class1 = []
class2 = []
class3 = []

with open("iris.csv", "r") as f:
	reader = list(csv.reader(f))
	attributes_row = list(reader[0])
	for i in range(1,len(reader)):
		row = list(reader[i])
		class_name = row[-1]
		row = list(map(float, row[:-1]))
		if class_name == "Iris-setosa":
			class1.append(row)
		elif class_name == "Iris-versicolor":
			class2.append(row)
		elif class_name == "Iris-virginica":
			class3.append(row)

x1 = [row[0] for row in class1]
y1 = [row[1] for row in class1]
z1 = [row[2] for row in class1]
c1 = [row[3] for row in class1]

x2 = [row[0] for row in class2]
y2 = [row[1] for row in class2]
z2 = [row[2] for row in class2]
c2 = [row[3] for row in class2]

x3 = [row[0] for row in class3]
y3 = [row[1] for row in class3]
z3 = [row[2] for row in class3]
c3 = [row[3] for row in class3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("sepal_length", {'size':18})
ax.set_ylabel("sepal_width", {'size':18})
ax.set_zlabel("petal_length", {'size':18})

c=c1+c2+c3
minc = min(c)
maxc = max(c)

ax.scatter(x1, y1, z1, c=c1, cmap=plt.hot(), marker="^", label="Iris-setosa", vmin=minc, vmax=maxc, s=100)
ax.scatter(x2, y2, z2, c=c2, cmap=plt.hot(), marker="*", label="Iris-versicolor", vmin=minc, vmax=maxc, s=100)
p = ax.scatter(x3, y3, z3, c=c3, cmap=plt.hot(), marker="D", label="Iris-virginica", vmin=minc, vmax=maxc, s=100)
cb = fig.colorbar(p, orientation='vertical', shrink=0.5)
cb.set_label(label="petal_width", size=18)
plt.legend(prop={'size': 18})
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('green')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('green')
plt.show()
