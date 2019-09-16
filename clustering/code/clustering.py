from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import numpy as np

class_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
class_num_to_name = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
data = []

with open("iris.csv", "r") as f:
	reader = list(csv.reader(f))
	attributes_row = list(reader[0])
	for i in range(1,len(reader)):
		row = list(reader[i])
		class_num = class_map[row[-1]]
		row = list(map(float, row[:-1]))
		row.append(class_num)
		data.append(row)

max_val = []
min_val = []

normalized_data = []

for feature in range(4):
	feature_arr = [row[feature] for row in data]
	max_val.append(max(feature_arr))
	min_val.append(min(feature_arr))

for row in data:
	normalized_row = []
	for feature in range(4):
		normalized_row.append((row[feature] - min_val[feature]) / (max_val[feature] - min_val[feature]))
	normalized_row.append(row[4])
	normalized_data.append(normalized_row)

def getCentroids(data):
	'''returns: centroids[[c1f1,c1f2,c1f3,c1f4],[c2f1,c2f2,c2f3,c2f4],[c3f1,c3f2,c3f3,c3f4]]
	   (where cifj = average value of feature j in class i)'''
	sums = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	counts = [0,0,0]
	for row in data:
		class_num = row[4]
		for feature in range(4):
			sums[class_num][feature] += row[feature]
		counts[class_num] += 1
	centroids = []
	for class_num in range(3):
		centroid = []
		for i in range(4):
			centroid.append(sums[class_num][i]/counts[class_num])
		centroids.append(centroid)
	return centroids

def dist(X,Y):
	total = 0
	for i in range(4):
		total += (X[i] - Y[i])**2
	return total**0.5

def getCluster(sample, centroids):
	'''returns: 0,1,2 (cluster number of sample, given the three centroids)'''
	min_dist = float("inf")
	min_ind = -1
	for i in range(3):
		distance = dist(sample,centroids[i])
		if distance < min_dist:
			min_ind = i
			min_dist = distance
	return min_ind

while True:
	centroids = getCentroids(normalized_data)
	num_changes = 0
	for row in normalized_data:
		cluster = getCluster(row, centroids)
		if cluster != row[4]: # new cluster != old cluster
			num_changes += 1
			row[4] = cluster	
	if num_changes == 0:
		break

count = [[0,0,0],[0,0,0],[0,0,0]]

for i in range(len(normalized_data)):
	count[normalized_data[i][4]][data[i][4]] += 1

for cluster_num in range(len(count)):
	out = "Cluster " + str(cluster_num+1) + ": "
	class_stats = count[cluster_num]
	for class_num in range(len(class_stats)):
		cluster_class_count = count[cluster_num][class_num]
		if cluster_class_count > 0:
			out += class_num_to_name[class_num] + " - " + str(cluster_class_count) + ", "
	out = out[:-2]
	print(out)

class Cluster():
	def __init__(self, name):
		self.name = 0
		self.f1 = []
		self.f2 = []
		self.f3 = []
		self.f4 = []

clusters = [Cluster("Iris-setosa"), Cluster("Iris-versicolor"), Cluster("Iris-virginica")]

for i in range(len(normalized_data)):
	cluster_num = normalized_data[i][4]
	clusters[cluster_num].f1.append(data[i][0])
	clusters[cluster_num].f2.append(data[i][1])
	clusters[cluster_num].f3.append(data[i][2])
	clusters[cluster_num].f4.append(data[i][3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("sepal_length", {'size':18})
ax.set_ylabel("sepal_width", {'size':18})
ax.set_zlabel("petal_length", {'size':18})

cs = clusters[0].f4 + clusters[1].f4 + clusters[2].f4
minc = min(cs)
maxc = max(cs)

label1 = "Cluster 1: " + str(len(clusters[0].f1)) + " samples"
label2 = "Cluster 2: " + str(len(clusters[1].f1)) + " samples"
label3 = "Cluster 3: " + str(len(clusters[2].f1)) + " samples"

ax.scatter(clusters[0].f1, clusters[0].f2, clusters[0].f3, c=clusters[0].f4, cmap=plt.hot(), marker="v", label=label1, vmin=minc, vmax=maxc, s=100)
ax.scatter(clusters[1].f1, clusters[1].f2, clusters[1].f3, c=clusters[1].f4, cmap=plt.hot(), marker="h", label=label2, vmin=minc, vmax=maxc, s=100)
p = ax.scatter(clusters[2].f1, clusters[2].f2, clusters[2].f3, c=clusters[2].f4, cmap=plt.hot(), marker="X", label=label3, vmin=minc, vmax=maxc, s=100)
cb = fig.colorbar(p, orientation='vertical', shrink=0.5)
cb.set_label(label="petal_width", size=18)
plt.legend(prop={'size': 20})
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('green')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('green')
plt.show()
