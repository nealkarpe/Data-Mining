import csv
import numpy as np
import matplotlib.pyplot as plt

class_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
class_num_to_name = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

def read_file():
	data = []
	labels = []
	with open("iris.csv", "r") as f:
		reader = list(csv.reader(f))
		attributes_row = list(reader[0])
		for i in range(1,len(reader)):
			row = list(reader[i])
			labels.append(row[-1])
			row = list(map(float, row[:-1]))
			data.append(np.array(row))
	return data, labels

def input(data, labels, test_ind):
	train_samples = data[:test_ind] + data[test_ind+1:]
	train_labels = labels[:test_ind] + labels[test_ind+1:]
	test_sample = data[test_ind]
	test_label = labels[test_ind]
	return train_samples, train_labels, test_sample, test_label

def get_classes(training_data, training_labels):
	classes = {}
	for i in range(len(training_data)):
		label = training_labels[i]
		if label not in classes:
			classes[label] = [training_data[i]]
		else:
			classes[label].append(training_data[i])
	return classes

def findMeans(classes):
	means = {}
	for label in classes:
		classpoints = np.array(classes[label])
		means[label] = sum(classpoints)/len(classpoints)
	return means

def findStds(classes):
	stds = {}
	for label in classes:
		class_std = []
		classpoints = np.array(classes[label])
		num_dimensions = classpoints.shape[1]
		for d in range(num_dimensions):
			class_std.append(np.std(classpoints[:,d]))
		stds[label] = np.array(class_std)
	return stds

def likelihood(test_sample, mean, std):
	prod = 1
	num_dimensions = len(test_sample)
	for d in range(num_dimensions):
		x = test_sample[d]
		try:
			prod *= (1.0/np.sqrt(2*np.pi*std[d]**2))*np.exp(-1.0*(x-mean[d])**2/(2*std[d]**2))
		except:
			return 0
	return prod

def predict(test_sample, prior, means, stds):
	max_posterior = -1
	out = ""
	for label in classes:
		posterior = prior[label]*likelihood(test_sample, means[label], stds[label])
		if posterior > max_posterior:
			max_posterior = posterior
			out = label
	return out

confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]

data, labels = read_file()
num_correct = 0
for i in range(len(data)):
	training_data, training_labels, test_sample, test_label = input(data, labels, i)
	classes = get_classes(training_data, training_labels)
	means = findMeans(classes)
	stds = findStds(classes)
	prior = {}
	for label in classes:
		prior[label] = len(classes[label])/len(training_labels)
	predicted_label = predict(test_sample,prior,means,stds)
	row_num = class_map[test_label]
	col_num = class_map[predicted_label]
	confusion_matrix[row_num][col_num] += 1
	if predicted_label == test_label:
		num_correct += 1
accuracy = num_correct/len(data)
print("Number of correctly classified samples:", num_correct)
print("Total number of samples tested:", len(data))
print("Accuracy (% of correctly classified samples):", str(round(accuracy*100,2))+"%")
print("Confusion Matrix", confusion_matrix)
