import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file():
	data = []
	labels = []
	with open("heart-disease.csv", "r") as f:
		reader = list(csv.reader(f))
		attributes_row = list(reader[0])
		for i in range(1,len(reader)):
			row = list(reader[i])
			labels.append(int(row[-1]))
			row = list(map(float, row[:-1]))
			data.append(np.array(row))
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

def input(data, labels):
	inds = np.arange(data.shape[0])
	np.random.shuffle(inds)
	data = data[inds]
	labels = labels[inds]
	train_test_ratio = 0.7
	num_train_samples = int(train_test_ratio*data.shape[0])
	return data[:num_train_samples], labels[:num_train_samples], data[num_train_samples:], labels[num_train_samples:]

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

data, labels = read_file()
accuracy_sum = 0
accuracy_percentages = []
num_iters = 10000
for iteration in range(num_iters):
	training_data, training_labels, test_data, test_labels = input(data, labels)
	classes = get_classes(training_data, training_labels)
	means = findMeans(classes)
	stds = findStds(classes)
	prior = {}
	for label in classes:
		prior[label] = len(classes[label])/len(training_labels)
	correct_count = 0
	for i in range(len(test_data)):
		predicted_label = predict(test_data[i],prior,means,stds)
		if predicted_label == test_labels[i]:
			correct_count += 1
	accuracy = correct_count/len(test_data)
	print("Split #" + str(iteration+1) + ":", str(round(accuracy*100,2))+"%")
	accuracy_sum += accuracy
	accuracy_percentages.append((accuracy*100))
avg_accuracy = accuracy_sum/num_iters
min_accuracy = min(accuracy_percentages)
max_accuracy = max(accuracy_percentages)
print("-----------------------------")
print("Min accuracy:", str(round(min_accuracy,2))+"%")
print("Max accuracy:", str(round(max_accuracy,2))+"%")
print("Average accuracy:", str(round(avg_accuracy*100,2))+"%")
print("-----------------------------")

plt.hist(accuracy_percentages)
plt.title("Historgram of accuracy % for " + str(num_iters) + " random 70-30 splits")
plt.xlabel("Accuracy %")
plt.ylabel("Number of splits")
plt.show()
