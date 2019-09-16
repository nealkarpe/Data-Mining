import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def read_file():
	data = []
	labels = []
	with open("heart-disease.csv", "r") as f:
		reader = list(csv.reader(f))
		attributes_row = list(reader[0])[:-1]
		for i in range(1,len(reader)):
			row = list(reader[i])
			labels.append(int(row[-1]))
			row = list(map(float, row[:-1]))
			data.append(np.array(row))
	data = np.array(data)
	labels = np.array(labels)
	return data, labels, attributes_row

def input(data, labels):
	inds = np.arange(data.shape[0])
	np.random.shuffle(inds)
	data = data[inds]
	labels = labels[inds]
	train_test_ratio = 0.7
	num_train_samples = int(train_test_ratio*data.shape[0])
	return data[:num_train_samples], labels[:num_train_samples], data[num_train_samples:], labels[num_train_samples:]

def classify(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X,y)
    return model

data, labels, attributes_row = read_file()
accuracy_sum = 0
accuracy_percentages = []
# importance_sum = np.array([0.0]*len(attributes_row))
num_iters = 500
for iteration in range(num_iters):
	training_data, training_labels, test_data, test_labels = input(data, labels)
	model = classify(training_data, training_labels)
	# importance_sum += np.array(model.feature_importances_)
	pred = model.predict(test_data)
	accuracy = accuracy_score(test_labels, pred)
	# print("Split #" + str(iteration+1) + ":", str(round(accuracy*100,2))+"%")
	print(iteration)
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
# importances = (importance_sum/num_iters)*100

# fig, ax = plt.subplots()
# width = 0.5 # the width of the bars 
# ax.barh(attributes_row, importances, width)
# plt.title('Importance % of features')
# plt.ylabel('Feature',size=12)
# plt.xlabel('Importance %',size=12)
# for i, v in enumerate(importances):
#     ax.text(v + 0.08, i-0.14, str(round(v,2))+"%", color='dimgrey', fontweight='bold', size=14)
# plt.figure()
# plt.hist(accuracy_percentages)
# plt.title("Historgram of accuracy % for " + str(num_iters) + " random 70-30 splits")
# plt.xlabel("Accuracy %")
# plt.ylabel("Number of splits")
# plt.show()




plt.hist(accuracy_percentages)
# plt.title("Historgram of accuracy % for " + str(num_iters) + " random 70-30 splits")
plt.tick_params(axis='both', which='major', labelsize=32)
plt.xlabel("accuracy %",size=32)
plt.ylabel("num of iterations",size=32)
plt.show()