import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
	return data, labels, attributes_row

def input(data, labels, test_ind):
	train_samples = data[:test_ind] + data[test_ind+1:]
	train_labels = labels[:test_ind] + labels[test_ind+1:]
	test_sample = data[test_ind]
	test_label = labels[test_ind]
	return train_samples, train_labels, test_sample, test_label

def classify(X, y):
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X,y)
    return model

data, labels, attributes_row = read_file()
num_correct = 0
# importance_sum = np.array([0.0]*len(attributes_row))
for i in range(len(data)):
	print(i)
	training_data, training_labels, test_sample, test_label = input(data, labels, i)
	model = classify(training_data, training_labels)
	# importance_sum += np.array(model.feature_importances_)
	pred = model.predict([test_sample])[0]
	if pred == test_label:
		num_correct += 1
accuracy = num_correct/len(data)
print("Number of correctly classified samples:", num_correct)
print("Accuracy (% of correctly classified samples):", str(round(accuracy*100,2))+"%")

# importances = (importance_sum/len(data))*100
# fig, ax = plt.subplots()
# width = 0.5 # the width of the bars 
# ax.barh(attributes_row, importances, width)
# plt.title('Importance % of features')
# plt.ylabel('Feature',size=12)
# plt.xlabel('Importance %',size=12)
# for i, v in enumerate(importances):
#     ax.text(v + 0.08, i-0.14, str(round(v,2))+"%", color='dimgrey', fontweight='bold', size=14)
# plt.show()