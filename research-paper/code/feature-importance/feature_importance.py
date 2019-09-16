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

def classify(X, y):
    model = RandomForestClassifier(n_estimators=50000)
    model.fit(X,y)
    return model

data, labels, attributes_row = read_file()
model = classify(data, labels)
importances = np.array(model.feature_importances_)*100

fig, ax = plt.subplots()
width = 0.5 # the width of the bars 
ax.barh(attributes_row, importances, width, label="importance %")
plt.tick_params(axis='y', which='major', labelsize=30)
plt.tick_params(axis='x', which='major', labelsize=26)
ax.legend(loc=4,prop={'size': 22, 'weight': 'bold'})
# plt.title('Importance % of features')
# plt.ylabel('Feature',size=12)
# plt.xlabel('Importance %',size=12)
for i, v in enumerate(importances):
    ax.text(v + 0.08, i-0.14, str(round(v,2))+"%", color='dimgrey', fontweight='bold', size=22)
plt.show()