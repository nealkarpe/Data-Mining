# Iris Dataset Classification

### Data Visualization

###### 4-D plot of samples of the dataset

* Run `python3 plot_classes.py` (make sure `iris.csv` is in same directory)
	* Output will be `class_visualization.png`

###### Classification

* Uses Naive Bayes algorithm with Gaussian Distribution as probabilistic model.
* Run `python3 holdout.py` to evaluate classifier using Holdout Method
* Run `python3 cross-validation.py` to evaluate classifier using Cross Validation Method (leave-out-1)
* The results will be printed to stdout
