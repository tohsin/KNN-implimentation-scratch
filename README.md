# KNN Implimentation

python3 -m venv env to create virtual environmentw
pip install -r requirements.txt to install all libraries necessary

## Introduction

An implementation of a classifier using the K - Nearest Neighbour algorithm. The algorithm is tested on data and result metrics are provided. The model used performed suitable on the acting an expert system for determining decision to buy a vehicle.

### Folder
The src folder contains the python class to create a KNN model, the math functions to find distance between data point and a utility function to split the data set

The Dataset folder contains the diabetes data set and with it the training data and output data after spliting the  diabetes data set

### Notebooks
The DataPreprocessing file is the notebook used to load the data set and analyse it, as well as peform the split between data and output data and is then saved as files to the Dataset folder

The TrainTestModel notebook is where the model is tested and analysed, The data is loaded from their respective folder.

The dataframes get converted to numpy array and using the utilties module the trainTestsplit to split the data 80 percent to 20.

The model is creted using the KNN Classfier class the fit function is called passing the train data, the correspoding labels and the distance metrics to be used.

The predict method is then called and compared to the corresponding labels to obtained the  accuracy

The predict method is tried with multiple k values to obtain the maxmimum accuracy possible

This process is done multiple times with diffrent normalisation and sclaing algorithms to test the best accuraccy


Finally using the best model obtained we plot the confuisaon matrix  and confusion report

### Using The KNN Classifier
After intiialisation of  a KNN calssifer object or a model the fit funtion has to be called to cache in the data, labels and teh distance metrics to be used.

Following that the predict function can then be called to obtain predictions using the KNN algorithm  and the accuracy to get accuracy of the model

### Similarity metrics
This file contains the distance functions namely
-euclidian distance
-manhattan distance
-cosine similarity distance
-jaccard distance