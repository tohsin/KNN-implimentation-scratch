
from src.similarity_metrics import euclidian_distance, cosine_similarity,jaccard_distance,manhattan_distance
from scipy.stats import mode
import numpy as np
class  KNNClassfier:
    def __init__(self):
        self.metrics=None
        self.train_data=None   
        self.labels=None     
    def fit(self,train_data,labels,metrics="euclidian_distance"):
        self.metrics=metrics
        self.train_data=train_data        
        self.labels=labels
    def metrics_dist(self, x1, x2, metrics):
        if metrics=="euclidian_distance":
            return euclidian_distance(x1,x2)
        elif (metrics=="manhattan_distance"):
            return manhattan_distance(x1,x2)
        elif (metrics=="cosine_similarity"):
            return cosine_similarity(x1,x2)
        elif (metrics=="jaccard_distance"):
            return jaccard_distance (x1,x2)
        else:
            print("metrics not recoginised")
            return "Error"
    
    def predict(self, test_data, k):
        if not (self.metrics and self.labels and self.metrics):
            print ("Please call teh fit function first ")
            return
        #code adapted from K-Nearest Neighbors from Scratch with Python - AskPython, 2022
        #loop through eavch data to be classified
        
        predictions = []
        for i in range(len(test_data)):  
            distances = []
            for j in range(len(self.train_data)):
                euc_dist=self.metrics_dist(np.array(self.train_data[j,:]),test_data[i],self.metrics)
                distances.append(euc_dist)
            distances=np.array(distances)
            dist = np.argsort(distances)[:k] # get the first k elements of lowest euclidian distance
            labels_ = self.labels[dist]
         
            #Majority voting
            labels_occ = mode(labels_) 
            lab = labels_occ.mode[0]
            # add prediction to the main list
            predictions.append(lab)
    
        return predictions
        #end of adapted code
    
    # def accuracy(self,predicted_labels, actual_labels):
    #     diff = predicted_labels - actual_labels
    #     return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
    
    def accuracy(self, pred , y_test):
        count = 0
        for i in range(len(pred)):
            # if values are the same then model is correct
            if pred[i] == y_test[i]:
                count +=1
        # print("Accuracy =", (count/len(pred))*100, "%")   
        return  count/len(pred)
