'''
    Evaluater is module that evaluates the output to calculate Confusion matrix for the predictions done
    by the classifiers
'''
import csv

def calculate_parameters(output_directory, classifier):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    output_file_name = 'testDataCSV_'+classifier+'.csv'
    reader = csv.reader(open(output_directory+ '/' + output_file_name, 'r',newline=''))
    for row in reader:
        if(row[1] == '1' and row[2] == 'sponsored'):
            TP = TP + 1
        elif(row[1] == '1' and row[2] == 'unsponsored'):
            FN = FN + 1
        elif(row[1] == '0' and row[2] == 'unsponsored'):
            TN = TN + 1
        elif(row[1] == '0' and row[2] == 'sponsored'):
            FP = FP + 1
    return TP,FP,TN,FN
            
def calculate_measures(TP,FP,TN,FN):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    error_rate = ((FP+FN)*100)/(TP+TN+FP+FN)
    return accuracy,recall,precision,error_rate

def evaluator(output_directory,classifier ):
    TP,FP,TN,FN = calculate_parameters(output_directory, classifier)
    accuracy,recall,precision,error_rate = calculate_measures(TP,FP,TN,FN)
    return accuracy,recall,precision,error_rate
    
    
    
    
    
    
    
    
    
    