from sklearn.metrics import confusion_matrix
#y_actu = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 ,1 ,1 ,1 ,1]
#y_pred = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,1 ,1 ,0 ,1 ,0 ,1]
#y_pred =  [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1 ,1 ,0 ,1]

def fmeasure(y_pred, y_actu):
    
    cm = confusion_matrix(y_pred, y_actu)
    print(cm)
    
    TP = float(cm[0][0])
    TN = float(cm[1][1])
    FN = float(cm[1][0])
    FP = float(cm[0][1])

    accuracy = float(format((TP+TN)/(TP+FP+TN+FN), '.4f'))    
    precision = float(format(TP/(TP+FP), '.4f'))
    recall = float(format(TP/(TP+FN), '.4f'))
    F1 = float(format(2*precision*recall/(precision+recall), '.4f'))
    
    return accuracy, F1