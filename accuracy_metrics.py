

import numpy as np



def try_it():

    y_true =['M', 'F', 'M', 'F', 'O', 'F', 'M', 'F', 'M', 'O', 'M', 'F', 'O', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'O', 'F', 'M', 'F', 'M', 'F', 'M', 'O', 'O', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'O']
    y_pred =['F', 'F', 'M', 'O', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'O', 'F', 'F', 'M', 'O', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'O', 'F', 'F', 'M', 'O', 'F', 'F', 'M', 'M', 'M', 'F', 'O']


    results = generate_classification_metrics(y_true, y_pred, skip_label='O', confusion_csv="confusion.csv", precisions_csv="precisions.csv")
    print(results)

    ''' 
    results:
    {'N_counts': [17, 13, 7], 'uar': 43.67, 'war': 43.33, 'precision': 0.405, 'recall': 0.405, 'f1_score': 0.405, 'precision_sk': 0.433, 'recall_sk': 0.419, 'f1_score_sk': 0.426}


    confusion_csv:

    True:,F,M,O
    F,7,7,3
    M,6,6,1
    O,4,1,2


    precisions_csv:

    Class,N,P,R,F1
    F,45.95%,41.2,41.2,41.2
    M,35.14%,46.2,42.9,44.5
    O,18.92%,28.6,33.3,31.0
    Mean,,38.6,39.1,39.1
    Overall,37,40.5,40.5,40.5
    Overall_skiped,30,43.3,41.9,42.6

    '''


def get_accuracy_metrics(confusion_matrix, skip=-1):
    '''
    Given a confusion matrix, calculates unweighted (balanced among classes) and weighted (typical) accuracy.
    '''
    class_acc_arr = np.array([])
    true_n_arr = np.array([])
    total_correct = 0
    total_n = 0
    
    for row in range(0, len(confusion_matrix)):
        if(row!=skip) and (np.nansum(confusion_matrix[row]) > 5): #skipping O
            true_n = np.nansum(confusion_matrix[row])
            
            total_correct += confusion_matrix[row, row]
            total_n += true_n
            class_acc_arr = np.append(class_acc_arr, confusion_matrix[row, row] / true_n)
            true_n_arr = np.append(true_n_arr, true_n)
        #else:
            #class_recalls = np.append(class_recalls, confusion_matrix[row, row] * 0)
            #class_weights = np.append(class_weights, np.nansum(confusion_matrix[row]) / np.nansum(confusion_matrix))
    uar = np.mean(class_acc_arr)
    class_weights_arr = true_n_arr/total_n
    war = np.nansum(class_acc_arr*class_weights_arr)
    
    prec_skip_O = total_correct/total_n #same as precision_wO


    true_pos = np.diag(confusion_matrix)
    #np.seterr(divide='ignore', invalid='ignore')
    #print( true_n_arr)
    #print( np.nansum(confusion_matrix, axis=0))
    non_zero_cm_sum_0 = np.array([(x if x!=0 else 1) for x in np.nansum(confusion_matrix, axis=0)])
    non_zero_cm_sum_1 = np.array([(x if x!=0 else 1) for x in np.nansum(confusion_matrix, axis=1)])
    #print(non_zero_cm_sum_0)
    #print(np.nansum(confusion_matrix, axis=0))
    
    all_precisions = (true_pos / non_zero_cm_sum_1)
    all_recalls = (true_pos / non_zero_cm_sum_0)
    


    precision =  np.sum(((true_pos / non_zero_cm_sum_1) * np.nansum(confusion_matrix, axis=1)))/np.nansum(confusion_matrix)
    recall =  np.sum(((true_pos / non_zero_cm_sum_0) * np.nansum(confusion_matrix, axis=0)))/np.nansum(confusion_matrix)
    f1_score = ((precision+recall)/2)
    #calculating without O
    precision_wO =  (true_pos / non_zero_cm_sum_1) * np.nansum(confusion_matrix, axis=1)
    precision_wO = np.delete(precision_wO, skip, 0)
    precision_wO =  np.nansum(precision_wO)/np.nansum(np.delete(confusion_matrix, skip, 0))

    recall_wO =  (true_pos / non_zero_cm_sum_0) * np.nansum(confusion_matrix, axis=0)
    recall_wO = np.delete(recall_wO, skip, 0)
    recall_wO =  np.nansum(recall_wO)/np.nansum(np.delete(confusion_matrix, skip, 1))
    f1_wO = (precision_wO+recall_wO)/2
    #print(prec_skip_O, precision_wO, recall_wO)
    

    return round(uar*100, 2), round(war*100, 2), round(precision,3), round(recall,3), round(f1_score,3), round(prec_skip_O,3), round(recall_wO,3), round(f1_wO,3), all_precisions, all_recalls

def create_conf_matrix(expected, predicted, classes_unique):
        n_classes = len(classes_unique)
        m = np.zeros((n_classes, n_classes), dtype=np.uint16)

        for i in range(len(expected)):
            for exp in range(n_classes):
                for pred in range(n_classes):
                    if(expected[i]==classes_unique[exp]) and (predicted[i]==classes_unique[pred]):
                        m[exp][pred] += 1
        return m




def save_stats(confusion, headings, all_precisions, all_recalls, N_counts, precision, recall, f1_score, precision_wO, recall_wO, f1_wO, skip_label=None, confusion_csv=None, precisions_csv=None):

    import csv
    
    if (confusion_csv is not None):
        with open(confusion_csv, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', lineterminator = '\n')
            true_head = [*["True:"], *headings]
            writer.writerow(true_head)
            for row in range(0,len(confusion)):
                this_row = [*[headings[row]], *confusion[row]]
                writer.writerow(this_row)
    index_of_O = -1
    if (precisions_csv is not None):
        with open(precisions_csv, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', lineterminator = '\n')
            true_head = ["Class", "N", "P", "R", "F1"]
            writer.writerow(true_head)
            for row in range(0,len(headings)):
                if((skip_label is not None) and (headings[row]==skip_label)): index_of_O = row
                N_perc = N_counts[row]*100/sum(N_counts) if (N_counts[row] > 0) else 0
                this_row = [headings[row], str(round(N_perc, 2))+"%", round(all_precisions[row]*100,1), round(all_recalls[row]*100,1), round((all_precisions[row]+all_recalls[row])*100/2,1) ]
                writer.writerow(this_row)
            writer.writerow(["Mean", "", round(np.mean(all_precisions)*100,1), round(np.mean(all_recalls)*100,1), round((np.mean(all_recalls)+np.mean(all_recalls))*100/2,1) ])
            writer.writerow(["Overall", sum(N_counts), round(precision*100,1), round(recall*100,1), round(f1_score*100,1) ])
            if (index_of_O >= 0):
                writer.writerow(["Overall_skiped", sum(N_counts)-N_counts[index_of_O], round(precision_wO*100,1), round(recall_wO*100,1), round(f1_wO*100,1) ])
        


def generate_classification_metrics(y_true, y_pred, skip_label=None, confusion_csv=None, precisions_csv=None):
    '''
    `skip_label`: The row (or col) of the confusion matric to skip when calculating UAR. If you want to calculate metrics without considering a particular label. E.g, in calculating UAR skipped label would be given zero weight when calculating the final mean. It's useful to eliminate a class that dominates the results so that other classes can be analyzed.
    '''
    #print("generate_classification_metrics")
    np_unique_y = np.sort(np.unique(y_true))
    #print(np_unique_y)
    conf = create_conf_matrix(y_true, y_pred, np_unique_y)
    uar, war, precision, recall, f1_score, precision_wO, recall_wO, f1_wO, all_precisions, all_recalls = get_accuracy_metrics(conf, skip=(np.where(np_unique_y == skip_label )[0][0]) if skip_label is not None else -1)
    
    N_counts = []
    
    for lbl in range(len(np_unique_y)):
        
        N_counts.append(list(y_true).count(np_unique_y[lbl]))
   
    
    
    save_stats(conf, np_unique_y, all_precisions, all_recalls, N_counts, precision, recall, f1_score, precision_wO, recall_wO, f1_wO, skip_label, confusion_csv, precisions_csv)

    return {"N_counts": N_counts, "uar": uar, "war": war, "precision": precision, "recall": recall, "f1_score": f1_score, "precision_sk": precision_wO, "recall_sk": recall_wO, "f1_score_sk": f1_wO}
    



        
if __name__ == "__main__":
    try_it()
