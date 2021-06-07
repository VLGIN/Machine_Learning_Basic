import numpy as np
import random
import math

#calculate distance
def dist_calculate(X, M):
    X2 = np.sum(X*X, 0)
    M2 = np.sum(M*M, 0)
    D = (X.T).dot(M)
    for i in range(len(X2)):
        for j in range(len(M2)):
            D[i][j] = M2[j] + X2[i] - 2*D[i][j]
    return D

# find_min
def find_min(X):
    temp = -1
    min = 999999999
    for i in range(len(X)):
        if(X[i] < min):
            temp = i
            min = X[i]
    return temp

# assign cluster for each point
def assign_cluster(D, label):
    for i in range(len(D)):
        min_index = find_min(D[i])
        label[i] = [0] * len(D[i])
        label[i][min_index] = 1
    return label

# predict centroids
def predict_centroids(X, label):
    M = X.dot(label)
    total = np.sum(label, 0)
    for i in range(len(total)):
        for j in range(len(M)):
            M[j][i] = M[j][i] / total[i]
    return M

#create random centroids
def random_centroids(X):
    temp = X.T
    result = list()
    for i in range(20):
        index = random.randrange(len(temp))
        result.append(temp[index])
    a = np.array(result)
    a = a.T
    return a

if __name__ == "__main__":
    #Read data from input file
    train_data_file = open("train_tf_idf.txt", "r")
    input_train_data = train_data_file.readlines()

    test_data_file = open("test_tf_idf.txt", "r")
    input_test_data = test_data_file.readlines()

    #get the label of data for checking
    train_label_for_checking = [0] * 20
    for each_data in input_train_data:
        train_label_for_checking[int(each_data[0:each_data.index("<")])] += 1

    test_label_for_checking = [0] * 20
    for each_data in input_test_data:
        test_label_for_checking[int(each_data[0:each_data.index("<")])] += 1

    print("Result for checking: ")
    print("Train data: ")
    print(train_label_for_checking)
    print("Test data: ")
    print(test_label_for_checking)
    #Processing data
    train_data = list()
    for each_line in input_train_data:
        a = each_line.rindex(">")
        temp = each_line[(a+1):]
        train_data.append(temp.split())

    test_data = list()
    for each_line in input_test_data:
        a = each_line.rindex(">")
        temp = each_line[(a+1):]
        test_data.append(temp.split())

    # Transform the raw data to the dense vector
    usable_train_data = list()
    for each_data in train_data:
        temp = [0]*10290
        for each_item in each_data:
            temp1 = each_item.split(":")
            temp[int(temp1[0])] = float(temp1[1])
        usable_train_data.append(temp)
    train_data_for_used = np.array(usable_train_data)
    train_data_for_used = train_data_for_used.T

    usable_test_data = list()
    for each_data in test_data:
        temp = [0] * 10290
        for each_item in each_data:
            temp1 = each_item.split(":")
            temp[int(temp1[0])] = float(temp1[1])
        usable_test_data.append(temp)
    test_data_for_used = np.array(usable_test_data)
    test_data_for_used = test_data_for_used.T

    label = [[0] * 20]*len(usable_train_data)

    # main
    i = 1
    centroids = random_centroids(train_data_for_used)
    while True:
        D = dist_calculate(train_data_for_used, centroids)
        print("[{}] Update cluster centroid and label for each example.".format(i))
        i += 1
        label = assign_cluster(D, label)
        temp_centroids = predict_centroids(train_data_for_used, label)
        if((temp_centroids == centroids).all()):
            print("The final cluster for each point in train data: ")
            centroids = temp_centroids
            result = [0] * 20
            for each_data in label:
                result[each_data.index(1)]+= 1
            print(result)
            break
        else:
            centroids = temp_centroids

    # take the test data into model
    D = dist_calculate(test_data_for_used, centroids)
    label_for_test_data = assign_cluster(D, label)
    result = [0] * 20
    print("The cluster for each point in test data: ")
    for each_data in label_for_test_data:
        result[each_data.index(1)] += 1
    print(result)
