import numpy as np
import random

#calculate distance from centroids to each point
#D[i][j] is the distance from the point i to the centroids j
def dist_ps_fast(X, M):
    X2 = np.sum(X*X, 0)
    M2 = np.sum(M*M, 0)
    D = (X.T).dot(M)
    for i in range(len(X2)):
        for j in range(len(M2)):
            D[i][j] = M2[j] + X2[i] - 2*D[i][j]
    return D

#assign cluster for each point
def assign_cluster(D, label):
    for i in range(len(D)):
        temp = -1
        min = 9999999999
        for j in range(len(D[i])):
            if(D[i][j]<min):
                temp = j
                min = D[i][j]
        for j in range(len(D[i])):
            if(j == temp):
                label[i][j] = 1
            else:
                label[i][j] = 0
    return label

#predict centroids
def predict_centroids(X, label):
    M = X.dot(label)
    total = np.sum(label, 0)
    for i in range(len(total)):
        for j in range(len(M)):
            M[j][i] = M[j][i] / total[i]
    return M

#randomly create k centroids
def randomly_create_centroids(train_data):
    centroids = list()
    for i in range(20):
        temp = random.randrange(len(train_data))
        centroids.append(train_data[temp])
    a = np.array(centroids)
    return a

if __name__ == "__main__":
    #Read data from input file
    train_data_file = open("train_tf_idf.txt", "r")
    input_train_data = train_data_file.readlines()

    test_data_file = open("test_tf_idf.txt", "r")
    input_test_data = test_data_file.readlines()

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

    usable_test_data = list()
    for each_data in test_data:
        temp = [0] * 10290
        for each_item in each_data:
            temp1 = each_item.split(":")
            temp[int(temp1[0])] = float(temp1[1])
        usable_test_data.append(temp)

    used_train_data = np.array(usable_train_data)
    centroids = randomly_create_centroids(used_train_data)
    label = [[0] * 20]*len(usable_train_data)
    i = 1
    while True:
        D = dist_ps_fast(used_train_data.T, centroids.T)
        label = assign_cluster(D, label)
        print("[{}] Update cluster centroid and label for each example.".format(i))
        i+=1
        temp_centroids = predict_centroids(used_train_data.T,label).T
        if((temp_centroids == centroids).all()):
            break
        else:
            centroids = temp_centroids
    print("Label for each example: ",label, "\n")
