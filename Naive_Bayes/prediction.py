import time
#Separate dataset by class of age
def separate_by_age(dataset):
    separated = dict()
    for data in dataset:
        if(data[0] not in separated):
            separated[data[0]] = list()
        separated[data[0]].append(data)
    return separated

#calculate probability for each group in each group age
def calculate_probability_for_each_group(dataset):
    probability_for_group = dict()
    for label in dataset:
        if(label not in probability_for_group):
            probability_for_group[label] = dict()
            probability_for_group[label]["total"] = 0
        for item in dataset[label]:
            for i in range(1, len(item)):
                if (item[i] not in probability_for_group[label]):
                    probability_for_group[label][item[i]] = 0
                probability_for_group[label][item[i]] += 1
            probability_for_group[label]["total"] += 1
    for label in probability_for_group:
        for i in probability_for_group[label]:
            if(i == "total"):
                continue
            probability_for_group[label][i] = (probability_for_group[label][i] +1 )*100000/ probability_for_group[label]["total"]
    return probability_for_group

#calculate probabilities for each input data
def calculate_probabilities_for_each_data(data, probability):
    probabilities = dict()
    for label in probability:
        temp = 1
        for i in range(1, len(data)):
            if(data[i] not in probability[label]):
                temp *= 1000 / probability[label]["total"]
            else:
                temp*= probability[label][data[i]]
        probabilities[label] = temp
    return probabilities

#prediction label for each input data
def prediction_age_group(data, probability):
    probabilities = calculate_probabilities_for_each_data(data, probability)
    max = 0
    result = "not defined"
    for label in probabilities:
        if(probabilities[label] > max):
            result = label
            max = probabilities[label]
    return result

#Read data from input file
train_data_file = open("agedetector_group_train.v1.0.txt", "r")
input_train_data = train_data_file.readlines()
train_data_file.close()

#Read test data from input file
test_data_file = open("test.txt", "r")
input_test_data = test_data_file.readlines()
test_data_file.close()

#Processing data
train_data = list()
for each_line in input_train_data:
    train_data.append(each_line.split())

test_data = list()
for each_line in input_test_data:
    test_data.append(each_line.split())

#Separated dataset
separated_data = separate_by_age(train_data)
probability = calculate_probability_for_each_group(separated_data)

#prediction
prediction = list()
start = time.process_time()
for each_data in test_data:
    prediction.append(prediction_age_group(each_data, probability))
print(time.process_time() - start)
#print(prediction)

exactly = 0
for i in range(len(prediction)):
    if (prediction[i] == test_data[i][0]):
        exactly +=1

for i in range(len(prediction)):
    if(prediction[i] == "not defined"):
        print(i)
print(exactly / 2123)
