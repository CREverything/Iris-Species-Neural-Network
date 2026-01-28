import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

df = pd.read_csv("./iris.csv")
data = np.matrix(df)
sep_len, sep_wid, pet_len, pet_wid, species = [], [], [], [], []

for row in data:
    sep_len.append(row[0, 0])
    sep_wid.append(row[0, 1])
    pet_len.append(row[0, 2])
    pet_wid.append(row[0, 3])
    species.append(row[0, 4])

def reformat_target(categories):
    categories_ref = []
    
    for i in range(len(categories)):
        if(categories[i] == 'Iris-versicolor'):
            categories_ref.append([1, 0, 0])
        elif(categories[i] == 'Iris-virginica'):
            categories_ref.append([0, 1, 0])
        else:
            categories_ref.append([0, 0, 1])
    
    return categories_ref

combined = list(zip(sep_len, sep_wid, pet_len, pet_wid, species))
random.shuffle(combined)

combined = np.array(combined)

input = combined[:, :4]
target = combined[:, 4:]

training_input = input[:int(len(combined) * 0.8)].astype(float)
training_target = target[:int(len(combined) * 0.8)]
training_target_ref = reformat_target(training_target)

testing_input = input[int(len(combined) * 0.8):].astype(float)
testing_target = target[int(len(combined) * 0.8):]
testing_target_ref = reformat_target(testing_target)

weight_in = np.random.uniform(-0.1, 0.1, (4,6))
bias_in = [0, 0, 0, 0, 0, 0]

weight_out = np.random.uniform(-0.1, 0.1, (6,3))
bias_out = [0, 0, 0]

def relu(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    
    return x

def softmax(x):

    x_max = x[0]

    for i in range(len(x)):
        if(x[i] > x_max):
            x_max = x[i]

    for i in range(len(x)):
        x[i] -= x_max

    sum = 0
    for i in range(len(x)):
        x[i] = math.exp(x[i])
        sum += x[i]

    for i in range(len(x)):
        x[i] /= sum
    
    return x

def forward_step(sample, in_weight, in_bias, out_weight, out_bias):
    value = np.dot(sample, in_weight)
    value = np.add(value, in_bias)

    pre_activation = value
        
    value = relu(value)
    hidden_layer = value
    
    value = np.dot(value,out_weight)
    value = np.add(value, out_bias)

    output = softmax(value)

    return output, hidden_layer, pre_activation

def loss_calc(estimate, target, index):

    for channel in range(len(target[index])):
        if target[index][channel] == 1:
            loss_val = -math.log(estimate[channel] + 0.00000000001)
    
    return loss_val

def test_calc(estimate, target):
    
    pred_class = 0
    max_prob = estimate[0]
    
    for channel in range(len(estimate)):
        if estimate[channel] > max_prob:  # Use 'channel' not 'i'
            max_prob = estimate[channel]
            pred_class = channel
    
    # Find which index has the 1 in target
    actual_class = 0
    for i in range(len(target)):
        if target[i] == 1:
            actual_class = i
            break
    
    # Return 1 if correct, 0 if wrong
    if pred_class == actual_class:
        return 1
    else:
        return 0
    
def relu_diff(pre_act):

    relu_array = []

    for i in range(len(pre_act)):
        if(pre_act[i] <= 0):
            relu_array.append(0)
        else:
            relu_array.append(1)

    return relu_array


def backprop(prediction, hidden, target, weight, input, pre_act):
    output_gradient= np.subtract(prediction, target)
    weight_out_gradient = np.outer(hidden, output_gradient)
    bias_out_gradient = output_gradient

    hidden_gradient = np.dot(output_gradient, np.transpose(weight))
    hidden_gradient = hidden_gradient * relu_diff(pre_act)
    bias_in_gradient = hidden_gradient
    weight_in_gradient = np.outer(input, hidden_gradient)

    return output_gradient, weight_out_gradient, bias_out_gradient, bias_in_gradient, weight_in_gradient

def gradient_descent(out_weight, out_bias, in_bias, in_weight, w_in, w_out, b_in, b_out):
    
    step = 0.01
    
    w_in -= in_weight * step
    b_in -= in_bias * step
    b_out -= out_bias * step
    w_out -= out_weight * step

    return w_in, w_out, b_in, b_out

epoch_list = []
epoch_count = []

forward, hidden_layer_activations, pre_activation = [], [], []

for val in range(1,100):

    loss = 0

    for i, row in enumerate(training_input):
        forward, hidden_layer_activations, pre_activation = forward_step(row, weight_in, bias_in, weight_out, bias_out)
        gradient_out, gradient_weight_out,  gradient_bias_out, gradient_bias_in, gradient_weight_in = backprop(forward, hidden_layer_activations, training_target_ref[i], weight_out, row, pre_activation)
        weight_in, weight_out, bias_in, bias_out = gradient_descent(gradient_weight_out,  gradient_bias_out, gradient_bias_in, gradient_weight_in, weight_in, weight_out, bias_in, bias_out)
        loss += loss_calc(forward , training_target_ref, i)

    loss/= len(training_input)
    epoch_list.append(loss)
    epoch_count.append(val)
    #print(loss)
  

def test_output(testing_data, targets):
    correct = 0
    for i, row in enumerate(testing_data):
        tests, dummy_1, dummy_2 = forward_step(row, weight_in, bias_in, weight_out, bias_out)
        correct += test_calc(tests, targets[i])
    percentage = 100 * (correct/ len(testing_data))
    print(percentage, "%")

test_output(testing_input, testing_target_ref)

plt.scatter(epoch_count, epoch_list)
plt.show()