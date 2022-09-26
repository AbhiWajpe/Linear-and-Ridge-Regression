import numpy as np
import matplotlib.pyplot as plt
from  math import sqrt

# Importing the training dataset
trng = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_train.csv", delimiter=',',
                encoding='utf8')

t_train = trng[:,6]

# constructing the phi matrix with different features from training dataset
def No_of_features_in_training(m):
 
    phi_train = np.zeros([len(trng),6*m])
    for i in range(1,m+1):
      phi_train[:,6*(i-1):(6*i)]  = trng[:,0:6]**i
      
    # Normalizing the phi matrix for train dataset between 0 and 1  
    phi_train_max = np.max(phi_train, axis=0, keepdims=True)
    phi_train_min = np.min(phi_train, axis=0, keepdims=True)
    phi_train = (phi_train - phi_train_min)/(phi_train_max - phi_train_min)
    # Adding the bais of 1 at the begining 
    ones_array = np.ones([len(phi_train),1])
    phi_train = np.c_[ones_array,phi_train]
    # Calculating the weights 
    w_train = np.linalg.solve(np.dot(phi_train.T,phi_train),np.dot(phi_train.T,t_train))
    # Calculating the predicted output for training dataset
    y_train_pred = np.dot(phi_train,w_train)
    # Calculating the training RMS error
    error = sqrt(np.mean((y_train_pred-t_train)**2))
    return error, w_train

training_error = np.zeros(6)
weights = [0] * 6
for i in range(6):
    training_error[i], weights[i] = No_of_features_in_training(i + 1)

# Importing the test dataset
test = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_test.csv", delimiter=',',
                encoding='utf8')

t_test = test[:,6]

# constructing the phi matrix with different features from test dataset
def No_of_features_in_test(m):
    phi_test = np.zeros([len(test),6*m])
    for i in range(1,m+1):
      phi_test[:,6*(i-1):(6*i)]  = test[:,0:6]**i
      # Normalizing the phi matrix for train dataset between 0 and 1  
    phi_test_max = np.max(phi_test, axis=0, keepdims=True)
    phi_test_min = np.min(phi_test, axis=0, keepdims=True)
    phi_test = (phi_test - phi_test_min)/(phi_test_max - phi_test_min)
    # Adding the bais of 1 at the begining 
    ones_array = np.ones([len(phi_test),1])
    phi_test = np.c_[ones_array,phi_test]
    return phi_test

phi_test = [0]*6
y_test_pred = [0]*6
testing_error = np.zeros(6)

for i in range(6):
    phi_test[i] = No_of_features_in_test(i + 1)
    # Calculating the predicted output for test dataset
    y_test_pred[i] = np.dot(phi_test[i],weights[i])
    # Calculating the training RMS error
    testing_error[i] = np.sqrt(np.mean((y_test_pred[i]-t_test)**2))

# plotting the RMS error against each features 
m = np.array([1,2,3,4,5,6])
plt.plot(m,training_error,marker='o')
plt.plot(m,testing_error,marker='o')
plt.xlabel('No of features(m)')
plt.ylabel('Error')
plt.legend(['Training Error','Test Error'])
plt.title('Training error and Test error as RMSE against m')
plt.show()


