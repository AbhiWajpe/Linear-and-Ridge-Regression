# %%
import numpy as np
import matplotlib.pyplot as plt

# Importing the training dataset

trng = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_train.csv", delimiter=',',
                encoding='utf8')

t_train = trng[:,6]

# Importing the test dataset
test = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_test.csv", delimiter=',',
                encoding='utf8')

t_test = test[:,6]

phi_train = np.zeros([len(trng),36])

for i in range(1,7):
  phi_train[:,6*(i-1):(6*i)]  = trng[:,0:6]**i
phi_train_max = np.max(phi_train, axis=0, keepdims=True)
phi_train_min = np.min(phi_train, axis=0, keepdims=True)
phi_train = (phi_train - phi_train_min)/(phi_train_max - phi_train_min)
ones_array = np.ones([len(phi_train),1])
phi_train = np.c_[ones_array,phi_train]

phi_test = np.zeros([len(test),36])
for i in range(1,7):
   phi_test[:,6*(i-1):(6*i)]  = test[:,0:6]**i
phi_test_max = np.max(phi_test, axis=0, keepdims=True)
phi_test_min = np.min(phi_test, axis=0, keepdims=True)
phi_test = (phi_test - phi_train_min)/(phi_train_max - phi_train_min)
ones_array = np.ones([len(phi_test),1])
phi_test = np.c_[ones_array,phi_test]

def range_of_lambda(l):
    l = np.exp(l)
    w_train = np.linalg.inv(phi_train.T @ phi_train + l * np.eye(37)) @ phi_train.T @ t_train
    y_train_pred = np.dot(phi_train,w_train)
    error = np.sqrt(np.mean((y_train_pred-t_train)**2))
    return error, w_train 

weights = [0]*41
y_test_pred = [0]*41
training_error = np.zeros(41)
testing_error = np.zeros(41)
lambdas = list(range(-30, 11))
for i, l in enumerate(lambdas):
    training_error[i], weights[i] = range_of_lambda(l)
    y_test_pred[i] = np.dot(phi_test,weights[i])
    testing_error[i] = np.sqrt(np.mean((y_test_pred[i]-t_test)**2))

# %%
# plotting the RMS error against each features 
plt.plot(list(range(-30,11)),training_error,marker='.',markersize=7.5)
plt.plot(list(range(-30,11)),testing_error,marker='.',markersize=7.5)
plt.xlabel('$ln\lambda$')
plt.ylabel('RMS Error')
plt.legend(['Training Error','Test Error'])
plt.title('Training error and Test error as RMSE against $ln\lambda$')
plt.show()



# %%
