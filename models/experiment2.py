import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os


epoch = np.linspace(1, 30, 30)

plt.subplot(1, 2, 1)
filename = './baseline'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err = 1 - val_acc
val, = plt.plot(epoch, val_err, label='Validation')

train_acc = np.load(os.path.join(filename, 'train_accuracies.npy'))
train_err = 1 - train_acc
train, = plt.plot(epoch, train_err, label='Training')
plt.ylim([0.1, 0.5])

plt.title('Error rate of baseline model')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.legend(handles=[val, train])



plt.subplot(1, 2, 2)
filename = './psize_6'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err = 1 - val_acc
val, = plt.plot(epoch, val_err, label='Validation')

index = np.argmin(val_err)
x = np.ones(20)*index
y = np.linspace(0, val_err[index], 20)
plt.plot(x, y, 'r.')

train_acc = np.load(os.path.join(filename, 'train_accuracies.npy'))
train_err = 1 - train_acc
train, = plt.plot(epoch, train_err, label='Training')

plt.title('Error rate of one layer model')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.ylim([0.1, 0.5])
plt.legend(handles=[val, train])


plt.show()
