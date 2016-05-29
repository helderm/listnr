import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os

epoch = np.linspace(1, 30, 30)

plt.subplot(1, 2, 1)
filename = './psize_1'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err_1 = 1 - val_acc
val_1, = plt.plot(epoch, val_err_1, label='Size 1')


filename = './psize_3'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err_3 = 1 - val_acc
val_3, = plt.plot(epoch, val_err_3, label='Size 3')

filename = './psize_5'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err_5 = 1 - val_acc
val_5, = plt.plot(epoch, val_err_5, label='Size 5')

#plt.plot(val_step, val_err, train_step, train_err)
plt.title('Error rate for different pooling size')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.legend(handles=[val_1, val_3, val_5])


plt.subplot(1, 2, 2)
filenames = ['./ksize_quarter', './ksize_half', './baseline', './ksize_2x', './ksize_4x' ]

val_acc = np.load(os.path.join(filenames[0], 'val_accuracies.npy'))
val_err_1 = 1 - val_acc
val_1, = plt.plot(epoch, val_err_1, label='1/4')

val_acc = np.load(os.path.join(filenames[1], 'val_accuracies.npy'))
val_err_2 = 1 - val_acc
val_2, = plt.plot(epoch, val_err_2, label='1/2')

val_acc = np.load(os.path.join(filenames[2], 'val_accuracies.npy'))
val_err_3 = 1 - val_acc
val_3, = plt.plot(epoch, val_err_3, label='baseline')

val_acc = np.load(os.path.join(filenames[3], 'val_accuracies.npy'))
val_err_4 = 1 - val_acc
val_4, = plt.plot(epoch, val_err_4, label='2x')

val_acc = np.load(os.path.join(filenames[4], 'val_accuracies.npy'))
val_err_5 = 1 - val_acc
val_5, = plt.plot(epoch, val_err_5, label='4x')

plt.title('Error rate for different kernel size')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.legend(handles=[val_1, val_2, val_3, val_4, val_5])

plt.show()