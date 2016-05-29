import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os

epoch = np.linspace(1, 30, 30)

fig, ax1 = plt.subplots()
filename = './baseline'
tr_acc = np.load(os.path.join(filename, 'tr_accuracy.npy'))
tr_loss = np.load(os.path.join(filename, 'tr_loss.npy'))
step = tr_loss.shape[0]
steps = np.linspace(1, step, step)*100

loss, = ax1.plot(steps, tr_loss, 'g', label='Training_Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax2 = ax1.twinx()
acc, = ax2.plot(steps, tr_acc, 'b', label='Batch_train_accuracy')
ax2.set_ylabel('Accuracy')

plt.title('Experiment1: Training_Loss witht the Batch_train_accuracy')
plt.legend(handles=[acc, loss])
plt.show()