import re
import numpy as np
import matplotlib.pyplot as plt

path = ""

log_data, acc = [], []

with open(path + "trainlog5.txt") as train_log:
    lines = train_log.readlines()
    for line in lines:
        entry = re.search(r"[\d]+[.][\d]+e?-?\d*", line)
        if entry != None:
            acc.append(float(entry.group(0).split("\t")[-1]))
        if len(acc) == 8:
            log_data.append(acc)
            acc = []

train_data = np.array([log_data[i] for i in range(len(log_data)) if not i % 2]) # time, cross ent, cluster, sep, avg sep, accu, l1, dist pair
test_data = np.array([log_data[i] for i in range(len(log_data)) if i % 2])      # time, cross ent, cluster, sep, avg sep, accu, l1, dist pair
train_data, test_data = train_data.T, test_data.T       


train_loss, test_loss = train_data[1], test_data[1]
train_accu, test_accu = train_data[5] / 100, test_data[5] / 100

# x = np.array(range(len(train_loss)))
plt.plot(train_loss, c = "tab:red", label = "train loss")
# plt.plot(x, train_accu, c = "darkred", label = "train accuracy")
plt.plot(test_loss, c = "tab:blue", label = "test loss")
# plt.plot(test_accu, c = "midnightblue", label = "test accuracy")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Cross Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
