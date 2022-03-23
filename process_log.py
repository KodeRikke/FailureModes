import re
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/OhRai/desktop/IAI/"
coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}
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

#train_loss, test_loss = np.log(train_data[1]), np.log(test_data[1])
train_accu, test_accu = train_data[5] / 100, test_data[5] / 100
train_loss = (coefs['crs_ent'] * train_data[1] + coefs['clst'] * train_data[2] + coefs['sep'] * train_data[3] + coefs['l1'] * train_data[6])
test_loss = (coefs['crs_ent'] * test_data[1] + coefs['clst'] * test_data[2] + coefs['sep'] * test_data[3] + coefs['l1'] * test_data[6])

# x = np.array(range(len(train_loss)))
plt.plot(train_loss, c = "tab:red", label = "train loss")
plt.plot(train_accu, c = "darkred", label = "train accuracy")
plt.plot(test_loss, c = "tab:blue", label = "test loss")
plt.plot(test_accu, c = "midnightblue", label = "test accuracy")
plt.axvline(5, c = "darkorange", label = "warm epochs")
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.title('Cross Validation Loss (remember log transformed)')
plt.legend()
plt.grid(True)
plt.show()
