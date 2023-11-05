from minitorch import tensor

t1 = [1.234,5.23,5.32,21.5,123.4,]

t2 = tensor(t1)
for i in range(len(t1)):
    print(t1[i] == t2[i])