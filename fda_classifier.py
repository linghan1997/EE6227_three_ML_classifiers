import numpy as np


class Fisher:
    def __init__(self, data_train, label_train, num_classes=3):
        self.s_w = 0  # 4*4
        self.s_b = 0  # 4*4
        self.s_total = self.s_w + self.s_b
        self.x = data_train
        self.y = label_train
        self.num_classes = num_classes

    def train(self):
        # calculate means
        self.total_mean = 0
        self.mean_cls = np.zeros((self.num_classes, self.x.shape[1]))
        self.num_cls = np.zeros((self.num_classes, 1))

        for i in range(self.num_classes):
            self.mean_cls[i, :] = np.mean(self.x[np.where(self.y == i + 1)[0]], axis=0)  # 3*4
            self.num_cls[i, :] = self.x[np.where(self.y == i + 1)[0]].shape[0]  # 3*1
            self.total_mean += self.mean_cls[i] * self.num_cls[i]

        self.total_mean /= np.sum(self.num_cls)

        # calculate covariances
        for i in range(self.num_classes):
            s = np.matmul((self.x[np.where(self.y == i + 1)[0]] - self.mean_cls[i]).T,
                          self.x[np.where(self.y == i + 1)[0]])
            self.s_w += s
            self.s_b += self.num_cls[i] * np.outer((self.mean_cls[i] - self.total_mean),
                                                   (self.mean_cls[i] - self.total_mean).T)

        self.s_total = self.s_w + self.s_b

        # find c-1 largest eigen-vectors
        eigen_val, eigen_vec = np.linalg.eig(np.matmul(np.linalg.pinv(self.s_w), self.s_b))

        # get the original idx from large to small
        idx = np.argsort(-eigen_val)
        # choose c-1 largest
        idx = idx[:2]
        self.W = np.array([eigen_vec[i] for i in idx])  # 2 * 4

        # calculate bias
        g_x = np.dot(self.W, self.x.T).T  # m * 2

        mean_wave = np.zeros((g_x.shape[1], self.num_classes))  # 2 * 3 means of different projections on w
        for i in range(self.num_classes):
            mean_wave[:, i] = np.mean(g_x[np.where(self.y == i + 1)[0]], axis=0)  # axis for num_class

        # print(mean_wave)
        self.bias = np.array([- (mean_wave[0, 0] + mean_wave[0, 2]) / 2,
                              - (mean_wave[1, 0] + mean_wave[0, 1]) / 2])
        self.bias = self.bias.reshape(-1, 1)

    def predict(self, data_test):
        res = []

        for _, x in enumerate(data_test):
            x = x.reshape(-1, 1)
            g_x = np.dot(self.W, x) + self.bias  # 2 * 1, g1(x) and g2(x)

            if g_x[0][0] > 0 and g_x[1][0] < 0:
                res.append(1)
            elif g_x[0][0] < 0 and g_x[1][0] > 0:
                res.append(2)
            elif g_x[0][0] < 0 and g_x[1][0] < 0:
                res.append(3)
            elif g_x[0][0] > 0 and g_x[1][0] > 0:
                if g_x[0][0] > g_x[1][0]:
                    res.append(1)
                else:
                    res.append(2)

        print(res)
