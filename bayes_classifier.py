import numpy
import numpy as np


class Bayes:
    def __init__(self, data_train, label_train, num_classes=3):
        self.x = data_train  # 120 * 4
        self.y = label_train  # 120 * 1
        self.num_classes = num_classes

    def train(self):
        self.mean_cls = []
        self.cov_cls = []

        # calculate parameters for maximum likelihood
        for i in range(self.num_classes):
            # y_shape is 120*1, only use the index in 1st dimension
            x_mean = np.mean(self.x[np.where(self.y == (i + 1))[0]], axis=0)  # 4*1

            x_cov = np.cov(self.x[np.where(self.y == (i + 1))[0]].T)  # 4*4
            # by hand
            # x_cov = np.matmul((self.x[np.where(self.y == (i + 1))[0]].T - x_mean),
            #                   (self.x[np.where(self.y == (i + 1))[0]].T - x_mean).T) / \
            #         self.x[np.where(self.y == (i + 1))[0]].T.shape[1]

            self.mean_cls.append(x_mean)
            self.cov_cls.append(x_cov)

        # calculate prior
        idx = np.arange(1, self.num_classes + 1)  # [1, 2, 3]
        num_idx = self.y == idx
        self.prior = np.sum(num_idx, 0) / self.y.shape[0]  # 1*3

    def predict(self, data_test):
        res = []
        post_prob = np.zeros(self.num_classes)

        for _, x in enumerate(data_test):
            for i in range(self.num_classes):
                first_term = 1 / (((2 * np.pi) ** (4 / 2)) * (np.linalg.det(self.cov_cls[i]) ** 0.5))

                second_term = np.exp(
                    - 0.5 * np.matmul(np.matmul(np.array([(x - self.mean_cls[i])]), np.linalg.pinv(self.cov_cls[i])),
                                      np.array([x - self.mean_cls[i]]).T))
                cond_prob = first_term * second_term

                post_prob[i] = self.prior[i] * cond_prob

            res.append(np.argmax(post_prob) + 1)

        return res

    def cal_acc(self):
        pred = self.predict(self.x)  # test on the training dataset
        pred = np.array(pred)
        truth = self.y.reshape(-1,)
        accuracy = np.sum(pred == truth) / truth.shape[0]
        return accuracy
