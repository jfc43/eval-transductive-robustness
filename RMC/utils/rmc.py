from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.lib import AverageMeter
import copy

BUFFER_SIZE = 10000
BATCH_SIZE = 128
K = 1024
EPOCHS = 100
EARLY_STOP = 5
LEARNING_RATE = 2.5e-5

class RMC:
    def __init__(self, model_path, aug_x, aug_y, feature_ds):
        self.aug_x = aug_x
        self.aug_y = aug_y
        self.feature_ds = feature_ds
        self.model_path = model_path
        
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.k = K
        self.epochs = EPOCHS
        self.early_stop = EARLY_STOP
        self.learning_rate = LEARNING_RATE
        
        # Load pretrained model
        self.base_model = torch.load(model_path).cuda()
        self.base_model.eval()
        self.model = torch.load(model_path).cuda()
        
        # Evaluate the performance of RMC
        self.loss_object = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loss = AverageMeter()
        self.train_accuracy = AverageMeter()

        self.test_loss = AverageMeter()
        self.test_accuracy = AverageMeter()
    
    def update_model(self, new_model):
        self.model = copy.deepcopy(new_model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def update_base_model(self, new_model):
        self.base_model = copy.deepcopy(new_model)
        self.base_model.eval()
        
    def train_step(self, images, labels):
        self.model.train()
        outputs = self.model(images)
        preds = torch.argmax(outputs, axis=1)
        loss = self.loss_object(outputs, labels)
        acc = torch.mean((preds==labels).float())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss.update(loss, images.shape[0])
        self.train_accuracy.update(acc, images.shape[0])

    def test_step(self, images, labels):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
        preds = torch.argmax(outputs, axis=1)
        loss = self.loss_object(outputs, labels)
        acc = torch.mean((preds==labels).float())
        self.test_loss.update(loss, images.shape[0])
        self.test_accuracy.update(acc, images.shape[0])
        
    def get_pred(self, adv_img):
        # Load test data
        adv_img = adv_img.cuda()

        # Extract feature
        with torch.no_grad():
            adv_repre = self.base_model.get_feature(adv_img).cpu().numpy()[0]

        # Find KNN based on Euclidean distance, where K=4096
        l2_norm_list = []
        for feature in self.feature_ds:
            l2_norm = np.linalg.norm(adv_repre - feature, ord=2)
            l2_norm_list.append(l2_norm)

        # Find KNN based on WebNN and DeepNN defense mechanisms
        l2_norm = np.array(l2_norm_list)
        sorted_indices = np.argsort(l2_norm)
        knn_top20 = sorted_indices[:int(K*0.2)]
        knn_idx = sorted_indices[int(K*0.2):K]
        
        # Local adaptation
        self.adapt(knn_top20, knn_idx)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(adv_img)
        preds = torch.argmax(outputs, axis=1)

        return preds
        
    def adapt(self, knn_top20, knn_idx):
        # Random split top 20% NN examples to form a validation set
        X_train, X_test, y_train, y_test = train_test_split(self.aug_x[knn_top20], 
                                                            self.aug_y[knn_top20], 
                                                            test_size=0.25)
        
        # Create data for local adaptation
        X_train = np.concatenate((X_train, self.aug_x[knn_idx]))
        y_train = np.concatenate((y_train, self.aug_y[knn_idx]))
        BUFFER_SIZE = len(X_train)

        X_train = torch.Tensor(X_train) 
        y_train = torch.Tensor(y_train).long()
        train_dataset = TensorDataset(X_train, y_train)
        train_ds = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 

        X_test = torch.Tensor(X_test) 
        y_test = torch.Tensor(y_test).long()
        test_dataset = TensorDataset(X_test, y_test)
        eval_ds = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 

        # Adapting with early stop
        min_loss = np.inf
        count = 0

        for epoch in range(EPOCHS):
            for images, labels in train_ds:
                images = images.cuda()
                labels = labels.cuda()
                self.train_step(images, labels)

            for test_images, test_labels in eval_ds:
                test_images = test_images.cuda()
                test_labels = test_labels.cuda()
                self.test_step(test_images, test_labels)

            # Record minimum loss
            if self.test_loss.avg < min_loss:
                min_loss = self.test_loss.avg
                count = 0
            else:
                count += 1

            curr_train_loss = self.train_loss.avg

            # Reset the metrics for the next epoch
            self.train_loss.reset()
            self.train_accuracy.reset()
            self.test_loss.reset()
            self.test_accuracy.reset()

            # Early stop if val loss does not decrease
            if curr_train_loss < 0.01 and count >= EARLY_STOP:
                break
                
        return epoch
        
    def calibrate(self):
        # Random sample K examples from augmented dataset
        M = int(len(self.aug_x)/5)
        random_idx = np.random.randint(M, size=K)
        X_train, X_test, y_train, y_test = train_test_split(self.aug_x[random_idx], 
                                                            self.aug_y[random_idx], 
                                                            test_size=0.125)
        BUFFER_SIZE = len(X_train)

        X_train = torch.Tensor(X_train) 
        y_train = torch.Tensor(y_train).long()
        train_dataset = TensorDataset(X_train, y_train)
        train_ds = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 

        X_test = torch.Tensor(X_test) 
        y_test = torch.Tensor(y_test).long()
        test_dataset = TensorDataset(X_test, y_test)
        eval_ds = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 

        # Calibrating with early stop
        min_loss = np.inf
        count = 0

        for epoch in range(EPOCHS):
            for images, labels in train_ds:
                images = images.cuda()
                labels = labels.cuda()
                self.train_step(images, labels)

            for test_images, test_labels in eval_ds:
                test_images = test_images.cuda()
                test_labels = test_labels.cuda()
                self.test_step(test_images, test_labels)

            # Record minimum loss
            if self.test_loss.avg < min_loss:
                min_loss = self.test_loss.avg
                count = 0
            else:
                count += 1

            curr_train_loss = self.train_loss.avg

            # Reset the metrics for the next epoch
            self.train_loss.reset()
            self.train_accuracy.reset()
            self.test_loss.reset()
            self.test_accuracy.reset()

            # Early stop if val loss does not decrease
            if curr_train_loss < 0.01 and count >= EARLY_STOP:
                break
                
