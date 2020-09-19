import torch
import config
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import model
from tqdm import tqdm
from utils.helpers import draw_seg_maps, label_colors_list
from utils.helpers import save_model_dict
from utils.metrics import eval_metric

class Trainer:
    def __init__(self, model, train_data_loader, train_dataset, valid_data_loader, valid_dataset, classes_to_train):
        super(Trainer, self).__init__()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print('OPTIMIZER INITIALIZED')
        self.criterion = nn.CrossEntropyLoss() 
        print('LOSS FUNCTION INITIALIZED')

        self.train_data_loader = train_data_loader
        self.train_dataset = train_dataset
        self.valid_data_loader = valid_data_loader
        self.valid_dataset = valid_dataset
        self.model = model
        self.num_classes = len(classes_to_train)
        print(f"NUM CLASSES: {self.num_classes}")

    def fit(self, epoch, epochs):
        print('Training')
        model.train()
        train_running_loss = 0.0
        train_running_inter, train_running_union = 0, 0
        train_running_correct, train_running_label = 0, 0
        for i, data in tqdm(enumerate(self.train_data_loader), 
                            total=int(len(self.train_dataset)/self.train_data_loader.batch_size)):
            data, target = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            outputs = outputs['out']

            ##### BATCH-WISE LOSS #####
            loss = self.criterion(outputs, target)
            train_running_loss += loss.item()
            ###########################

            ##### BATCH-WISE METRICS ####
            correct, num_labeled, inter, union = eval_metric(outputs, 
                                                             target, 
                                                             self.num_classes)
            train_running_inter += inter
            train_running_union += union
            #############################

            ##### BACKPROPAGATION AND PARAMETER UPDATION #####
            loss.backward()
            self.optimizer.step()
            ##################################################
            
        ##### PER EPOCH LOSS #####
        train_loss = train_running_loss / len(self.train_data_loader.dataset)
        ##########################

        ##### PER EPOCH METRICS ######
        IoU = 1.0 * train_running_inter / (np.spacing(1) + train_running_union)
        mIoU = IoU.mean()
        ##############################
        return train_loss, mIoU

    def validate(self, epoch, epochs):
        print('Validating')
        model.eval()
        valid_running_loss = 0.0
        valid_running_inter, valid_running_union = 0, 0
        valid_running_correct, valid_running_label = 0, 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.valid_data_loader), 
                                total=int(len(self.valid_dataset)/self.valid_data_loader.batch_size)):
                data, target = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
                outputs = self.model(data)
                outputs = outputs['out']
                
                # save the validation segmentation maps every...
                # ... last batch of each epoch
                if i == int(len(self.valid_dataset)/self.valid_data_loader.batch_size) - 1:
                    draw_seg_maps(outputs, label_colors_list, epoch, i)

                ##### BATCH-WISE LOSS #####
                loss = self.criterion(outputs, target)
                valid_running_loss += loss.item()
                ###########################

                ##### BATCH-WISE METRICS ####
                correct, num_labeled, inter, union = eval_metric(outputs, 
                                                                target, 
                                                                self.num_classes)
                valid_running_inter += inter
                valid_running_union += union
                #############################
            
        ##### PER EPOCH LOSS #####
        valid_loss = valid_running_loss / len(self.valid_data_loader.dataset)
        ##########################

        ##### PER EPOCH METRICS ######
        IoU = 1.0 * valid_running_inter / (np.spacing(1) + valid_running_union)
        mIoU = IoU.mean()
        ##############################
        return valid_loss, mIoU

    def save_model(self, epochs):
        save_model_dict(self.model, epochs, self.optimizer, self.criterion)