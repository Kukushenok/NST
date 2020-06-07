import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import os
from nst.exceptions import NoSuchLayerException,InvalidHCoefficientException
from nst.plotter import Plotter
from nst.pretrained_data import GetPretrainedData
import copy
from IPython.display import clear_output

def gram_matrix(input):
        batch_size , H, W, f_map_num = input.size()
        features = input.view(batch_size * H, W * f_map_num) 
        G = torch.mm(features, features.t())
        return G.div(batch_size * H * W * f_map_num)

class NSTContentLoss(nn.Module):
    def __init__(self, task):
        super(NSTContentLoss, self).__init__()
        self.target = task.GetContentImage().detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class NSTStyleLoss(nn.Module):
    def __init__(self, task):
        super(NSTStyleLoss, self).__init__()
        self.target_features = []
        self.task_device = task.device
        for style_data in task.GetStyleData():
            addition = {"style_image":gram_matrix(style_data["style_image"]).detach(),"weight":style_data["weight"]}
            self.target_features.append(addition)
        self.loss = F.mse_loss(self.target_features[0]["style_image"], self.target_features[0]["style_image"])

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = torch.zeros((1)).to(self.task_device)
        for feature in self.target_features:
            self.loss += F.mse_loss(G, feature["style_image"])*feature["weight"]
        
        return input
        
class NSTNormalization(nn.Module):
    
    def __init__(self, mean, std):
        super(NSTNormalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class NST():
    def OptimizeFN(self):
        self.input_image.data.clamp_(0, 1)
        self.optimizer.zero_grad()
        self.model(self.input_image)
        
        style_score = 0
        content_score = 0

        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss
        style_score *= self.style_weight
        content_score *= self.content_weight

        loss = style_score + content_score
        loss.backward(retain_graph=True)

        self.iterations += 1
        clear_output(wait=True)
        
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            style_score.item(), content_score.item()))
        
        self.buffer.append(copy.deepcopy(self.input_image))
        
        if len(self.buffer)>=self.buffer_length: self.buffer.pop(0)
        
        self.current_loss = style_score + content_score
        
        return style_score + content_score
            
    def __init__(self,task,pretrained_type = "vgg19", buffer_length = 10):
        self.current_task = task
        self.device = task.device
        data = GetPretrainedData(pretrained_type,self.device)
        self.normalization = NSTNormalization(data["mean"],data["std"]).to(self.device)
        cnn = copy.deepcopy(data["model"])

        self.content_losses = []
        self.style_losses = []
        self.buffer_length = buffer_length
        self.model = nn.Sequential(self.normalization)

        i = 0  
        
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise NoSuchLayerException('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in data["content_layers"]:
                target = self.model(self.current_task.GetContentImage()).detach()
                content_loss = NSTContentLoss(target)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in data["style_layers"]:
                style_loss = NSTStyleLoss(self.current_task.GetStyleData())
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], NSTContentLoss) or isinstance(self.model[i], NSTStyleLoss):
                break

        self.model = self.model[:(i + 1)]
        self.optimizer = optim.LBFGS([self.current_task.GetContentImage().requires_grad_()]) 
        self.image_buffer = []
        self.previous_loss = None
        self.current_loss = None
        
        
    def Run(self,max_steps=500,style_weight = 1e+5,content_weight = 1, H = 0.7, non_stop = False):
        if (H>1 or H<0): raise InvalidHCoefficientException("H cannot be lesser than 0 and bigger than 1")
        self.input_image = self.current_task.GetContentImage()
        self.iterations = 0
        previousExpBuffer = None
        while self.iterations <= max_steps:
            self.optimizer.step(self.OptimizeFN)
            if previousExpBuffer is None:
                previousExpBuffer = self.current_loss
            else:
                wasBuffer = previousExpBuffer
                previousExpBuffer = previousExpBuffer*H+(1-H)*self.current_loss
                if(wasBuffer<previousExpBuffer):
                    print("WARNING: The model might be overtrained. Do you want to proceed the training? (Y if yes)")
                    result = input()
                    if result.upper()!="Y":
                        labels = ["" for i in self.buffer_length]
                        labels[self.buffer_length//2] = "Select the nicest result (left = 1, right = {0})".format(self.buffer_length)
                        Plotter().ShowMultipleImages(self.image_buffer)
                        self.input_image = self.buffer[int(input())-1]
                        break
        self.input_image.data.clamp_(0, 1)
        return self.input_image
    