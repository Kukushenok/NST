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
from torch.optim.lr_scheduler import StepLR
import copy

def gram_matrix(input):
        batch_size , H, W, f_map_num = input.size()
        features = input.view(batch_size * H, W * f_map_num) 
        G = torch.mm(features, features.t())
        return G.div(batch_size * H * W * f_map_num)

class NSTContentLoss(nn.Module):
    def __init__(self, task):
        super(NSTContentLoss, self).__init__()
        self.target = task.binded_NST.model(task.GetContentImage()).detach()
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
            addition = {"style_image":gram_matrix(task.binded_NST.model(style_data["style_image"])).detach(),"weight":style_data["weight"]}
            self.target_features.append(addition)
        self.loss = F.mse_loss(self.target_features[0]["style_image"], self.target_features[0]["style_image"])

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = 0
        for feature in self.target_features:
            self.loss += F.mse_loss(G, feature["style_image"])*feature["weight"]
        
        return input
        
class NSTNormalization(nn.Module):
    
    def __init__(self, mean, std, device = "cuda"):
        super(NSTNormalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
        
    def forward(self, img):
        return ((img - self.mean) / self.std)

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
        if(self.iterations%self.output_delay==0):
            print("Style Score: {0}; Content Score: {1}; Learning Rate: {2}".format(
                style_score.item(), content_score.item(),self.sheduler.get_lr()[0]))
        self.sheduler.step()
        
        self.current_loss = style_score + content_score
        
        if(self.current_loss<self.min_loss or self.image_buffer is None):
            self.image_buffer = copy.deepcopy(self.input_image)
            self.min_loss = self.current_loss
            
        return style_score + content_score
            
    def __init__(self,task,pretrained_type = "vgg19", sheduler_gamma = 0.5,sheduler_step = 200):
        self.current_task = task
        task.binded_NST = self
        self.device = task.device
        data = GetPretrainedData(pretrained_type,self.device)
        self.normalization = NSTNormalization(data["mean"],data["std"]).to(self.device)
        cnn = copy.deepcopy(data["model"])
        self.max_steps = task.epoch_count
        self.content_losses = []
        self.style_losses = []
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
                content_loss = NSTContentLoss(self.current_task)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in data["style_layers"]:
                style_loss = NSTStyleLoss(self.current_task)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], NSTContentLoss) or isinstance(self.model[i], NSTStyleLoss):
                break

        self.model = self.model[:(i + 1)]
        self.optimizer = optim.LBFGS([self.current_task.GetContentImage().requires_grad_()])
        self.sheduler = StepLR(self.optimizer, step_size=sheduler_step, gamma=sheduler_gamma)
        self.image_buffer = None
        self.min_loss = -1
        self.previous_loss = None
        self.current_loss = None
        
        
    def Run(self,max_steps=-1,style_weight = 1e+5,content_weight = 1, H = 0.7, non_stop = False, output_delay = 50):
        if(max_steps>0): self.max_steps = max_steps
        elif(self.max_steps<0): self.max_steps = 500
        if (H>1 or H<0): raise InvalidHCoefficientException("H cannot be lesser than 0 and bigger than 1")
        
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.output_delay = output_delay
        self.input_image = self.current_task.GetContentImage()
        self.iterations = 0
        previousExpBuffer = None
        while self.iterations <= self.max_steps:
            self.optimizer.step(self.OptimizeFN)
            if previousExpBuffer is None:
                previousExpBuffer = self.current_loss
            else:
                wasBuffer = previousExpBuffer
                previousExpBuffer = previousExpBuffer*H+(1-H)*self.current_loss
                if(wasBuffer<previousExpBuffer and not non_stop):
                    print("WARNING: The model might be overtrained (slight loss function increased). Training cycle is stopped, make non_stop = True if you want to ignore that")
                    self.input_image = self.image_buffer
                    break
        self.input_image.data.clamp_(0, 1)
        return self.input_image
    