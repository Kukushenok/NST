import json
from PIL import Image
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import torchvision.transforms as transforms
#import torchvision.models as models
import os
from nst.exceptions import InvalidSettingsException,NoSolutionException,NoSettingsException,InvalidDataException,GetAndAssertPath,GetAndAssertInt, GetAndAssertFloat
from nst.plotter import Plotter
class DataLoader():
    
    def __OpenImage(self,image_path):
        image = Image.open(image_path)
        image = self.image_loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def __ApplySettings(self):
        isValid = ""
        
        try:
            self.image_size =self.settings.get("image_size",None)
            if not isinstance(self.image_size,list): self.image_size = GetAndAssertInt(self.settings, "image_size")
            elif(len(self.image_size)!=2): assert (False, "No-square image size can be set only by [X,Y]")
            else: self.image_size[0],self.image_size[1]=self.image_size[1],self.image_size[0]
            self.image_loader = transforms.Compose([
                transforms.Resize(self.image_size),  
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()])
            
            self.content_image = GetAndAssertPath(self.settings,"content_image",self.solution_name)
            self.content_image = self.__OpenImage(self.content_image)
            self.epoch_count = self.settings.get("max_epochs",-1)
            self.style_data = []
            for data in self.settings.get("style_data"):
                style_image = GetAndAssertPath(data,"style_image",self.solution_name)

                style_image = self.__OpenImage(style_image)
                
                weight = GetAndAssertFloat(data,"weight")
                
                assert (weight>=0,"Weights can't be lesser than 0")
                
                style = {"style_image":style_image,"weight":weight}
                
                self.style_data.append(style)
            
        except Exception as e: isValid = e.args[0]
        
        if isValid: raise InvalidSettingsException("Task are not in correct format. Read README.md for documentation. Error: {0}".format(isValid))
        
        self.__CheckStyleData()
        
    def __CheckStyleData(self):
        sumOfWeights = sum(list(map(lambda x: x["weight"],self.style_data)))
        shape = self.content_image.shape
        if sumOfWeights>1:
            print("WARNING: Sum of weights are not equals 1. They were converted to do that.")
            for i in range(len(self.style_data)):
                self.style_data[i]["weight"]/=sumOfWeights
                if(self.style_data[i]["style_image"].shape!=shape):
                    raise InvalidDataException("Some pictures have different channels")
                    
    def GetNumberOfStyles(self):
        return len(list(filter(lambda x: x["weight"]!=0,self.style_data)))
    
    def __init__(self, solution_name, device = "cuda"):
        
        self.device = device
        
        if not os.path.exists(solution_name):
            raise NoSolutionException("Cannot find task {0}".format(solution_name))
        self.solution_name = solution_name
        self.settings_path = self.solution_name+"/settings.json"
        
        if not os.path.exists(self.settings_path):
            raise NoSettingsException("Cannot find settings.json in task {0}".format(solution_name))
        
        with open(self.settings_path,"r") as file:
            self.settings = json.loads(file.read())
            
        self.__ApplySettings()
        self.binded_NST = None
        
    def PlotData(self):
        plotter = Plotter()
        images = [self.content_image]
        labels = ["Content"]
        for data in self.style_data: 
            images.append(data["style_image"])
            labels.append("Style (weight = {0})".format(data["weight"]))
    
        plotter.ShowMultipleImages(images,labels)
    
    def GetStyleData(self):
        return self.style_data
    
    def GetContentImage(self):
        return self.content_image
        
        