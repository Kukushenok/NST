import json
from PIL import Image
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import torchvision.transforms as transforms
#import torchvision.models as models
import os
import exceptions

class DataLoader():
    
    def __OpenImage(self,image_path):
        image = Image.open(image_path)
        image = self.image_loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def __ApplySettings(self):
        isValid = True
        
        try:
            self.image_size = exceptions.GetAndAssertInt(self.settings, "image_size")
            
            self.image_loader = transforms.Compose([
                transforms.Resize(self.image_size),  
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()])
            
            self.content_image = exceptions.GetAndAssertPath(self.settings,"content_image")
            self.content_image = self.__OpenImage(self.content_image)
            
            self.style_data = []
            for data in self.settings.get("style_data"):
                style_image = exceptions.GetAndAssertPath(data,"style_image")

                style_image = self.__OpenImage(style_image)
                
                weight = exceptions.GetAndAssertInt(data,"weight")
                
                style = {"style_image":style_image,"weight":weight}
                
                self.style_data.append(style)
            
        except Exception: isValid = False
        
        
        if not isValid: raise exceptions.InvalidSettingsException("Solution are not in correct format. Read README.md for documentation")
    
    def __init__(self, solution_name, device = "cuda"):
        
        self.device = device
        
        if not os.path.exists(solution_name):
            raise exceptions.NoSolutionException("Cannot find {0}".format(solution_name))
            
        self.settings_path = solution_name+"/settings.json"
        
        if not os.path.exists(self.settings_path):
            raise exceptions.NoSettingsException("Cannot find settings in {0}".format(solution_name))
        
        with open(self.settings_path,"r") as file:
            self.settings = json.loads(file.read())
            
        self.__ApplySettings()
        
        