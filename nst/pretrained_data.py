import torchvision.models as models
import torch
from nst.exceptions import GetAndAssert
pretrained_data = {"vgg19":
 {"mean":torch.tensor([0.485, 0.456, 0.406]),
  "std":torch.tensor([0.229, 0.224, 0.225]),
  "content_layers":['conv_4'],
  "style_layesrs":['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']}
}

def GetPretrainedData(model_type,device = "cuda"):
    dataToReturn = GetAndAssert(pretrained_data,model_type)
    model = None
    exec("model = models.{0}(pretrained=True).features.to(device).eval()".format(model_type)) #idk how to make that without exec
    dataToReturn["model"] = model
    return dataToReturn