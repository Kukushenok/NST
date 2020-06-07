import matplotlib.pyplot as plt
import torchvision.transforms as transforms
class Plotter():
    def __init__(self):
        self.toPIL = transforms.ToPILImage()
    def ShowImage(self,tensor, title=None):
        image = tensor.cpu().clone()
        print(image.shape)
        image = image.squeeze(0)
        image = self.toPIL(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) 