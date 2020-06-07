import matplotlib.pyplot as plt
import torchvision.transforms as transforms
class Plotter():
    def __init__(self):
        self.toPIL = transforms.ToPILImage()
        plt.ion()
    def ShowOneImage(self, tensor, title=None):
        image = tensor.cpu().clone()
        plt.figure()
        image = image.squeeze(0)
        image = self.toPIL(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
    def ShowMultipleImages(self, tensors,titles = []):
        plt.figure()
        for i in range(len(tensors)):
            image = tensors[i].cpu().clone()
            plt.subplot(1,len(tensors),i+1)
            image = image.squeeze(0)
            image = self.toPIL(image)
            plt.imshow(image)
            if len(titles)>=len(tensors):
                plt.title(titles[i])
            plt.axis('off')
        plt.pause(0.001)
        