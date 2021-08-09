import torch, numpy as np
from torchvision.datasets import CIFAR10, MNIST

def image_transform(x):
    x = np.array(x, dtype='float32') / 255
    x = np.transpose(x, (2, 0, 1))
    x = 2*x - 1
    return torch.from_numpy(x)

def image_transform2(x):
    x = np.array(x, dtype='float32') / 255
    x = 2*x - 1
    return torch.from_numpy(x).unsqueeze_(0)

def inv_image_transform(x):
    x = x.detach().cpu().numpy()
    return (x+1)/2

class realdata(torch.utils.data.Dataset):
    def __init__(self, max_len=5000, d_name='cifar-10'):
        super(realdata, self).__init__()
        self.__name = d_name.lower()
        self.__max_len = max_len
        if self.__name == "mnist":
            self.__dataset = MNIST('./data', train=True, transform=image_transform2, download=True)
        else:
            self.__dataset = CIFAR10('./data', train=True, transform=image_transform, download=True)

    def __getitem__(self, index):
        return self.__dataset[index%self.__max_len]
    
    def __len__(self):
        return self.__max_len

class fakedata(torch.utils.data.Dataset):
    def __init__(self, acgan, max_len=5000):
        super(fakedata, self).__init__()
        self.__noise_array = [self.__GenerateNoise() for i in range(max_len)]
        self.__max_len = max_len
        self.__acgan = acgan
    
    def __GenerateNoise(self):
        noise = torch.from_numpy(np.random.normal(0, 1, 110))
        return noise.cuda() if torch.cuda.is_available() else noise

    def __getitem(self, index):
        im = self.__acgan.G(self.__noise_array[index])
        _, label = self.__acgan.D(self.__acgan.G(im).detach())
        return im, label.data.tolist()
        
    def __len__(self):
        return self.__max_len