from torch import nn
from torch.autograd import Variable
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import torch, torchvision.models as models, torchvision, torchvision.utils as vutils
import numpy as np, matplotlib.pyplot as plt, random, time
import models, dataset

#https://github.com/clvrai/ACGAN-PyTorch/blob/master/main.py

class ACGAN:
    def __init__(self, batch_size=32, epoch=500, n_type=10, max_len=10000, d_name='cifar-10', nz=70):
        self.__type = n_type
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__channels = 3
        self.__nz = nz
        self.__data_max_len = max_len + batch_size - (max_len % batch_size)
        self.__d_name = d_name
        self.__truth = torch.ones(self.__batch_size).cuda() if torch.cuda.is_available() else torch.ones(self.__batch_size)
        self.__fake = torch.zeros(self.__batch_size).cuda() if torch.cuda.is_available() else torch.zeros(self.__batch_size)
        self.D = models.createDiscriminator(self.__type, input_chan=self.__channels)
        self.G = models.createGenerator(nz=self.__nz, output_chan=self.__channels)
        self.__Goptm = torch.optim.Adam(self.G.parameters(), lr=1e1, betas=(0.5, 0.999))
        self.__Doptm = torch.optim.Adam(self.D.parameters(), lr=1e-6, betas=(0.5, 0.999))#, weight_decay=-5e-2)
        self.__g_criterion = nn.NLLLoss()
        self.__d_criterion = nn.BCELoss()
        data = dataset.realdata(max_len=self.__data_max_len, d_name=self.__d_name)
        self.__dataset = torch.utils.data.DataLoader(data, batch_size=self.__batch_size, shuffle=True, drop_last=True)

    def __GenerateNoise(self, raw_label):
        label = np.zeros((self.__batch_size, self.__type))
        label[np.arange(self.__batch_size), raw_label] = 1
        noise = np.random.normal(0, 1, (self.__batch_size, self.__nz))
        noise[np.arange(self.__batch_size), :self.__type] = label[np.arange(self.__batch_size)]
        return torch.from_numpy(noise).cuda() if torch.cuda.is_available() else torch.from_numpy(noise)

    def __GenerateLabel(self, raw_label):
        label = np.zeros((self.__batch_size, self.__type))
        label[np.arange(self.__batch_size), raw_label] = 1
        return torch.from_numpy(label).cuda() if torch.cuda.is_available() else torch.from_numpy(label)

    def __bincalcAcc(self, output, label):
        pred_label = [1 if i > 0.5 else 0 for i in output]
        num_correct = sum([1 if label[i] == pred_label[i] else 0 for i in range(output.shape[0])])
        return num_correct / output.shape[0]

    def __multcalcAcc(self, output, label):
        pred_label = [((i==max(i)).nonzero())[0] for i in output]
        num_correct = sum([1 if label[i] == pred_label[i] else 0 for i in range(output.shape[0])])
        return num_correct / output.shape[0]

    def loadModel(self):
        self.G = torch.load("./G.pth")
        self.D = torch.load("./D.pth")

    def train(self):
        fakeLoss, realLoss, genLoss = [], [], []
        fakeAcc, realAcc, classAcc = [], [], []
        for epoch in range(self.__epoch):
            for batchNo, data in enumerate(self.__dataset, 0):
                self.D.zero_grad()
                # Training the Discriminator, Truth First
                image, label = data
                image = image.cuda() if torch.cuda.is_available() else image
                label = label.cuda() if torch.cuda.is_available() else label
                real_output, real_types = self.D(image)
                real_bce = self.__d_criterion(real_output, self.__truth)
                real_cel = self.__g_criterion(real_types, label)
                real_loss = real_cel + real_bce
                real_loss.backward()
                
                # Training the Discriminator, Then Fake
                raw_label = np.random.randint(0,self.__type,(self.__batch_size))
                fake_label = self.__GenerateLabel(raw_label)
                noise = self.__GenerateNoise(raw_label)
                noise = noise.cuda() if torch.cuda.is_available() else noise
                raw_label = torch.from_numpy(raw_label).type(torch.LongTensor).cuda() if torch.cuda.is_available() else torch.from_numpy(raw_label).type(torch.LongTensor)
                fake_im = self.G(noise.float())
                fake_output, fake_types = self.D(fake_im.detach())
                fake_bce = self.__d_criterion(fake_output, self.__fake)
                fake_cel = self.__g_criterion(fake_types, raw_label)
                fake_loss = fake_cel + fake_bce
                fake_loss.backward()
                D_G_z1 = fake_output.data.mean()
                self.__Doptm.step()
                
                fakeDis_acc = self.__bincalcAcc(fake_output, self.__fake)
                realDis_acc = self.__bincalcAcc(real_output, self.__truth)
                classification_acc = self.__multcalcAcc(real_types, label)

                self.G.zero_grad()
                #Training the Generator
                fake_output, fake_types = self.D(fake_im.detach())
                fake_bce = self.__d_criterion(fake_output, self.__truth)
                fake_cel = self.__g_criterion(fake_types, raw_label)
                Gfake_loss = fake_bce + fake_cel
                Gfake_loss.backward()
                D_G_z1 = fake_output.data.mean()
                self.__Goptm.step()

                print("[%3d/%3d][%3d/%3d]: FakeLoss_D: %.4f (%.4f) TruthLoss_D: %.4f (%.4f) Classify_Acc: %.4f Loss_G: %4f" % (epoch+1, self.__epoch, batchNo+1, len(self.__dataset), fake_loss.data.tolist(), fakeDis_acc, real_loss.data.tolist(), realDis_acc, classification_acc, Gfake_loss.data.tolist()))
                fakeLoss.append(fake_loss.data.tolist())
                realLoss.append(real_loss.data.tolist())
                genLoss.append(Gfake_loss.data.tolist())
                fakeAcc.append(fakeDis_acc)
                realAcc.append(realDis_acc)
                classAcc.append(classification_acc)
                
            raw_label = np.random.randint(0,self.__type,(self.__batch_size))
            fake_label = self.__GenerateLabel(raw_label)
            noise = self.__GenerateNoise(raw_label)
            vutils.save_image(self.G(noise.float()).data,'./fake_samples_epoch_%03d.png' % (epoch))
        
        plt.plot(fakeLoss)
        plt.plot(realLoss)
        plt.plot(genLoss)
        plt.show()
        plt.plot(fakeAcc)
        plt.plot(realAcc)
        plt.plot(classAcc)
        plt.show()
        torch.save(self.G, "./G.pth")
        torch.save(self.D, "./D.pth")

    def genRandon(self):
        im = self.G(self.__GenerateNoise)
        _, type = self.D(im.detach())
        return im, type

class ResNet18:
    def __init__(self, acgan, batch_size=16, epoch=50, type=10, max_len=5000, d_name='cifar-10'):
        self.acgan = acgan
        self.__type = type
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__d_name = d_name
        self.__data_max_len = max_len + batch_size - (max_len % batch_size)
        self.resnet = models.createResNet(self.__type)
        self.__optim = torch.optim.SGD(self.resnet.parameters(), lr=1e-3)
        self.__criterion = nn.CrossEntropyLoss()
        realdata = dataset.realdata(max_len=self.__data_max_len, d_name=self.__d_name)
        fakedata = dataset.fakedata(acgan, max_len=self.__data_max_len)
        self.__realdataset = torch.utils.data.DataLoader(realdata, batch_size=self.__batch_size, shuffle=True)
        self.__fakedataset = torch.utils.data.DataLoader(fakedata, batch_size=self.__batch_size, shuffle=True)

    def train(self):
        pass