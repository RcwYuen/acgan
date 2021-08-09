import torch, torchvision.models as models, numpy
from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class EvalResNet(nn.Module):
    def __init__(self, types):
        super(EvalResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.lr = nn.Linear(1000, types)

    def forward(self, x):
        return self.lr(self.resnet(x))
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_size = 32//4
        self.linear = nn.Linear(110, 384*self.init_size**2, bias=True)

        self.conv_trans1 = nn.ConvTranspose2d(384, 192, 5, 2)

        self.bn1 = nn.BatchNorm2d(192)
        self.conv_trans2 = nn.ConvTranspose2d(192, 96, 5, 2)

        self.bn2 = nn.BatchNorm2d(96)
        self.conv_trans3 = nn.ConvTranspose2d(96, 3, 5, 2)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, x.shape[1])
        x = self.relu(self.linear(x))
        x = x.view(x.shape[0], 384, self.init_size, self.init_size)
        print (x.shape)
        x = self.relu(self.bn1(self.conv_trans1(x)))
        print (x.shape)
        x = self.relu(self.bn2(self.conv_trans2(x)))
        print (x.shape)
        return self.tanh(self.conv_trans3(x))

class Discriminator(nn.Module):
    def __init__(self, types):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, kernal_size, stride)
            
            nn.Conv2d(3, 16, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),

            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),

            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),

            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
        )

        self.lr1 = nn.Linear(98304, 1, bias=True)
        self.lr2 = nn.Linear(98304, types, bias=True)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.apply(weights_init)

    def forward(self, x):
        x = self.net(x)
        adv, typeclass = self.lr1(x), self.lr2(x)
        return self.sig(adv), self.softmax(typeclass)
'''

class Generator(nn.Module):
    def __init__(self, nz=110, output_chan=3):
        super(Generator, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, output_chan, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, input):
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        return tconv5


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, input_chan=3):
        super(Discriminator, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_chan, 16, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 4*4*512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        return self.sigmoid(fc_dis).view(-1, 1).squeeze(1), self.softmax(fc_aux)

def createGenerator(nz=110, output_chan=3):
    return Generator(nz=nz, output_chan=output_chan).cuda() if torch.cuda.is_available() else Generator(nz=nz, output_chan=output_chan)


def createDiscriminator(ntype, input_chan=3):
    return Discriminator(ntype, input_chan=input_chan).cuda() if torch.cuda.is_available() else Discriminator(ntype, input_chan=input_chan)


def createResNet(ntype):
    return EvalResNet(ntype).cuda() if torch.cuda.is_available() else EvalResNet(ntype)