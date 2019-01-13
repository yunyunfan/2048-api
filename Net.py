import torch.nn as nn
import torch.nn.functional as F


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[1, 1])  # 16*4*4
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[1, 2])  # 64*4*3
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[2, 1])  # 128*3*3

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 2])  # 256*3*2
        # self.conv30 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1])  # 256*3*2
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[2, 1])  # 256*1*1

        self.fc1 = nn.Linear(256 * 3 * 2, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 4)

    # self.drop= nn.Dropout2d(p=0)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # x=self.drop(x)

        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

        # x = F.relu(self.conv30(x))
        # x=F.relu(self.conv4(x))

        # x=self.drop(x)

        x = x.view(-1, 256 * 3 * 2)
        # x=self.drop(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = self.fc3(x)
        # x=self.fc5(x)
        return F.log_softmax(x)




