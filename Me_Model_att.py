import torch
from torch import nn
import torch.nn.functional as F
# 这里是双分支

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, k_size=3):
        super(CBAM, self).__init__()

        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1
                                , kernel_size=k_size, stride=1, padding=k_size // 2)

    def channel_attention(self, x):
        # 使用自适应池化缩减map的大小，保持通道不变
        avg_out = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        max_out = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid1(avg_out + max_out)

    def spatial_attention(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid2(self.conv2d(out))
        return out

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# 搭一个Inception模块,要么用Inception,要么用ROINet
class Inception(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        # 4个分支在dim=1即沿着channel(张量是0batch,1channel,2weight,3height)上进行concatenate。6+6+6+6=24(输出通道数)
        return torch.cat(outputs, dim=1)


# 搭一个ROINet,要么用ROINet,要么用Inception
class MEROINet(nn.Module):
    def __init__(self, in_channels=3):
        super(MEROINet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 12, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 24, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MyMEROI(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MyMEROI, self).__init__()
        self.MEROINet1 = MEROINet()
        self.MEROINet2 = MEROINet()
        self.cbam = CBAM(24)

        self.fc = torch.nn.Sequential(
            # 仅一层Linear的情况：
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(in_features=2*56 * 56 * 24, out_features=num_classes),
            # 28->2次池化->12*7*7； 224->56*56*24, 且因为后面要concat所以*2

            # 若是两层及以上Linear的情况：
            torch.nn.Linear(in_features=2 * 56 * 56 * 24, out_features=1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x1, x2):
        out_x1 = self.MEROINet1(x1)
        out_x2 = self.MEROINet2(x2)
        out_x1 = self.cbam(out_x1)

        out_x1 = out_x1.reshape(out_x1.shape[0], -1)  # flatten 变成全连接层的输入
        out_x2 = out_x2.reshape(out_x2.shape[0], -1)

        x = torch.cat((out_x1, out_x2), 1)
        # print(x.size())
        x = self.fc(x)

        return x


class MEROIInception(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MEROIInception, self).__init__()
        self.Inception1_1 = Inception(in_channels=3, out_channels=6)
        self.Inception1_2 = Inception(in_channels=24, out_channels=12)
        self.Inception2_1 = Inception(in_channels=3, out_channels=6)
        self.Inception2_2 = Inception(in_channels=24, out_channels=12)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cbam = CBAM(48)  # 单层24，双层48

        # 把48*28*28的Inception输出变成24*28*28
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(48, 24, kernel_size=1),
        #     torch.nn.BatchNorm2d(24),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(kernel_size=2),
        # )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=7 * 7 * 48 * 2, out_features=1024),  # 单层14*14*24，双层7*7*48
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x1, x2):
        out_x1 = self.Inception1_1(x1)
        out_x1 = self.maxpool(out_x1)
        out_x1 = self.Inception1_2(out_x1)
        out_x1 = self.maxpool(out_x1)
        out_x1 = self.cbam(out_x1)

        out_x2 = self.Inception2_1(x2)
        out_x2 = self.maxpool(out_x2)
        out_x2 = self.Inception2_2(out_x2)
        out_x2 = self.maxpool(out_x2)
        out_x2 = self.cbam(out_x2)  # 如果这里也加cbam呢，发现更低只有65%

        x = torch.cat((out_x1, out_x2), 1)
        # x = self.conv(out_x)
        x = x.reshape(x.shape[0], -1)  # flatten 变成全连接层的输入

        x1 = out_x1.reshape(x1.shape[0], -1)  # 对比需要展平
        x2 = out_x2.reshape(x2.shape[0], -1)   #若有对比损失，需要展平x1，x2，并return
        # print(x.size())
        x = self.fc(x)

        # return x
        return x1, x2, x

