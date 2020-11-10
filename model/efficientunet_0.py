from collections import OrderedDict
import sys
sys.path.insert(0,'..')
from model.layers import *
from model.efficientnet import EfficientNet
import pdb
import torchvision

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res  = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class ResUnet(nn.Module):
    def __init__(self,out_channels=1):
        super().__init__()
        res = torchvision.models.resnet50(pretrained=True)
        self.head = nn.Sequential(*list(res.children())[:4])
        self.blocks = nn.ModuleList([
            list(res.children())[4],
            list(res.children())[5],
            list(res.children())[6],
            list(res.children())[7]
        ])
        self.in_channels = [256,512,1024,2048]

        self.conv1_1 = nn.ModuleList([])
        for i in range(4):
            self.conv1_1.append(
                RRB(self.in_channels[i]*2,self.in_channels[i])
            )
        self.upconvs = nn.ModuleList([
            up_conv(self.in_channels[-1], self.in_channels[-2]),
            up_conv(self.in_channels[-2], self.in_channels[-3]),
            up_conv(self.in_channels[-3], self.in_channels[-4]),
            up_conv(self.in_channels[-4], self.in_channels[-4])
        ])

        self.double_convs = nn.ModuleList([
            double_conv(self.in_channels[-2] * 2, self.in_channels[-2]),
            double_conv(self.in_channels[-3] * 2, self.in_channels[-3]),
            double_conv(self.in_channels[-4] * 2, self.in_channels[-4]),
        ])
        # self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.final_up_conv = nn.Sequential(
            up_conv(self.in_channels[0], self.in_channels[0]),
            up_conv(self.in_channels[0], self.in_channels[0])
        )
        self.final_conv = nn.Conv2d(self.in_channels[0], out_channels, kernel_size=1)

    def normal_block(self, block, x):
        return block(x)

    def concat_block(self, block, conv1_1, small, big):
        big_out = block(big)
        small_out = block(small)
        big_crop = self.crop(big_out, small_out.size())
        small_out = torch.cat([small_out, big_crop], dim=1)
        small_out = conv1_1(small_out)
        # small_out = small_out + big_crop
        return small_out, big_out

    def encode_block(self,x, big):
        results = []
        for i in range(4):
            x, big = self.concat_block(self.blocks[i], self.conv1_1[i], x, big)
            results.append(x)
        return x, results

    def crop(self, big, size):
        _, _, x, y = big.shape
        _, _, cropx, cropy = size
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return big[:, :, startx:startx + cropx, starty:starty + cropy]

    def forward(self, x, big):

        x = self.head(x)
        big = self.head(big)
        x, results = self.encode_block(x, big)

        x = self.upconvs[0](x)
        x = torch.cat([x,results[-2]],dim=1)
        x = self.double_convs[0](x)

        x = self.upconvs[1](x)
        x = torch.cat([x,results[-3]],dim=1)
        x = self.double_convs[1](x)

        x = self.upconvs[2](x)
        x = torch.cat([x,results[-4]],dim=1)
        x = self.double_convs[2](x)

        # x = self.upsample(x)
        x = self.final_up_conv(x)
        x = self.final_conv(x)
        return x


class EfficientUnet_m(nn.Module):
    def __init__(self, encoder, out_channels=1, concat_input=False):
        super().__init__()

        self.encoder = encoder
        self.in_encoder = nn.Sequential(*list(encoder.children())[:3])
        self.out_encoder = nn.Sequential(*list(encoder.children())[-3:])
        middle_block = nn.Sequential(*list(encoder.children())[3])
        self.blocks = nn.ModuleList([])
        for i in range(7):
            self.blocks.append(nn.Sequential(*list(middle_block.children())[self.block_ix[i]:self.block_ix[i+1]]))
        self.concat_input = concat_input

        self.conv1_1 = nn.ModuleList([])
        for i in range(4):
            self.conv1_1.append(
                RRB(self.in_channels[i]*2,self.in_channels[i])
            )

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
            self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(self.size[5]*2, out_channels, kernel_size=1)
    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def in_channels(self):
        in_channels_dict = {'efficientnet-b0': [16,24,40,80,112,192,320], 'efficientnet-b3': [24,32,48,96,136,232,384]}
        return in_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    @property
    def block_ix(self):
        size_dict = {'efficientnet-b0':[0,1,3,5,8,11,15,16],'efficientnet-b3':[0,2,5,8,13,18,24,26]}
        #{'efficientnet-b0': [0,2,4,7],'efficientnet-b3':[1,4,7,12]}
        return size_dict[self.encoder.name]

    def concat_block(self,block, conv1_1, small, big):
        big_out = block(big)
        small_out = block(small)
        big_crop = self.crop(big_out,small_out.size())
        small_out = torch.cat([small_out,big_crop],dim=1)
        small_out = conv1_1(small_out)
        # small_out = small_out + big_crop
        return small_out, big_out

    def normal_block(self,block,x):
        return block(x)

    def crop(self, big, size):
        _,_,x,y= big.shape
        _,_,cropx, cropy = size
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return big[:,:,startx:startx + cropx,starty:starty + cropy]

    def forward(self, x, big):

        input_ = x
        x = self.in_encoder(x)
        big = self.in_encoder(big)
        results = []

        x, big = self.concat_block(self.blocks[0], self.conv1_1[0], x, big)
        results.append(x)
        x, big = self.concat_block(self.blocks[1], self.conv1_1[1], x, big)
        results.append(x)
        x, big = self.concat_block(self.blocks[2], self.conv1_1[2], x, big)
        results.append(x)
        x, big = self.concat_block(self.blocks[3], self.conv1_1[3], x, big)
        results.append(x)
        # x = self.in_encoder(x)
        # results = []
        # x = self.normal_block(self.blocks[0], x)
        # results.append(x)
        # x = self.normal_block(self.blocks[1], x)
        # results.append(x)
        # x = self.normal_block(self.blocks[2], x)
        # results.append(x)
        # x = self.normal_block(self.blocks[3], x)
        # results.append(x)

        x = self.normal_block(self.blocks[4], x)
        x = self.normal_block(self.blocks[5], x)
        x = self.normal_block(self.blocks[6], x)

        x = self.out_encoder(x)

        x = self.up_conv1(x)
        x = torch.cat([x, results[-1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, results[-2]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, results[-3]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, results[-4]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_.to(x.device)], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)
        return x


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=1, concat_input=False):
        super().__init__()

        self.encoder = encoder
        self.in_encoder = nn.Sequential(*list(encoder.children())[:3])
        self.out_encoder = nn.Sequential(*list(encoder.children())[-3:])
        middle_block = nn.Sequential(*list(encoder.children())[3])
        self.blocks = nn.ModuleList([])
        for i in range(7):
            self.blocks.append(nn.Sequential(*list(middle_block.children())[self.block_ix[i]:self.block_ix[i+1]]))
        self.concat_input = concat_input

        self.conv1_1 = nn.ModuleList([])
        for i in range(4):
            self.conv1_1.append(
                RRB(self.in_channels[i]*2,self.in_channels[i])
            )

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
            self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(self.size[5]*2, out_channels, kernel_size=1)
    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def in_channels(self):
        in_channels_dict = {'efficientnet-b0': [16,24,40,80,112,192,320], 'efficientnet-b3': [24,32,48,96,136,232,384]}
        return in_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    @property
    def block_ix(self):
        size_dict = {'efficientnet-b0':[0,1,3,5,8,11,15,16],'efficientnet-b3':[0,2,5,8,13,18,24,26]}
        #{'efficientnet-b0': [0,2,4,7],'efficientnet-b3':[1,4,7,12]}
        return size_dict[self.encoder.name]

    def normal_block(self,block,x):
        return block(x)

    def forward(self, x):

        input_ = x
        x = self.in_encoder(x)
        results = []
        x = self.normal_block(self.blocks[0], x)
        results.append(x)
        x = self.normal_block(self.blocks[1], x)
        results.append(x)
        x = self.normal_block(self.blocks[2], x)
        results.append(x)
        x = self.normal_block(self.blocks[3], x)
        results.append(x)

        x = self.normal_block(self.blocks[4], x)
        x = self.normal_block(self.blocks[5], x)
        x = self.normal_block(self.blocks[6], x)

        x = self.out_encoder(x)

        x = self.up_conv1(x)
        x = torch.cat([x, results[-1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, results[-2]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, results[-3]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, results[-4]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_.to(x.device)], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)
        return x




def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

if __name__ == '__main__':
    inp = torch.randn((2, 3, 96, 96))
    big = torch.randn((2,3,256,256))
    model = ResUnet(out_channels=1)
    temp = model(inp,big)
    print(temp.shape)
