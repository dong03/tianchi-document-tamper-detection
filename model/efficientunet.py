from collections import OrderedDict
import sys
sys.path.insert(0,'..')
from model.layers import *
from model.efficientnet import EfficientNet
import pdb
import torchvision

__all__ = ['EfficientUnet', 'get_efficientunet_d_b0', 'get_efficientunet_d_b3']


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


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x1,x2): # high, low
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


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


        self.rrb_d = nn.ModuleList([
            RRB(self.in_channels[0], self.in_channels[0]),
            RRB(self.in_channels[1], self.in_channels[1]),
            RRB(self.in_channels[2], self.in_channels[2]),
            RRB(self.in_channels[3], self.in_channels[3]),
        ])
        self.cab = nn.ModuleList([
            CAB(self.in_channels[0] * 2, self.in_channels[0]),
            CAB(self.in_channels[1] * 2, self.in_channels[1]),
            CAB(self.in_channels[2] * 2, self.in_channels[2]),
            CAB(self.in_channels[3] * 2, self.in_channels[3]),
        ])
        self.rrb_u = nn.ModuleList([
            RRB(self.in_channels[0], self.in_channels[0]),
            RRB(self.in_channels[0], self.in_channels[0]),
            RRB(self.in_channels[1], self.in_channels[1]),
            RRB(self.in_channels[2], self.in_channels[2]),
        ])

        self.upconvs = nn.ModuleList([
            up_conv(self.in_channels[0], self.in_channels[0]),
            up_conv(self.in_channels[1], self.in_channels[0]),
            up_conv(self.in_channels[2], self.in_channels[1]),
            up_conv(self.in_channels[3], self.in_channels[2])
        ])


        # self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.final_up_conv = up_conv(self.in_channels[0], self.in_channels[0])
        self.final_conv = nn.Conv2d(self.in_channels[0], out_channels, kernel_size=1)

    def normal_block(self, block, x):
        return block(x)

    def encode_block(self,x):
        results = []
        for i in range(4):
            x = self.normal_block(self.blocks[i], x)
            results.append(x)
        return x, results

    def forward(self, x):

        # input_ = x
        x = self.head(x)
        x, results = self.encode_block(x)

        global_feature = self.avg_pool(x)
        deeper = nn.Upsample(size=results[-1].size()[2:], mode="nearest")(global_feature)

        for ix in range(1,5):
            shallow = self.rrb_d[-ix](results[-ix])
            deeper = self.cab[-ix](shallow,deeper)
            deeper = self.upconvs[-ix](deeper)
            deeper = self.rrb_u[-ix](deeper)

        deeper = self.final_up_conv(deeper)
        deeper = self.final_conv(deeper)
        return deeper


class EfficientUnet_dup(nn.Module):
    def __init__(self, encoder, out_channels=1):
        super().__init__()

        self.encoder = encoder

        self.out_conv = nn.Conv2d(self.n_channels,self.size[3],kernel_size=1)
        self.final_conv = nn.Conv2d(self.size[0], out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=False)

        self.rrb_d = nn.ModuleList([
            RRB(self.size[0], self.size[0]),
            RRB(self.size[1], self.size[1]),
            RRB(self.size[2], self.size[2]),
            RRB(self.size[3], self.size[3]),
        ])
        self.cab = nn.ModuleList([
            CAB(self.size[0] * 2, self.size[0]),
            CAB(self.size[1] * 2, self.size[1]),
            CAB(self.size[2] * 2, self.size[2]),
            CAB(self.size[3] * 2, self.size[3]),
        ])
        self.rrb_u = nn.ModuleList([
            RRB(self.size[0], self.size[0]),
            RRB(self.size[1], self.size[0]),
            RRB(self.size[2], self.size[1]),
            RRB(self.size[3], self.size[2]),
        ])

        self.upconvs = nn.ModuleList([
            up_conv(self.size[0], self.size[0]),
            up_conv(self.size[1], self.size[1]),
            up_conv(self.size[2], self.size[2]),
            up_conv(self.size[3], self.size[3])
        ])
    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [16,24,40,80,112,192,320], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [24,32,48,96,136,232,384],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    @property
    def block_ix(self):
        size_dict = {'efficientnet-b0': [0,2,4,7],'efficientnet-b3':[1,4,7,12]}
        return size_dict[self.encoder.name]

    def run_encoder(self,x):
        in_encoder = nn.Sequential(*list(self.encoder.children())[:3])
        out_encoder = nn.Sequential(*list(self.encoder.children())[-3:])
        middle_block = nn.Sequential(*list(self.encoder.children())[3])
        models = [nn.Sequential(list(middle_block.children())[i]) for i in range(len(middle_block))]
        x = in_encoder(x)
        results = []
        for ix, each_model in enumerate(models):
            x = each_model(x)
            if ix in self.block_ix:
                results.append(x)
        x = out_encoder(x)
        return x, results

    def forward(self, x):
        x, results = self.run_encoder(x)
        global_feature = self.global_pool(x)
        deeper = nn.Upsample(size=results[-1].size()[2:], mode="nearest")(global_feature)
        deeper = self.out_conv(deeper)
        for ix in range(1,5):
            shallow = self.rrb_d[-ix](results[-ix])
            deeper = self.cab[-ix](shallow,deeper)
            deeper = self.upconvs[-ix](deeper)
            deeper = self.rrb_u[-ix](deeper)
        deeper = self.final_conv(deeper)
        return deeper


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=1):
        super().__init__()

        self.encoder = encoder

        self.out_conv = nn.Conv2d(self.n_channels,self.size[6],kernel_size=1)
        # self.final_conv = nn.Conv2d(self.size[0], out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=False)

        self.rrb_d = nn.ModuleList([
            RRB(self.size[3], self.size[3]),
            RRB(self.size[4], self.size[4]),
            RRB(self.size[5], self.size[5]),
            RRB(self.size[6], self.size[6]),
        ])

        self.cab = nn.ModuleList([
            CAB(self.size[3] * 2, self.size[3]),
            CAB(self.size[4] * 2, self.size[4]),
            CAB(self.size[5] * 2, self.size[5]),
            CAB(self.size[6] * 2, self.size[6]),
        ])

        self.rrb_u = nn.ModuleList([
            RRB(self.size[3], self.size[2]),
            RRB(self.size[4], self.size[3]),
            RRB(self.size[5], self.size[4]),
            RRB(self.size[6], self.size[5]),
        ])

        self.upconvs = nn.ModuleList([
            up_conv(self.size[3], self.size[3]),
            up_conv_samesize(self.size[4], self.size[4]),  # same size
            up_conv(self.size[5], self.size[5]),
            up_conv_samesize(self.size[6], self.size[6]),
        ])

        self.double_conv_u = nn.ModuleList([
            double_conv(self.size[3], self.size[2]),
            double_conv(self.size[4], self.size[3]),
            double_conv(self.size[5], self.size[4]),
            double_conv(self.size[6], self.size[5]),
        ])

        self.final_upconvs = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            up_conv(self.size[2], self.size[1]),
            double_conv(self.size[1], self.size[1]),

            up_conv(self.size[1], self.size[0]),
            double_conv(self.size[0], self.size[0]),

            up_conv(self.size[0], self.size[0]),

            nn.Conv2d(self.size[0], out_channels, kernel_size=1)
            # double_conv(self.size[0], out_channels),
        )

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [16,24,40,80,112,192,320], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [24,32,48,96,136,232,384],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    @property
    def block_ix(self):
        # size_dict = {'efficientnet-b0': [0,2,4,7],'efficientnet-b3':[1,4,7,12]}
        # size_dict = {'efficientnet-b0': [2, 4, 7, 10], 'efficientnet-b3': [4, 7, 12, 17]}
        size_dict = {'efficientnet-b0': [7, 10, 14, 15], 'efficientnet-b3': [12, 17, 23, 25]}
        return size_dict[self.encoder.name]

    def run_encoder(self,x):
        in_encoder = nn.Sequential(*list(self.encoder.children())[:3])
        out_encoder = nn.Sequential(*list(self.encoder.children())[-3:])
        middle_block = nn.Sequential(*list(self.encoder.children())[3])
        models = [nn.Sequential(list(middle_block.children())[i]) for i in range(len(middle_block))]
        x = in_encoder(x)
        results = []
        for ix, each_model in enumerate(models):
            x = each_model(x)
            if ix in self.block_ix:
                results.append(x)
        x = out_encoder(x)
        return x, results

    def forward(self, x):
        x, results = self.run_encoder(x)
        global_feature = self.global_pool(x)
        deeper = nn.Upsample(size=results[-1].size()[2:], mode="nearest")(global_feature)
        deeper = self.out_conv(deeper)
        for ix in range(1,5):
            shallow = self.rrb_d[-ix](results[-ix])
            deeper = self.cab[-ix](shallow, deeper)
            deeper = self.upconvs[-ix](deeper)
            deeper = self.rrb_u[-ix](deeper)
            # deeper = self.double_conv_u[-ix](deeper)
        deeper = self.final_upconvs(deeper)
        return deeper


def get_efficientunet_d_b0(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_b1(out_channels=2,  pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_b2(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_d_b3(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_b4(out_channels=2,  pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_b5(out_channels=2,  pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_b6(out_channels=2, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model


def get_efficientunet_b7(out_channels=2,  pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels)
    return model

if __name__ == '__main__':
    model = get_efficientunet_d_b3(out_channels=1)
    print(model.block_ix)
    #model = ResUnet(out_channels=1)
    inp = torch.randn((2,3,256,256))
    out = model(inp)
    print(out.shape)