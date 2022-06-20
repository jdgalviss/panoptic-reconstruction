import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

style_layers_default = [8]

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(b * c * d) #G.div(a * b * c * d)



class Model(nn.Module):
    def __init__(self, vgg, normalization_mean, normalization_std,
                    style_layers=style_layers_default,
                    to_cuda=True):
        super(Model, self).__init__()
        self.style_layers = np.sort(style_layers)
        max_conv_layer = self.style_layers[-1]
    
        # normalization module
        # normalization = Normalization(normalization_mean, normalization_std)
        # if to_cuda:
        #     normalization.cuda()

        self.modules = []
    
        i = 0  # increment every time we see a conv
        self.indices = set()
        for layer in vgg.features.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
                self.indices.add(len(self.modules))
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    
            self.modules.append(layer)
    
            if i == max_conv_layer:
                break
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = []
        for i in range(len(self.modules)):
            x = self.modules[i](x)
            if i in self.indices:
                outputs.append(x)
        return outputs

