import torch as t
from torch import nn
from torchvision.models import vgg16


def caffevgg16(vgg16path):
    # the 30th layer of features is relu of conv5_3
    model = vgg16(pretrained=False)
    caffevgg = t.load(vgg16path)
    caffekeys = list(caffevgg.keys())
    i = 0
    for name, para in model.state_dict().items():
        para.copy_(caffevgg[caffekeys[i]])
        i += 1
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    del classifier[6]
    use_drop = True
    if not use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier
