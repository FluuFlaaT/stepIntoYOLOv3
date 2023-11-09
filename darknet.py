import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

from utils import *

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parseCfg(cfgfile):

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def createModules(blocks):
    netInfo = blocks[0]
    moduleList = nn.ModuleList()
    prevFilters = 3                # initial filter is 3 due to the image's 3 rgb channel
    outputFilters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if(x["type"] == "convolutional"):
            activation = x['activation']
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size -1) // 2
            else :
                pad = 0

            conv = nn.Conv2d(prevFilters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index) , conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, True)
                module.add_module("leaky_{0}".format(index), activ)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',') # multiple vars may available
            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = outputFilters[index + start] + outputFilters[index + end]
            else:
                filters = outputFilters[index + start]

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        moduleList.append(module)
        prevFilters = filters
        outputFilters.append(filters)

    return (netInfo, moduleList)

class Darknet(nn.Module):
    def __init__(self, cfgFile):
        super(Darknet, self).__init__()
        self.blocks = parseCfg(cfgFile)
        self.netInfo, self.moduleList = createModules(self.blocks)

    def forward(self, x, ifCUDA):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            moduleType = (module["type"])

            if moduleType == 'convolutional' or moduleType == 'upsample':
                x = self.moduleList[i](x)
                outputs[i] = x

            elif moduleType == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

                outputs[i] = x

            elif moduleType == 'shortcut':
                from_ = int(modules[i]['from'])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif moduleType == 'yolo':
                anchors = self.moduleList[i][0].anchors  # 6,7,8 like array
                
                inpImgDim = int(self.netInfo['height'])

                num_classes = int(modules[i]["classes"])

                x = x.data
                x = predictTransform(x, inpImgDim, anchors, num_classes, False)

                if not write:
                    detections = x
                    write = 1
                    
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]
            
        try:
            return detections
        except:
            return 0

    def loadWeight(self, weightfile):
        fp = open(weightfile, 'rb')

        header = np.fromfile(fp, dtype = np.int32, count = 5)

        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.moduleList)):
            moduleType = self.blocks[i + 1]["type"]

            if(moduleType == 'convolutional'):
                model = self.moduleList[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
