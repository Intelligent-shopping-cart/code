import torch
import torch.nn as nn
from torchvision.models.shufflenetv2 import *
from model.fpn import *
from model.backbone.shufflenetv2 import *

class Detector(nn.Module):
    def __init__(self, classes, anchor_num, load_param):
        super(Detector, self).__init__()
        out_depth = 72
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.backbone = ShuffleNetV2(stage_out_channels, load_param)
        self.fpn = LilghtFPN(stage_out_channes[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

    def forward(self, x):
        C2, C3 = self.backbone(x)
        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(C2, C3)
        
        out_reg_2 = self.output_reg_layers(reg_2)
        out_obj_2 = self.output_obj_layers(obj_2)
        out_cls_2 = self.output_cls_layers(cls_2)

        out_reg_3 = self.output_reg_layers(reg_3)
        out_obj_3 = self.output_obj_layers(obj_3)
        out_cls_3 = self.output_cls_layers(cls_3)
        if not torch.onnx.is_in_onnx_export():
            return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3
        else:
            # for out in (out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3):
            #     print(out.shape)

            c = out_reg_2.shape[1]
            out_reg_2 = out_reg_2.permute(0, 2, 3, 1).view(-1, c)
            c = out_obj_2.shape[1]
            out_obj_2 = out_obj_2.permute(0, 2, 3, 1).view(-1, c)
            c = out_cls_2.shape[1]
            out_cls_2 = out_cls_2.permute(0, 2, 3, 1).view(-1, c)
            out_reg_2 = torch.sigmoid(out_reg_2)
            out_obj_2 = torch.sigmoid(out_obj_2)
            out_cls_2 = F.softmax(out_cls_2, dim=1)
            out2 = torch.cat((out_reg_2, out_obj_2, out_cls_2), dim=1)

            c = out_reg_3.shape[1]
            out_reg_3 = out_reg_3.permute(0, 2, 3, 1).view(-1, c)
            c = out_obj_3.shape[1]
            out_obj_3 = out_obj_3.permute(0, 2, 3, 1).view(-1, c)
            c = out_cls_3.shape[1]
            out_cls_3 = out_cls_3.permute(0, 2, 3, 1).view(-1, c)
            out_reg_3 = torch.sigmoid(out_reg_3)
            out_obj_3 = torch.sigmoid(out_obj_3)
            out_cls_3 = F.softmax(out_cls_3, dim=1)
            out3 = torch.cat((out_reg_3, out_obj_3, out_cls_3), dim=1)
            return torch.cat((out2, out3), dim=0)

if __name__ == "__main__":
    model = Detector(80, 3, False)
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "test.onnx",               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization
    


