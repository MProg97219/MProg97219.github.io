import torch
from modeling.deeplab import *
import numpy as np
import onnx
ckpt_file = "./test.pt"

model = DeepLab(num_classes=2,
                backbone="xception",
                output_stride=16,
                sync_bn=False,
                freeze_bn=False)
model_dict = model.state_dict()
pretrained_dict = torch.load(ckpt_file, map_location="cpu")
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.state_dict().items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
model.load_state_dict(model_dict)

ckpt_file = "./test.onnx"
# image size(1, 3, 512, 512) BCHW
im = torch.ones(1, 3, 512,512).to('cpu')
torch.onnx._export(
    model,
    im,
    f=ckpt_file,
    verbose=False,
    opset_version=12,
    training=torch.onnx.TrainingMode.EVAL,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["output"],
    dynamic_axes=None
)

model_onnx = onnx.load(ckpt_file)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Simplify onnx
simplify = True
if simplify:
    import onnxsim
    print(
        f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=False,
        input_shapes=None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, ckpt_file)

print('Onnx model save as {}'.format(ckpt_file))
