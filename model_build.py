from intern_image import InternImage
from view_transformer import LSSViewTransformer, LSSViewTransformerBEVDepth
pretrained = 'backbone.pth' 
intern_image_model = InternImage(
        core_op='DCNv3',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg= pretrained
)
#transformer configurations ---------------------------------------
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (640, 1600),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-51.2, 51.2, 0.4],                                                # 1
    'y': [-51.2, 51.2, 0.4],                                                # 2
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

input_size=data_config['input_size']
in_channels=512
out_channels=128
depthnet_cfg=dict(use_dcn=False)
downsample=16

#---------------------------------------------------------------

# print(intern_image_model)        


if pretrained:
  intern_image_model.init_weights()

import torch
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn

# Set the GPU device if you have multiple GPUs
device = torch.device("cuda:0")  # Replace '0' with the GPU device ID you want to use
torch.cuda.set_device(device)

# Set the model in evaluation mode
intern_image_model.eval()

intern_image_model.to(device)

# Create some example input data (replace this with your actual data)
input_data = torch.randn(6, 3, 640, 1600).to(device)

# Perform inference
with torch.no_grad():
    output_features = intern_image_model(input_data)
    # print(output_features)
    out0 = output_features[2].cpu().numpy()
    out1 = output_features[3].cpu().numpy()
    print("InternImage_Backbone_out")
    print("------------------------")
    print("backbone_out0",out0.shape)
    print("backbone_out1",out1.shape)
# ===============================================================

view_transformer = LSSViewTransformerBEVDepth(
    grid_config=grid_config,
    input_size=input_size,
    in_channels=512,
    out_channels=128,
    accelerate=False,  # Set to True if using acceleration
    depthnet_cfg=dict(use_dcn=False),  # Adjust as needed
    downsample = 16,  # Additional depthnet configuration if needed
) 

#the preprocessor of the tensor befor passig to the transformer

# def prepare_inputs(self, inputs):
#         # split the inputs into each frame
#         B, N, _, H, W = inputs[0].shape
#         N = N // self.num_frame
#         imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
#         imgs = torch.split(imgs, 1, 2)
#         imgs = [t.squeeze(2) for t in imgs]
#         rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
#         extra = [
#             rots.view(B, self.num_frame, N, 3, 3),
#             trans.view(B, self.num_frame, N, 3),
#             intrins.view(B, self.num_frame, N, 3, 3),
#             post_rots.view(B, self.num_frame, N, 3, 3),
#             post_trans.view(B, self.num_frame, N, 3)
#         ]
#         extra = [torch.split(t, 1, 1) for t in extra]
#         extra = [[p.squeeze(1) for p in t] for t in extra]
#         rots, trans, intrins, post_rots, post_trans = extra
#         return imgs, rots, trans, intrins, post_rots, post_trans, bda



import torch

# Define dummy input tensors
B = 1  # Batch size
N = 12  # Total number of frames
num_frame = 2  # Number of frames per group
H, W = 640, 1600  # Image height and width

# Create dummy input tensors
inputs = [
    torch.randn(B, N, 3, H, W),  # Image data
    torch.randn(B, N, 3, 3),    # Rotation matrices
    torch.randn(B, N, 3),       # Translation vectors
    torch.randn(B, N, 3, 3),    # Intrinsic matrices
    torch.randn(B, N, 3, 3),    # Post-rotation matrices
    torch.randn(B, N, 3),       # Post-translation vectors
    torch.randn(B, 3, 3)        # BDA matrices
]

# Define the data preprocessing function
def preprocess_data(inputs, num_frame):
    # Place the provided preprocessing code here
    B, N, _, H, W = inputs[0].shape
    N = N // num_frame
    imgs = inputs[0].view(B, N, num_frame, 3, H, W)
    imgs = torch.split(imgs, 1, 2)
    imgs = [t.squeeze(2) for t in imgs]
    rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
    extra = [
        rots.view(B, num_frame, N, 3, 3),
        trans.view(B, num_frame, N, 3),
        intrins.view(B, num_frame, N, 3, 3),
        post_rots.view(B, num_frame, N, 3, 3),
        post_trans.view(B, num_frame, N, 3)
    ]
    extra = [torch.split(t, 1, 1) for t in extra]
    extra = [[p.squeeze(1) for p in t] for t in extra]
    rots, trans, intrins, post_rots, post_trans = extra
    return imgs, rots, trans, intrins, post_rots, post_trans, bda

# Test the preprocessing function
imgs, rots, trans, intrins, post_rots, post_trans, bda = preprocess_data(inputs, num_frame)

print("---------------------------------------")
# Print the shapes of the processed tensors
print("Image frames shapes:", [img.shape for img in imgs])
print("Rotation matrices shapes:", [rot.shape for rot in rots])
print("Translation vectors shapes:", [tr.shape for tr in trans])
print("Intrinsic matrices shapes:", [intr.shape for intr in intrins])
print("Post-rotation matrices shapes:", [post_rot.shape for post_rot in post_rots])
print("Post-translation vectors shapes:", [post_tr.shape for post_tr in post_trans])
print("BDA matrices shape:", bda.shape)
print("---------------------------------------")


# Assuming 'model' is your LSSViewTransformerBEVDepth model
with torch.no_grad():
    rot = rots[0].to(device)  # Ensure 'rot' is on the correct device
    tran = trans[0].to(device)  # Ensure 'tran' is on the correct device
    intrin = intrins[0].to(device)  # Ensure 'intrin' is on the correct device
    post_rot = post_rots[0].to(device)  # Ensure 'post_rot' is on the correct device
    post_tran = post_trans[0].to(device)  # Ensure 'post_tran' is on the correct device
    output = view_transformer.get_mlp_input(
        rot, tran, intrin, post_rot, post_tran, bda
    )
output = output.to(device)

print("---------------------------------------")
print("mlp_input:", output.shape)
print("---------------------------------------")

import numpy as np
import onnxruntime as ort
model_path = "hvdet_fuse_stage1.onnx"
session1 = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
################ONNX-DEPTHNET###########################
backbone_out0 = out0
backbone_out1 = out1
mlp_input = output.to(device)
# Prepare the input data as a dictionary
input_data = {
    'backbone_out0': backbone_out0,
    'backbone_out1': backbone_out1,
    'mlp_input': mlp_input.cpu().numpy(),  # Now it's a NumPy array
}
sess1_out_img_feat, sess1_out_depth, sess1_out_tran_feat = session1.run(
    ['img_feat', 'depth', 'tran_feat'], input_data)
print("------------------------------------------------------")    
print("stage1")
print("------------------------------------------------------")  
print(rot.device)
print(sess1_out_img_feat.shape ,sess1_out_depth.shape, sess1_out_tran_feat.shape)
sess1_out_img_feat = torch.tensor(sess1_out_img_feat).to(rot.device)
print("img_feature:",sess1_out_img_feat.shape)
sess1_out_depth = torch.tensor(sess1_out_depth).to(rot.device)
print("depth:",sess1_out_depth.shape)
sess1_out_tran_feat = torch.tensor(sess1_out_tran_feat).to(rot.device)
print("tran_feature:",sess1_out_tran_feat.shape)
print("--------------------------------------------------------------")
sess1_out_img_feat = torch.tensor(sess1_out_img_feat).to(rot.device)
sess1_out_depth = torch.tensor(sess1_out_depth).to(rot.device)
sess1_out_tran_feat = torch.tensor(sess1_out_tran_feat).to(rot.device)

inputs = [sess1_out_img_feat, rot, tran, intrin, post_rot, post_tran, bda, mlp_input]

bev_feat, _ = view_transformer.view_transform(inputs, sess1_out_depth, sess1_out_tran_feat)

bev_feat_list = []
model_path1 = "hvdet_fuse_stage1_1.onnx"
session1_1 = ort.InferenceSession(model_path1, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

sess1_1_bev_feat = session1_1.run(['out_bev_feat'],
                                                        {'bev_feat': bev_feat.cpu().numpy()})
bev_feat_list.append(sess1_1_bev_feat[0])

multi_bev_feat = np.concatenate(bev_feat_list, axis=1)
print("----------------------------------")
print("stage1_1:",multi_bev_feat.shape)
print("----------------------------------")
output_names=['bev_feat'] + [f'output_{j}' for j in range(36)]
model_path2 = "hvdet_fuse_stage2.onnx"
session2 = ort.InferenceSession(model_path2, providers=['CUDAExecutionProvider','CPUExecutionProvider'])


sess2_out = session2.run(output_names, 
                                            {
                                            'multi_bev_feat':multi_bev_feat,
                                            }) 
for i in range(len(sess2_out)):
  sess2_out[i] = torch.tensor(sess2_out[i]).cuda()
bev_feat = sess2_out[0]
pts_outs = sess2_out[1:]



