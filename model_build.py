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
voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 128

multi_adj_frame_id_cfg = (1, 8, 8, 1)
num_adj = len(range(
    multi_adj_frame_id_cfg[0],
    multi_adj_frame_id_cfg[1]+multi_adj_frame_id_cfg[2]+1,
    multi_adj_frame_id_cfg[3]
))
out_size_factor = 4
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
radar_cfg = {
    'bbox_num': 100,
    'radar_fusion_type': "medium_fusion",  # in ['post_fusion', 'medium_fusion']
    'voxel_size': voxel_size,
    'out_size_factor': out_size_factor,
    'point_cloud_range': point_cloud_range,
    'grid_config': grid_config,
    'norm_bbox': True,  
    'pc_roi_method': 'pillars',
    'img_feats_bbox_dims': [1, 1, 0.5],
    'pillar_dims': [0.4, 0.4, 0.1],
    'pc_feat_name': ['pc_x', 'pc_y', 'pc_vx', 'pc_vy'],
    'hm_to_box_ratio': 1.0,
    'time_debug': False,
    'radar_head_task': [
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        ]
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

bev_feat_list = [] #this array declaration outside the loop when you are implementing throught the frames
model_path1 = "hvdet_fuse_stage1_1.onnx"
session1_1 = ort.InferenceSession(model_path1, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

sess1_1_bev_feat = session1_1.run(['out_bev_feat'],
                                                        {'bev_feat': bev_feat.cpu().numpy()})
bev_feat_list.append(sess1_1_bev_feat[0])
#remove code in this section(start to end) when you are iterating the model throgh the frames in datapipeline
#-----------------start--------------------
multi_bev_feat = np.random.rand(B, 2176, 256, 256).astype(np.float32)
#-----------------End----------------------
# multi_bev_feat = np.concatenate(bev_feat_list, axis=1)

print("----------------------------------")
print("stage1_1:",len(sess1_1_bev_feat))
print("----------------------------------")
output_names=['bev_feat'] + [f'output_{j}' for j in range(36)]
model_path2 = "hvdet_fuse_stage2.onnx"
session2 = ort.InferenceSession(model_path2, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

print("------stage2----------")
sess2_out = session2.run(output_names, 
                                            {
                                            'multi_bev_feat':multi_bev_feat,
                                            }) 
print(len(sess2_out))    
print("-----------------------")                                       
for i in range(len(sess2_out)):
  sess2_out[i] = torch.tensor(sess2_out[i]).cuda()
bev_feat = sess2_out[0]
pts_outs = sess2_out[1:]
print("------------")
print("Bev_feat:",bev_feat.shape)
def pts_head_result_deserialize(outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

from process_radar import get_valid_radar_feat 

def radar_head_result_deserialize(outs):
        outs_ = []
        keys = ['sec_reg', 'sec_rot', 'sec_vel']
        for head_id in range(len(outs) // 3):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 3 + kid]
            outs_.append(outs_head)
        return outs_
radar_pc= [
          2.799999952316284,
          3.200000047683716,
          4.199999809265137,
          5.199999809265137,
          5.800000190734863,
          5,
          6.199999809265137,
          6,
          11.399999618530273,
          10.199999809265137,
          13.199999809265137,
          13,
          12,
          14.600000381469727,
          12.600000381469727,
          14.399999618530273,
          16.200000762939453,
          18.200000762939453,
          12.600000381469727,
          19.200000762939453,
          17.799999237060547,
          23,
          21.799999237060547,
          24,
          22.399999618530273,
          17.600000381469727,
          26.399999618530273,
          23.399999618530273,
          27,
          26,
          28,
          26,
          26.200000762939453,
          29.399999618530273,
          28.399999618530273,
          30.600000381469727,
          21.600000381469727,
          25.200000762939453,
          38,
          35.20000076293945,
          38.79999923706055,
          38.79999923706055,
          37.599998474121094,
          39.599998474121094,
          37.599998474121094,
          41,
          40.400001525878906,
          40,
          42.400001525878906,
          40.79999923706055,
          42.599998474121094,
          42,
          41,
          44.599998474121094,
          40.400001525878906,
          42.400001525878906,
          45.79999923706055,
          44,
          44,
          46.599998474121094,
          47.20000076293945,
          40.79999923706055,
          46.20000076293945,
          48.20000076293945,
          31,
          51.79999923706055,
          50.79999923706055,
          54.79999923706055,
          37,
          54,
          56.79999923706055,
          55.599998474121094,
          50.599998474121094,
          58.599998474121094,
          7.199999809265137,
          7.800000190734863,
          7.800000190734863,
          7.800000190734863,
          8,
          7.800000190734863,
          7.800000190734863,
          7.800000190734863,
          7.800000190734863,
          7.800000190734863,
          8,
          8,
          8,
          7.800000190734863,
          14,
          26,
          41.400001525878906,
          40.599998474121094,
          40.400001525878906,
          44.79999923706055,
          40.400001525878906,
          40.400001525878906,
          45.400001525878906,
          39,
          39.20000076293945,
          48.20000076293945,
          52.599998474121094,
          54.20000076293945,
          47.20000076293945,
          55.400001525878906,
          56.79999923706055,
          55.79999923706055,
          55.400001525878906,
          62.599998474121094,
          65.4000015258789,
          7.400000095367432,
          218.1999969482422,
          216.60000610351562,
          217.60000610351562,
          219.8000030517578,
          221.1999969482422,
          224,
          256.3999938964844,
          4,
          4.199999809265137,
          4,
          9.199999809265137,
          9.800000190734863,
          10.600000381469727,
          11.800000190734863,
          12.600000381469727,
          38.599998474121094,
          40.79999923706055,
          92.80000305175781,
          94.5999984741211,
          4.199999809265137,
          89.4000015258789,
          99.4000015258789,
          104.5999984741211,
          105.19999694824219,
          105.19999694824219,
          143,
          143.39999389648438,
          6.400000095367432,
          7,
          7.599999904632568,
          7.400000095367432,
          8.600000381469727,
          7.800000190734863,
          9.600000381469727,
          8.800000190734863,
          9.600000381469727,
          10.600000381469727,
          11.199999809265137,
          12.199999809265137,
          14.600000381469727,
          16.200000762939453,
          16,
          16.200000762939453,
          16.799999237060547,
          17,
          20,
          18.399999618530273,
          20.200000762939453,
          20.600000381469727,
          20,
          19.799999237060547,
          22.799999237060547,
          25.600000381469727,
          26,
          27.399999618530273,
          29.200000762939453,
          29.399999618530273,
          31.200000762939453,
          33.599998474121094,
          35.400001525878906,
          35,
          36.400001525878906,
          33.400001525878906,
          38.20000076293945,
          34.599998474121094,
          43.20000076293945,
          43.79999923706055,
          45.599998474121094,
          45.79999923706055,
          49.400001525878906,
          50,
          52.20000076293945,
          51.599998474121094,
          54,
          58.599998474121094,
          69.19999694824219,
          73,
          73.80000305175781,
          76.4000015258789,
          77.19999694824219,
          78.5999984741211,
          84.19999694824219,
          85.4000015258789,
          85.19999694824219,
          86.80000305175781,
          88.19999694824219,
          89.80000305175781,
          91.80000305175781,
          19.600000381469727,
          35.20000076293945,
          43.400001525878906,
          48.79999923706055,
          77.19999694824219,
          78.4000015258789,
          168,
          196.1999969482422,
          201.39999389648438,
          220.39999389648438,
          4.400000095367432,
          5.400000095367432,
          6.199999809265137,
          7.199999809265137,
          8.199999809265137,
          9.399999618530273,
          10.399999618530273,
          11.800000190734863,
          7.800000190734863,
          13,
          9.199999809265137,
          10,
          10.399999618530273,
          12,
          16,
          16.600000381469727,
          13.199999809265137,
          16.799999237060547,
          15.199999809265137,
          19.799999237060547,
          20.399999618530273,
          20.399999618530273,
          19.600000381469727,
          22.600000381469727,
          21.600000381469727,
          24.399999618530273,
          26.600000381469727,
          27.799999237060547,
          26,
          26.399999618530273,
          28,
          31,
          28.600000381469727,
          30.200000762939453,
          32.599998474121094,
          32.20000076293945,
          34.20000076293945,
          36.400001525878906,
          34.599998474121094,
          38.20000076293945,
          36,
          36.79999923706055,
          40,
          40.79999923706055,
          42.400001525878906,
          44.20000076293945,
          45.400001525878906,
          46.20000076293945,
          44.400001525878906,
          47.79999923706055,
          50,
          51.20000076293945,
          51.400001525878906,
          51.599998474121094,
          54.79999923706055,
          57.400001525878906,
          60,
          62,
          61.20000076293945,
          64.80000305175781,
          66.80000305175781,
          69,
          68.4000015258789,
          71.4000015258789,
          73.80000305175781,
          69.80000305175781,
          75.4000015258789,
          76.5999984741211,
          70,
          75.80000305175781,
          77.80000305175781,
          80.19999694824219,
          76.19999694824219
        ]
pts_out_dict = pts_head_result_deserialize(pts_outs)
session3 = session3 = ort.InferenceSession("hvdet_fuse_stage3.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])        
radar_feat = get_valid_radar_feat(pts_out_dict, radar_pc, radar_cfg)
sec_feats = torch.cat([bev_feat, radar_feat], 1) 
output_names=[f'radar_out_{j}' for j in range(15)]
print("session3 radar out -----------------------")
sess3_radar_out=session3.run(output_names, 
                                            {
                                            'sec_feat':sec_feats.cpu().numpy(),
                                            }) 
print(sess3_radar_out)                                            
for i in range(len(sess3_radar_out)):
  sess3_radar_out[i] = torch.tensor(sess3_radar_out[i]).to(pts_outs[0].device)
pts_outs = pts_head_result_deserialize(pts_outs)
sec_outs=radar_head_result_deserialize(sess3_radar_out)
print("---sec_outs-----")
print("sec_outs:",sec_outs)
    


