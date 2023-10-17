# SFusion
#Interimage + LSSViewTransformer + Depthnet

**Step 1.** Install requirement
```shell
!pip install mmcv==1.6.2
!pip install -q xformers==0.0.19 torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchtext==0.15.1 torchaudio==2.0.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu117
!git clone https://github.com/open-mmlab/mmdetection.git
```

```
%cd mmdetection
!git checkout v2.28.1
!pip install -r requirements/build.txt
!pip install -v -e .
```

**Step 2.** Install onnx
```
pip install onnx
pip install onnxruntime
```
```
!wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth
```
**BEV_POOLV2 setup.**
```
!python setup.py install
``` 

**OPS_DCNV3 setup.**
```
!python ops_dcnv3/setup.py install
```

**ONNX Download.** 

- download [onnx_stage1](https://drive.google.com/file/d/1Axj6HlAZ6hCEkWnqVesRDXjsE_LqSl_b/view?usp=sharing), [onnx_stage1_1](https://drive.google.com/file/d/1U0TqBTz3v-zkgTfyVgCMmrg3Dmo7Fqcy/view?usp=sharing), [onnx_stage2](https://drive.google.com/file/d/17WI0N9lyME1ZSfR4ftG_JcT5yYjkpEMs/view?usp=sharing), [onnx_stage3](https://drive.google.com/file/d/1uv95hDg-KW7Cw0RG8w9NfWGQAdoi0YY0/view?usp=sharing) 

**Model_build_with_sample_tensor.**
```
!python model_build.py
```
