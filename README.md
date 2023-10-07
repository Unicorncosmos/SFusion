# SFusion

!pip install mmcv==1.6.2

!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!git checkout v2.28.1
!pip install -r requirements/build.txt
!pip install -v -e .

!wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth

#BEV_POOLV2 setup
!python setup.py


#OPS_DCNV3 setup

