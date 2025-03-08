# SOD: Small Object Detection based on InternImage

This repository implements small object detection using the InternImage model.

---

## Environment Setup

Follow these steps to set up your environment.

### Step 1: Check Your Environment

#### Option 1: Using Anaconda (CUDA 12.4)
Ensure your environment includes:
- **Python:** 3.11.11
- **PyTorch:** 2.5.1
- **torchvision:** 0.20.1

#### Option 2: Using Anaconda (CUDA 11.3)
*(Instructions for CUDA 11.3 are not provided yet.)*

---

### Step 2: Install mmcv & Other Dependencies

#### 2.1 Install PyTorch with CUDA 12.4
Run the following command to install PyTorch 2.5.1:
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### 2.2 Install timm and mmcv-full
Install the necessary packages using:
```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

#### [Library Issue] 
Timm has some issue at the 0.6.11 about dataclasses. 

Please check the next issue.

(https://github.com/huggingface/pytorch-image-models/issues/1530#issuecomment-1872945374)

#### 2.3 Install Other Requirements
Install additional dependencies:
```bash
pip install opencv-python termcolor yacs pyyaml scipy
Please use a version of numpy lower than 2.0
pip install numpy==1.26.4
pip install pydantic==1.10.13
```
#### Step 3: Compile custom Operators
Before compiling, verify that your nvcc version matches the CUDA version of PyTorch by running:

```bash
nvcc -V
```

Then, complie the custom operators:
```bash
cd lib/models/backbone/internimage/ops_dcnv3
sh ./make.sh
python test.py
```
***
## Acknowledgments
- Thanks to the [InternImage](https://github.com/OpenGVLab/InternImage/tree/master) library for providing the tools that helped us quickly implement our ideas.