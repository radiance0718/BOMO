This work is based on [Diffusion Posterior Sampling](https://github.com/DPS2022/diffusion-posterior-sampling)




## Local environment setting

This project relies on some external codebases.

To begin, clone the following repository:

```bash
git clone https://github.com/LeviBorodenko/motionblur motionblurInstall dependencies
```

Then, install the required dependencies:

```lua
conda create -n BOMO python=3.8
conda activate BOMO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```



## Download Dataset and Model

This project uses the publicly available *fundus* dataset. We have already uploaded the *retinalS004* dataset in this repository, but if you need additional data, you can download the full dataset from [URL].

You can also download our pretrained model from [URL].



## Running the Code

Once the environment is set up and the necessary data is downloaded, you can run the following command to start the process:

```bash
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config_ddim.yaml \
    --task_config=configs/noise_speckle_config.yaml \
    --gpu=$1 \
    --save_dir=./results/BOMO;
```
