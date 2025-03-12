# $\text{SAM-MoPE}$

$\text{SAM-MoPE}$ is a fine-tuning model based on SAM, mainly designed for universal medical image segmentation.

## Installation Steps

You can follow the steps below to install:

1. Clone the project to your local machine using the following command:
   
 <!--  git clone https://github.com/YuancongLiang/SAM-MoPE.git -->

2. Navigate to the project directory:
   ```
   cd SAM-MoPE
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```
   conda create -n sammope python=3.10
   ```

## Usage

Here are some common usage examples:
you need a pre training weight from SAMMed2d or SAM-vit-b
SAMMed2d:
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link
sam-vit-b:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
or download from https://github.com/facebookresearch/segment-anything

- Example 1: Run the training script
  ```
  python train.py --run_name fives --epochs 60 --batch_size 32 --resume pretrain_model/sam-med2d_b.pth
  ```

- Example 2: Perform inference using a pre-trained model
  ```
  python test.py --sam_checkpoint ./workdir/models/mope/epoch60_sam.pth
  ```

Feel free to modify and adjust these examples according to your specific task and requirements.

## Contributing Guidelines

If you would like to contribute to this project, please follow these steps:

1. Fork the project and make your modifications.

2. Submit a Pull Request to submit your changes to our repository.

3. We will review your Pull Request and merge appropriate changes.

## Copyright and License

This project is licensed under the Apache License. For more details, please refer to the [LICENSE](LICENSE).
