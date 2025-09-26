## Vit4V: a Video Classification Method for the Detection of Varroa Destructor from Honeybee

The Varroa destructor mite threatens honey bee populations. We introduce **Vit4V**, a deep learning framework that analyzes video clips of bees for accurate, 98% detection of infestations. Our method outperforms existing techniques, offering a scalable, non-invasive solution for early detection, reducing chemical use, and supporting sustainable beekeeping.

- `src/video2segments.py` contains the code to convert the long videos to video clips of 32 frames
- `src/train.py` contains the code for model training the long videos to video clips of 32 frames
- `src/test_video.py` tests the trained model with the original videos cropping the bee in real-time

## Authors
- Luca Giovannesi, giovannesi@diag.uniroma1.it
- Paolo Russo prusso@diag.uniroma1.it
- Roberto Beraldi beraldi@diag.uniroma1.it

## Paper
Read the related paper [here](https://doi.ieeecomputersociety.org/10.1109/CVPRW67362.2025.00531)

## VD<sup>2</sup> Dataset
You can download the datasets using the following:
- Video dataset: [https://doi.org/10.34740/kaggle/dsv/11159683](https://doi.org/10.34740/kaggle/dsv/11159683)
- Frame dataset: [https://doi.org/10.34740/kaggle/dsv/11254629](https://doi.org/10.34740/kaggle/dsv/11254629)

## Model
A pretrained model can be downloaded [here](https://drive.google.com/file/d/1_iPDWghnZ9cbfWaGQ8wuxqkq1qq3s-qO/view?usp=sharing)

## Project
Additional information about the research project [here](https://alcorlab.diag.uniroma1.it/projects/ev2)

## How to run
This software utilizes a devcontainer with [Podman](https://podman.io/), providing a simple way to create a reproducible environment across devices.  

### Setup Instructions  
1. **Download the dataset** to your local machine.  
2. Open `.devcontainer/devcontainer.json`.  
3. Locate the `mounts/source` field and update it with the path where your dataset is stored.  
4. Open the devcontainer using the VSCode graphical interface
5. Create a python3.11 environment `python3.11 -m venv env`
6. Install the packages in the requirements.txt file `./env/bin/pip install -r requirements.txt`

### Running  `video2segments.py`

First, it is necessary to create the video-clips of 32 frames starting from the original video

```bash
python video2segments.py --video_path <path_to_videos> --output_path <path_to_output> [options]
```
Where parameters are:

- `--video_path`: Folder containing the varroa_infested and varroa_free folders with original videos (required).
- `--output_path`: Folder where the segmented videos will be saved (required).
- `--validation_split`: Percentage of videos to use for validation (default: 0.3).
- `--window_size`: Number of frames processed by the model (default: 16).
- `--seed`: Seed for random operations (default: 1234).
- `--jobs`: Number of threads to use for processing (default: 2).

### Running  `train.py`

We start the training using the generated segments.

```bash
python train.py --dataset <path_to_dataset> [options]
```
Where the parameters are:

- `--batch_size`: Batch size for training (default: 1).
- `--epochs`: Number of epochs for training (default: 30).
- `--dataset`: Path to the dataset (required).
- `--no_aug`: Disable data augmentation (default: False).
- `--lr`: Learning rate (default: 1e-3).
- `--devices`: Devices to use for training (default: "0").
- `--pre_trained_model`: Path to a pre-trained model (default: None).
- `--model`: Model to use (default: "vivit").
- `--hidden_layer`: Number of hidden layers (default: 0).
- `--worker`: Number of workers for data loading (default: 8).

### Running `test_video.py`

The model is tested on the original videos left for validation.

```bash
python test_video.py --model <path_to_model> --video <path_to_video> [options]
```
Where the parameters are:
- `--model`: Path to the model weights (required).
- `--video`: Path to the folder containing the folders *varroa_free* and *varroa_infested* with the mkv files inside (required).
- `--window_size`: Size of the temporal window to process (default: 32).

## Acknowledgments
This work was (partially) supported by project no. 202277 WMAE CUP B53D23012820006, “EdgeVision against Varroa (EV2): Edge computing in defence of bees” funded by the Italian’s MUR PRIN2022 (ERC PE6) research program.

## Licence
The code, models and dataset used by this repository subject to the licence [Attribution-NonCommercial-NoDerivatives 4.0 International](LICENCE.md)
