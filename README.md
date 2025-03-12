## Vit4V: a Video Classification Method for the Detection of Varroa Destructor from Honeybee

The Varroa destructor mite threatens honey bee populations. We introduce **Vit4V**, a deep learning framework that analyzes video clips of bees for accurate, 98% detection of infestations. Our method outperforms existing techniques, offering a scalable, non-invasive solution for early detection, reducing chemical use, and supporting sustainable beekeeping.

- `src/video2segments.py` contains the code to convert the long videos to video clips of 32 frames
- `src/train.py` contains the code for model trainingt the long videos to video clips of 32 frames
- `src/test_video.py` tests the trained model with the original videos cropping the bee in real-time

### Running  `video2segments.py`


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

```bash
python test_video.py --model <path_to_model> --video <path_to_video> [options]
```
Where the parameters are:
- `--model`: Path to the model weights (required).
- `--video`: Path to the video to process (required).
- `--window_size`: Size of the temporal window to process (default: 16).