## Vit4V: a Video Classification Method for the Detection of Varroa Destructor from Honeybee

The Varroa destructor mite threatens honey bee populations. We introduce **Vit4V**, a deep learning framework that analyzes video clips of bees for accurate, 98% detection of infestations. Our method outperforms existing techniques, offering a scalable, non-invasive solution for early detection, reducing chemical use, and supporting sustainable beekeeping.

The code is structured in the following way:
- `src/video2segments.py` contains the code to convert the long videos to video clips of 32 frames
- `src/train.py` contains the code for model training
- `src/test_video.py` tests the trained model with the original videos cropping the bee in real-time


