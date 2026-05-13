# Fedsoda

**Fedsoda** is a robust Federated Learning (FL) framework designed for medical image segmentation, specifically optimized for multi-center collaboration while maintaining data privacy. The framework implements a dynamic parameter fusion mechanism that accounts for client-side distribution shifts and model similarity.

## Key Features
- **Generic Client Architecture**: Easily scalable to any number of participating centers.
- **Dynamic Parameter Fusion**: Implements a similarity-aware weight fusion strategy using middle-layer activations.
- **Privacy-Preserving**: Only model weights are shared between clients and the server; raw data never leaves the local environment.
- **Medical Image Optimized**: Built-in support for SimpleITK and medical-specific data augmentations via MONAI.

## Project Structure
- `main.py`: The central entry point for all clients. Handles configuration and orchestration.
- `trainer.py`: Generic training logic for federated steps and local epochs.
- `dataloader.py`: Configurable data loader for different datasets and centers.
- `Params_fusion.py`: Implementation of the Fedsoda fusion algorithm.
- `Network.py`: UNet-based architecture for medical segmentation.
- `Loss.py`: Specialized loss functions including Balanced BCE and Dice-based losses.
- `Transform.py`: Data augmentation pipeline using MONAI.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fedsoda.git
   cd Fedsoda
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### 1. Data Preparation
Organize your data and create `.txt` files containing paths to your images and labels. Place them in a directory (e.g., `./Txt/Txt_Cell/`).

Each client should have:
- `c{id}_train_image.txt`
- `c{id}_train_label.txt`
- `c{id}_test_image.txt`
- `c{id}_test_label.txt`

### 2. Run a Client
Start a training process for a specific client (e.g., Client 0):
```bash
python main.py --client_id 0 --base_dir ./MyExperiment/ --data_dir ./MyData/ --txt_dir ./MyTxt/
```

To run multiple clients, open separate terminals and run the same command with different `--client_id` values (0 to 6 by default).

### 3. Training Options
You can customize the training using command-line arguments:
- `--batch_size`: Batch size for local training.
- `--lr`: Learning rate.
- `--n_epochs`: Total number of epochs.
- `--n_epoch_per_step`: Number of local epochs before a federated fusion step.
- `--ROI_shape`: Size of the image crops for training.

## Fusion Logic
Fedsoda uses a "waiting-file" synchronization method. Clients will wait for each other to complete a federated step by checking for the existence of weight files in the shared `checkpoint_dir`. Once all clients have finished a step, the fusion logic calculates model similarities based on synthetic data and updates the global and local models.

## Citation

If you use this code or our Fedsoda method in your research, please cite our paper:

```bibtex
@inproceedings{zhang2024fedsoda,
  title={Fedsoda: Federated cross-assessment and dynamic aggregation for histopathology segmentation},
  author={Zhang, Yuan and Qi, Yaolei and Qi, Xiaoming and Senhadji, Lotfi and Wei, Yongyue and Chen, Feng and Yang, Guanyu},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1656--1660},
  year={2024},
  organization={IEEE}
}
```

---
*Developed for robust medical image analysis.*
