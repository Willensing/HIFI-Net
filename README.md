# HIFI-Net

HIFI-Net is a deep learning network for saliency-aware semantic image manipulation localization. The model jointly predicts manipulation masks and boundaries and uses the Multi-Cue Relationship Suppression (MCRS) loss during training.

## Pretrained Weights

The pretrained U2-Net weights and the final trained HIFI-Net weights are available through Baidu Netdisk:

- Files: HIFI weight files
- Link: https://pan.baidu.com/s/1jwi5FOSaBUl0UHiN3I2MaQ?pwd=sn78
- Extraction code: `sn78`

## SML Dataset

The Semantic Manipulation Localization (SML) dataset used in our experiments is available through Baidu Netdisk:

- Files: SML dataset
- Link: https://pan.baidu.com/s/16_posYGRuTp5GDgrKv7H8g?pwd=tisz
- Extraction code: `tisz`

Repository: https://github.com/Willensing/HIFI-Net

## Project Structure

```text
HIFInet/
├── config/
│   └── config.yaml          # Main configuration file for training and data paths
├── dataloader/
│   └── myLoader.py          # Dataset and DataLoader implementation
├── model/
│   ├── model.py             # Main HIFI-Net model
│   ├── aspp.py
│   └── srm.py
├── model_U2net/             # U2-Net backbone-related code
├── utils/
│   └── utils.py             # Metrics, contrastive loss, logging, and utilities
├── trainer.py               # Training and validation loop
├── evaluate.py              # Standalone test-set evaluation script
└── bestModels/              # Best checkpoints generated during training
```

## Requirements

Python 3.8 or later and a CUDA-enabled version of PyTorch are recommended. The main dependencies are:

- `torch` and `torchvision`
- `numpy`
- `opencv-python` (`cv2`)
- `PyYAML`
- `scikit-learn`
- `Pillow`
- `tqdm`
- `scipy`

Example installation command:

```bash
pip install torch torchvision numpy opencv-python pyyaml scikit-learn pillow tqdm scipy
```

## Configuration

All training-related parameters are defined in `config/config.yaml`. The configuration contains three main sections:

| Section | Description |
|---|---|
| `model_params` | Optimizer, learning rate, number of epochs, and the MCRS loss weight `con_alpha` |
| `dataset_params` | Dataset paths, batch size, input size, and normalization mean and standard deviation |
| `contrastive_params` | MCRS hyperparameters, including temperature and patch length |

### `model_params`

```yaml
model_params:
  optimizer: 'adam'    # Supported options: 'adam' or 'sgd'
  lr: 0.0005
  epoch: 100
  con_alpha: 1         # Overall weight of the contrastive loss
```

### `dataset_params` (Required)

Replace the following paths with the corresponding dataset directories on your machine. Absolute paths are recommended. On Windows, forward slashes `/` can be used in paths.

```yaml
dataset_params:
  train_img_dir: 'D:/data/train/images'
  train_mask_dir: 'D:/data/train/masks'
  train_edge_dir: 'D:/data/train/edges'

  val_img_dir: 'D:/data/val/images'
  val_mask_dir: 'D:/data/val/masks'
  val_edge_dir: 'D:/data/val/edges'

  batch_size: 4
  im_size: 256
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

### `contrastive_params` (Optional)

```yaml
contrastive_params:
  temperature: 0.6
  p_len: 4
  in_out_weight: 0.3
  reliable_thresh: 0.05
  gamma: 2.0
```

### Standalone Evaluation Configuration

The paths used by `evaluate.py` must be configured directly at the beginning of the script because the test paths are not read from the YAML file:

- `CONFIG_PATH`: path to the configuration file
- `MODEL_PATH`: path to the `.pth` checkpoint
- `IMAGE_FOLDER`: directory containing the test images
- `MASK_FOLDER`: directory containing the ground-truth test masks

## Data Format

The data-loading logic is implemented in `dataloader/myLoader.py`. Each training or validation split requires three parallel directories:

| Configuration entry | Contents |
|---|---|
| `*_img_dir` | Original RGB images |
| `*_mask_dir` | Binary or grayscale manipulation masks with matching filenames |
| `*_edge_dir` | Manipulation boundary masks with matching filenames |

### File Naming Rules

1. Image filenames must begin with `img`, for example, `img001.png` or `img_0001.jpg`. The loader filters files using `f.startswith("img")`.
2. Each mask and edge map must have the same base filename as its corresponding image, although the file extensions may differ. Supported extensions are `png`, `PNG`, `tif`, `TIF`, `jpg`, and `JPG`.
3. Samples in the image, mask, and edge directories must have one-to-one correspondence. At startup, the loader prints a list of any missing masks or edge maps.

Example directory structure:

```text
train/
├── images/
│   ├── img001.png
│   └── img002.jpg
├── masks/
│   ├── img001.png
│   └── img002.png
└── edges/
    ├── img001.png
    └── img002.png
```

### Image and Annotation Formats

| Type | Requirements |
|---|---|
| Image | Any common OpenCV-readable format. Images are loaded in BGR order, resized to `im_size x im_size` (256 by default), scaled to `[0,1]`, and normalized using the ImageNet mean and standard deviation. |
| Mask | Single-channel grayscale image with values in the range 0-255. Masks are resized using `INTER_NEAREST` and divided by 255 to obtain floating-point labels in `[0,1]`. |
| Edge map | Same format as the manipulation mask. Edge maps are used by the boundary branch with `BCEWithLogitsLoss`. |

Edge maps must be prepared in advance, for example, by applying a morphological gradient or an edge detector to the binary manipulation masks. The training code does not automatically generate edge maps from the masks.

### Test Masks in `evaluate.py`

- Masks are loaded as grayscale images using Pillow and resized to 256 x 256.
- Pixels greater than 127 are treated as manipulated pixels (`1`); all other pixels are treated as background (`0`).
- Image and mask lists are aligned using `sorted(os.listdir(...))`. Their lengths must match, and their sorted orders must correspond exactly. The script does not match files by their base filenames.

## Training and Evaluation

### Training

1. Prepare the dataset and update the paths in `config/config.yaml`.
2. Run the following command from the project root:

```bash
python trainer.py
```

### Training Procedure

- During each epoch, the model is first optimized on the training set and then evaluated on the validation set without gradient computation.
- The objective includes the main segmentation loss, boundary loss, three auxiliary segmentation losses, and the MCRS contrastive losses computed from three feature levels.
- The learning rate is multiplied by 0.8 every 20 epochs using `StepLR`.
- Validation metrics include IoU and AUC. For each image, the larger value of the positive-class and negative-class AUCs is used before dataset-level averaging.
- The best checkpoints are saved in `bestModels/`:
  - `model_best_auc_epoch{N}.pth`: checkpoint with the highest validation AUC
  - `model_best_iou_epoch{N}.pth`: checkpoint with the highest validation IoU

### Training Logs

The `write_logger` function in `utils/utils.py` writes logs to the fixed path `home/user1/HIFI-Net/results/`. If this directory does not exist on your machine, update the path in `write_logger` or create the directory manually; otherwise, log writing may fail.

### Standalone Test Evaluation

1. Update `MODEL_PATH`, `IMAGE_FOLDER`, and `MASK_FOLDER` in `evaluate.py`.
2. Run:

```bash
python evaluate.py
```

For each image, the script reports AUC, F1-score, IoU, Dice, recall, precision, exact match, and MAE. It then prints the dataset-level average of each metric. Input images use the same size and normalization settings as the training data. Predicted masks are converted to probabilities using the sigmoid function and binarized using a threshold of 0.5.

> **Note:** The default `MODEL_PATH` in `evaluate.py` is `bestModels\best_HIFI.pth`, whereas the training script saves checkpoints as `model_best_auc_epoch*.pth` or `model_best_iou_epoch*.pth`. Update the path to the actual checkpoint or copy and rename the selected checkpoint to `best_HIFI.pth` before evaluation.

## Troubleshooting

### `FileNotFoundError: Image/Mask/Edge file not found`

Check the paths in the YAML file, verify that image filenames begin with `img`, and ensure that every mask and edge map has the same base filename as its corresponding image.

### Single-class AUC error on the validation or test set

A ground-truth mask containing only background pixels or only manipulated pixels does not define an ROC curve. `evaluate.py` skips these samples when calculating AUC and reports how many samples were skipped.

### Windows paths

The configuration path in `trainer.py` uses `config\config.yaml`. On Linux, change it to `config/config.yaml` if necessary.

### GPU selection

The default device is `cuda:0`. If no compatible GPU is available, the code automatically falls back to the CPU, which will be considerably slower.

## Quick Checklist

- [ ] All six dataset directories in `config/config.yaml` are configured and exist.
- [ ] Image filenames begin with `img`, and all masks and edge maps can be matched by base filename.
- [ ] PyTorch and the required dependencies are installed.
- [ ] The edge-map directories have been prepared.
- [ ] The paths and checkpoint filename in `evaluate.py` are correct.
