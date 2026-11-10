# Making Gaussians Count

Training and evaluation instructions for the final static 1.0 release.

## Installation

Use Linux, an NVIDIA GPU, a CUDA 12.4 toolkit (including `nvcc`), and a compatible C++ compiler. The tested Python stack is Python 3.11, PyTorch 2.4.0, and torchvision 0.19.0. The PyTorch wheel alone does not provide the CUDA compiler needed to build the bundled extensions.

```bash
git clone https://github.com/yudongbo1991/making_gaussians_count.git
cd making_gaussians_count

conda create -n making-gaussians-count python=3.11 pip -y
conda activate making-gaussians-count
python -m pip install setuptools wheel ninja packaging

# Change this path if your CUDA 12.4 toolkit is installed elsewhere.
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version

python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install -r requirements.txt
export MAX_JOBS=1
python -m pip install --no-build-isolation ./third_party/simple-knn
python -m pip install --no-build-isolation ./third_party/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./third_party/diff-surfel-rasterization
python -m pip install --no-build-isolation ./extensions/analytic
```

Build all extensions in this environment using the bundled sources. No recursive submodule download is needed. Build on the training GPU machine; if building for a different GPU, set `TORCH_CUDA_ARCH_LIST` to its compute capability before installation. Internet access is needed for package installation and the first download of LPIPS/VGG weights.

Check the installation:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python tests/smoke_cuda.py
```

## Data and configurations

Each dataset root contains its scene folders. Use undistorted COLMAP scenes with the following layout. Data and trained models are not included.

```text
/path/to/dataset/
  scene_name/
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
    images/           # or images_2 / images_4, as specified below
      ...
```

For Mip-NeRF 360, prepare the `images_4` and `images_2` folders before training; the scripts do not generate these downsampled folders.

| Dataset | Scenes | Voxel size | Images | Resolution argument |
|---|---:|---:|---|---:|
| Mip-NeRF 360 | 9 | 0.002 | `images_4` outdoors, `images_2` indoors | 1 |
| Tanks & Temples | 2 | 0.004 | `images` | -1 |
| Deep Blending | 2 | 0.004 | `images` | -1 |
| BungeeNeRF | 8 | 0.0005 | `images` | -1 |

These are the settings of **our method** used for the final experiments. With `resolution=-1`, the inherited loader limits image width to 1600 pixels while preserving aspect ratio; `resolution=1` preserves the selected image folder's resolution. Every eighth image in the sorted COLMAP image list is held out for testing.

All configurations use a 15,000-iteration teacher, 30,000 student iterations, 32-dimensional features, 10 slots per anchor, seed 0, and black backgrounds. The complete scene lists and dataset parameters are in [`configs/`](configs/). Keep the supplied configuration values to reproduce the released setting.

## Train

Activate the environment and run commands from the repository root. Replace the paths below with your own. `--data-root` points to the directory containing the scenes, not to an individual scene. Each script runs the complete training pipeline and evaluation automatically. Choose a new output directory; individual training/preparation stages refuse to overwrite existing results.

```bash
conda activate making-gaussians-count
bash scripts/run_mipnerf360.sh --data-root /path/to/mipnerf360 --output-root /path/to/outputs --gpu 0
bash scripts/run_tanks_and_temples.sh --data-root /path/to/tandt --output-root /path/to/outputs --gpu 0
bash scripts/run_deep_blending.sh --data-root /path/to/db --output-root /path/to/outputs --gpu 0
bash scripts/run_bungeenerf.sh --data-root /path/to/bungeenerf --output-root /path/to/outputs --gpu 0
```

Use `--scenes bicycle` (or another scene name) to run a subset. Add `--dry-run` to print commands without training. Scenes run sequentially on the specified GPU. The scripts use the active environment's `python`; set `PYTHON=/path/to/python` to select an interpreter explicitly.

```bash
# Print commands for one scene without starting training.
bash scripts/run_mipnerf360.sh --data-root /path/to/mipnerf360 \
  --output-root /path/to/outputs --scenes bicycle --gpu 0 --dry-run
```

### Reuse completed stages

An existing teacher directory must contain `cfg_args`, `input.ply`, `point_cloud/iteration_15000/point_cloud.ply`, and a successful `exit_status.json`. Its scene, image folder, resolution and train/test split must match the student. To reuse teachers arranged as `/path/to/teachers/<scene>/`:

```bash
bash scripts/run_mipnerf360.sh --data-root /path/to/mipnerf360 \
  --output-root /path/to/outputs --teacher-root /path/to/teachers \
  --stages score prepare train evaluate --gpu 0
```

Use `--stages train evaluate` after preparing inputs. These commands reuse completed stages in the same output directory layout; they do not resume a partially completed optimization run.

## Test and outputs

Evaluation runs automatically after training. To evaluate saved students again:

```bash
bash scripts/run_mipnerf360.sh --data-root /path/to/mipnerf360 \
  --output-root /path/to/outputs --scenes bicycle --stages evaluate --gpu 0
```

Or evaluate one saved student directly:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model /path/to/outputs/mipnerf360/bicycle/student --iteration 30000
```

For each scene, find the final model in `<output-root>/<dataset>/<scene>/student/point_cloud/iteration_30000/` and the results in `<output-root>/<dataset>/<scene>/student/evaluation.json`. Evaluation reports PSNR, SSIM, LPIPS, average active Gaussian count (Act. G.), FPS, and saved representation size in MiB. Keep the original dataset available at the path recorded during training when evaluating.

## Acknowledgements and license

This implementation builds on Scaffold-GS, 3D Gaussian Splatting, and 2D Gaussian Splatting. Upstream copyright notices and licenses are retained in this repository and in `third_party/` and `teacher/`. See [`LICENSE.md`](LICENSE.md) and the component license files for their applicable terms.
