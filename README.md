# Brain encoding and decoding of vision: Semantic Control for fMRI-to-Image Synthesis

<!-- Badges -->


<!-- Teaser Figure -->
<p align="center">
  <img src="assets/fig_1.png" width="90%">
</p>
<p align="center">
  <em>Brain decoding from fMRI and visual reconstruction by our model.</em>
</p>

---

## Abstract

<!-- Paste your final, polished abstract here. -->
Reconstructing visual experiences from brain activity is a key challenge in neuroscience and Brain-Computer Interfaces...

---

## Framework

<!-- Add your final model architecture diagram here. -->
<p align="center">
  <img src="[Link to your architecture diagram, e.g., assets/framework.png]" width="90%">
</p>
<p align="center">
  <em>The overall architecture of our proposed fMRI-to-image synthesis framework.</em>
</p>

---

## Getting Started

### 1. Environment Setup

We recommend using Conda for environment management.

```bash
# 1. Clone this repository
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate your_env_name
```

Alternatively, you can install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

This research utilizes the BOLD5000 and Generic Object Decoding (GOD) datasets.

* **BOLD5000:** Please download the data from the [official BOLD5000 website](http://bold5000.org/).
* **GOD:** Please download the data from the [official GOD page on OpenNeuro](https://openneuro.org/datasets/ds001246).

After downloading, please follow the preprocessing scripts in the `scripts/` directory and place the final data under the `./data` directory.

### 3. Pre-trained Models

* **Stable Diffusion v1.5:** The weights will be automatically downloaded from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) on the first run.
* **MBM Encoder:** Download our pre-trained MBM encoder from `[Link to your MBM encoder weights]` and place it in the `checkpoints/` directory.
* **Our Final Model:** We also provide the final fine-tuned model weights. Download them from `[Link to your final model weights]` and place them in the `checkpoints/` directory.

### 4. Training

To train the model from scratch, use the following command. Please modify the parameters in the config file as needed.

```bash
# Train our best-performing model on BOLD5000 (Subject CSI1)
python train.py \
    --config configs/your_best_model_config.yaml \
    --data_path ./data/bold5000/CSI1 \
    --output_dir ./outputs
```

### 5. Inference & Evaluation

To generate images using our pre-trained model:

```bash
# Reconstruct images from the test set
python inference.py \
    --checkpoint ./checkpoints/our_final_model.pth \
    --fmri_data_path ./data/bold5000/CSI1/test_fmri.npy \
    --output_dir ./reconstructions

# Evaluate the reconstructed images
python evaluate.py \
    --recon_dir ./reconstructions \
    --gt_dir [Path to ground truth images]
```

---

## Citation

If you use our work or code in your research, please cite our paper:

```bibtex
@article{your_lastname_2025_brain,
  title   = {Brain encoding and decoding of vision: Semantic Control for fMRI-to-Image Synthesis},
  author  = {[Your Name and Co-authors]},
  journal = {arXiv preprint arXiv:...},
  year    = {2025}
}
```

---

## Acknowledgements

This work is built upon the codebases of [MinD-Vis]([Link to MinD-Vis GitHub]) and [Hugging Face Diffusers](https://github.com/huggingface/diffusers). We thank the creators of the BOLD5000 and GOD datasets for their invaluable contributions to the community.
