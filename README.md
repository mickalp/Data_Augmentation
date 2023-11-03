# Data_Augmentation
New data augmentation method published by Furxi and Kalapus
https://www.tandfonline.com/doi/full/10.1080/17435390.2023.2268163

**Goal:** Increase the amount of data to improve regression results.

## Key features:
* Genereting artificial data based on UMAP -> convex hull -> poisson sampling methodolgy
* Number of generated data are dependent on radius set in poisson_disc_sampling(radius=0.1). Typically radius should be in range between 0.05 and 0.5.
* Possible Visualization of all steps


## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3
- NumPy
- Pandas
- UMAP-learn
- Matplotlib
- Pycaret
- Scikit-learn
- SciPy
- LightGBM
- Imbalanced-learn (for oversampling)
- PyTorch (if you want to utilize deep learning)

You can install most of these libraries using `pip` or `conda`.

```bash
pip install numpy pandas umap-learn matplotlib pycaret scikit-learn scipy lightgbm imbalanced-learn torch



