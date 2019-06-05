# superres

This is a benchmark project to make higher resolution versions of low resolution flower images:

## Getting started

1. Be sure to [sign up](https://app.wandb.ai/login?signup=true) for W&B.
2. Clone this repository: `git clone https://github.com/wandb/superres.git`
3. Run `pip install -U -r requirements.txt` to install requirements.
4. Run `python train.py` to train the baseline model. Modify this file and the data pipeline (or write your own scripts and create different model architectures!) to get better results.
5. Submit your results to the [benchmark](https://app.wandb.ai/wandb/superres/benchmark).

![rose](https://user-images.githubusercontent.com/17/58977464-a7104080-877e-11e9-82b1-24abe5677ee1.jpg)

## The dataset

The dataset is comprised of images of flowers.  The training set has 5000 images of flowers that have been resized.  The test set has 670 images of flowers.  The input size is 32x32 pixels and the output size is 256x256 pixels.


## The goal

The goal is to enhance a low resolution input image to be 8 times greater resolution with the least loss of quality.

## Evaluation

We use a [perceptual distance](https://www.compuphase.com/cmetric.htm) metric (val_perceptual_distance) on the validation set to rank results (lower values are better).

## Submitting your results

You can submit your best runs to our [benchmark](https://app.wandb.ai/wandb/superres/benchmark). More specifically, go the "Runs" table in the "Project workspace" tab of your project.
Hover over the run's name, click on the three-dot menu icon that appears to the left of the name, and select "Submit to benchmark".

## Things to try

- Implement a GAN
- Different loss functions
- Data augmentation
