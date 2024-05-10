# IGP-UHM DL Model

This repo holds the code for running inference on the IGP-UHM model for extreme eastern pacific El Niño events. The model architecture and results are discussed in  Rivera Tello et al. (2023).

## Get Started

After cloning the repo, you can download GODAS and NCEP data using the scripts inside the `data/raw` folder. Then you can run the notebooks in order using the numbered prefix. The notebooks are numbered in the order they should be run.

The inference would require the trained weights of the model. These can be downloaded from the [HuggingFace model hub](https://huggingface.co/GRiveraTello/IGP-UHM-v1.0). The weights should be placed in the `models` folder. Inside that folder there's a `Readme.md` file with instructions on how to download the weights.

## References

Rivera Tello, G. A., Takahashi, K., & Karamperidou, C. (2023). Explained predictions of strong eastern Pacific El Niño events using deep learning. Scientific Reports, 13(1), Article 1. https://doi.org/10.1038/s41598-023-45739-3

