# IGP-UHM DL Model

This repo holds the code for running inference on the IGP-UHM model for extreme eastern pacific El Niño events. The model shares a similar architecture to that of Ham et. al. (2021).

## Get Started

After cloning the repo, you can download GODAS and NCEP data using the scripts inside the `data/raw` folder. Then you can run the notebooks in order using the numbered prefix. The notebooks are numbered in the order they should be run.

The inference would require the trained weights of the model. These can be downloaded from the [HuggingFace model hub](https://huggingface.co/GRiveraTello/IGP-UHM-v1.0). The weights should be placed in the `models` folder. Inside that folder there's a `Readme.md` file with instructions on how to download the weights.

## References

Ham, Y.-G., Kim, J.-H., Kim, E.-S., & On, K.-W. (2021). Unified deep learning model for El Niño/Southern Oscillation forecasts by incorporating seasonality in climate data. Science Bulletin, 66(13), 1358–1366. https://doi.org/10.1016/j.scib.2021.03.009
