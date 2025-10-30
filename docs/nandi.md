## Nandi Land Cover and Crop Type Mapping

OlmoEarth-v1-FT-Nandi-Base is a model fine-tuned from OlmoEarth-v1-Base for predicting land cover and crop types across the Nandi county in Kenya using Sentinel-2 satellite images.

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-Nandi-Base/blob/main/model.ckpt
- Annotation GeoJSONs: https://huggingface.co/datasets/allenai/olmoearth_projects_nandi/tree/main
- rslearn dataset: https://huggingface.co/datasets/allenai/olmoearth_projects_nandi/blob/main/dataset.tar

## Model Details

The model inputs twelve timesteps of satellite image data, one [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) mosaic per 30-day period.

The model is trained to predict land cover and crop type for every pixel within each 16×16 input patches.

It achieves an overall accuracy of 87.3% on our validation set.

## Training Data



## Inference

TODO

## Fine-tuning

Fine-tuning is documented in [the main README](../README.md).