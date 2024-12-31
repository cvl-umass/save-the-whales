# Save the Whales

## Inference

Inference is performed using the Google Colab platform.

Please follow the steps indicated in [Google Colab User Guide](https://docs.google.com/document/d/1Yp4yD7NGfHaZEL4DK3kzPJ2NvI89Duiy/edit) (also present as a PDF file in the ``inference`` folder) to run the notebook on Google Colab.

[``Whales_Inference_Module.ipynb``](https://drive.google.com/file/d/1_5urD8bAHAMsjnNn7gv8NS9XPINoJy25/view?usp=sharing) is the Jupyter notebook used to perform inference for new whale images.

Model weights: [Detectron2_Models.zip](https://drive.google.com/file/d/10zFCacIIp-N0NjwuOfhb_RIdCw_Fs_v3/view?usp=drive_link)

## Model Training

There are 3 scripts present in this repo:

1. ``Create Segmentation Masks June 2023.ipynb``: This script is used to create the segmentation mask. I have already run this script and the result is stored under ``data/final_segmentation_masks_28_july_2023_10_percent_increased_distance.pkl``.
2. ``training_segmentation_mask.py``: Used to train the instance segmentation model. Usage: ``training_segmentation_mask.py --lr <LEARNING RATE> --batch_size <BATCH SIZE>``.
3. ``training_keypoint_detection.py``: Used to train the keypoint detection model. Usage: ``training_keypoint_detection.py --lr <LEARNING RATE> --batch_size <BATCH SIZE>``

The best learning rate and batch size for both the models according to my hyperparameter tuning experiments are ``0.00225`` and ``2``, respectively.

## Postprocessing

Please follow the ``Postprocessing guidelines.pdf`` file in the ``postprocessing`` folder to correct the points predicted by our model.