There are 4 scripts present in this repo:

1. ``Create Segmentation Masks June 2023.ipynb``: This script is used to create the segmentation mask. I have already run this script and the results are stored under ``data/final_segmentation_masks.pkl``.
2. ``training_segmentation_mask.py``: Used to train the instance segmentation model. Usage: ``training_segmentation_mask.py --lr <LEARNING RATE> --batch_size <BATCH SIZE>``.
3. ``training_keypoint_detection.py``: Used to train the keypoint detection model. Usage: ``training_keypoint_detection.py --lr <LEARNING RATE> --batch_size <BATCH SIZE>``
4. ``Whale_Inferene_Module.ipynb``: Used to run inference on images stored on Google Drive using Google Colab.

The best learning rate and batch size for both the models according to my hyperparameter tuning experiments are ``0.00225`` and ``2``, respectively.

