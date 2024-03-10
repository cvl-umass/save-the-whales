# Save the Whales

There are 3 scripts present in this repo:

1. ``Create Segmentation Masks June 2023.ipynb``: This script is used to create the segmentation mask. I have already run this script and the result is stored under ``data/final_segmentation_masks_28_july_2023_10_percent_increased_distance.pkl``.
2. ``training_segmentation_mask.py``: Used to train the instance segmentation model. Usage: ``training_segmentation_mask.py --lr <LEARNING RATE> --batch_size <BATCH SIZE>``.
3. ``training_keypoint_detection.py``: Used to train the keypoint detection model. Usage: ``training_keypoint_detection.py --lr <LEARNING RATE> --batch_size <BATCH SIZE>``

The best learning rate and batch size for both the models according to my hyperparameter tuning experiments are ``0.00225`` and ``2``, respectively.
