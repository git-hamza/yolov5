## Training with WandB

**Note:** The data is expected to be in the following format. `DIR/images` for images and `DIR/labels` for labels. DIR could be
the main directory if the data is not split, or could be any of the split folders. Moreover, incase you want to split your data
from the current fromat, please go to `data_preprocessing.ipynb` file.

### Upload local dataset and train
In this case you probably have your dataset locally, and you just want to train your model as well
as uplaod your dataset to wandb.

To upload your dataset to wandb, please run the following script with proper argument.
`utils/loggers/wandb/log_dataset.py`

After that you can proceed on training your model.

### Download data from wandb, combine it with local data and train.
For this you have to follow the following procedure.

- run `utils/loggers/wandb/download_dataset.py` script to download the dataset from wandb artifact.
- now run the jupyter notebook `data_preprocessing.ipynb` with proper paths of previous and your current data.
- upload your updated dataset using `utils/loggers/wandb/log_dataset.py` script.
- proceed training

Right now the latest model is saved in artifact autmatically, later we will merge script for
model fetching as well as manually uplaod a model.
