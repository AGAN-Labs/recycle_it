from pathlib import Path
from datetime import datetime

debug_flag = True

train_num_workers = 0
validation_num_workers = 0
data_dir = Path(__file__).parent.parent.joinpath('data/garbage_classification/')
data_model_path = Path(__file__).parent.parent.joinpath('data/models/image_model_{}.pkl'.format(str(datetime.today().date())))
num_epochs = 8
learning_rate = 5.5e-5
shuffle_training = True
pin_memory = True
batch_size = 16
save_model = True
load_model = True
external_images = Path(__file__).parent.parent.joinpath('data/external_images/')
load_data_path = Path(__file__).parent.parent.joinpath('data/models/image_model_2021-07-01.pkl')
