from pathlib import Path

debug_flag = True

train_num_workers = 0
validation_num_workers = 0
data_dir = Path(__file__).parent.parent.joinpath('data/garbage_classification/')
data_model_path = Path(__file__).parent.parent.joinpath('data/models/')
num_epochs = 8
learning_rate = 5.5e-5
shuffle_training = True
pin_memory = True
batch_size = 32