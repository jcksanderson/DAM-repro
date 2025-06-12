from datasets import load_dataset

dataset_name = "ufldl-stanford/svhn"
category = "cropped_digits"
save_directory = "data/svhn"

dataset = load_dataset(dataset_name, name = category)
dataset.save_to_disk(save_directory)
