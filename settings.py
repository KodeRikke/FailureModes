path = "./"

model_dir = path + 'saved_models/'

# number of experiments
num_experiments = 2

# reproducibility
seeds = [1337] * num_experiments

# training
epochs = 11
warm_epochs = 5
epoch_reached = 0
push_start = 10
last_layer_iterations = 5
push_epochs = [i for i in range(epochs) if i % 10 == 0]

# loss parameters
coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}

# batch sizes
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

# prototypes
img_size = 224
num_prototypes = 10 # (per class)
num_classes = 100

# model
model_names = [""] * num_experiments # ["2nopush0.123"]
save_name = ""
base_architectures = ["resnet34", "vgg19"]