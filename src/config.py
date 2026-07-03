import torch

TRAIN_DIR = '../dataset/train'
VALID_DIR = '../dataset/valid'
TEST_DIR = '../dataset/test'

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CALSSES = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "../models/best_model.pth"