import torch
import torch.nn as nn
import torch.optim as optim
from timit_dataset import get_data_loaders
from phoneme_model import PhonemeRecognitionModel

# Define constants
INPUT_DIM = 13  # Number of MFCC coefficients
HIDDEN_DIM = 128
OUTPUT_DIM = 61  # Number of phonemes
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Get data loaders
train_loader, test_loader = get_data_loaders(
    train_dir=r"C:\Users\daphn\Model\data\processed\train", 
    test_dir=r"C:\Users\daphn\Model\data\processed\test", 
    batch_size=BATCH_SIZE
)

# Define model, loss, and optimizer
model = PhonemeRecognitionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for mfcc, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(mfcc)  # Shape: (batch_size, seq_len, OUTPUT_DIM)
        #print(outputs)

        # Check if the labels are of shape (batch_size, seq_len)
        #print(f"outputs.shape = {outputs.shape}, labels.shape = {labels.shape}")

        # Flatten the outputs and labels for CrossEntropyLoss
        # Reshape outputs to (batch_size * seq_len, OUTPUT_DIM)
        outputs = outputs.view(-1, OUTPUT_DIM)  # Shape: (batch_size * seq_len, OUTPUT_DIM)

        # Flatten labels to (batch_size * seq_len)
        labels = labels.view(-1)  # Shape: (batch_size * seq_len)

        # Ensure that the labels are within the valid range [0, OUTPUT_DIM-1]
        # CrossEntropyLoss expects labels in the range [0, OUTPUT_DIM-1]
        assert labels.max() < OUTPUT_DIM, "Labels should be in the range [0, OUTPUT_DIM-1]"

        # Calculate the loss
        loss = criterion(outputs, labels)  # CrossEntropyLoss compares the flattened outputs and labels
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}")

# Save the model
torch.save(model.state_dict(), "phoneme_recognition_model.pth")
print("Training complete. Model saved.")

