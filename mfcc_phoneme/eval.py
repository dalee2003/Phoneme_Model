import torch
from torch.utils.data import DataLoader
from phoneme_model import PhonemeRecognitionModel  # Import your model definition
from timit_dataset import get_data_loaders  # Import the function to load your dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define constants (make sure they are the same as in your training script)
INPUT_DIM = 13  # Number of MFCC coefficients
HIDDEN_DIM = 128
OUTPUT_DIM = 61  # Number of phonemes (adjust this as per your labels)
NUM_LAYERS = 2
BATCH_SIZE = 32

# Load the test data
test_loader = get_data_loaders(
    train_dir=r"C:\Users\daphn\Model\data\processed\train", 
    test_dir=r"C:\Users\daphn\Model\data\processed\test", 
    batch_size=BATCH_SIZE
)[1]  # Get the test loader (the second item in the tuple)

# Initialize the model and load the trained weights
model = PhonemeRecognitionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
model.load_state_dict(torch.load("phoneme_recognition_model.pth"))
model.eval()  # Set the model to evaluation mode

# Evaluate the model on the test set
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():  # No need to compute gradients during evaluation
    for mfcc, labels in test_loader:
        # Forward pass
        outputs = model(mfcc)

        # Get the predicted phonemes (the index of the max probability for each input)
        _, predicted = torch.max(outputs, 1)

        # Update the total number of samples and the number of correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect the true labels and predictions for further analysis
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy}%')

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# Confusion matrix visualization
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(OUTPUT_DIM), yticklabels=range(OUTPUT_DIM))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
