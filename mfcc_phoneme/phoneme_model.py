# import torch
# import torch.nn as nn

# class PhonemeRecognitionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
#         super(PhonemeRecognitionModel, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         logits = self.fc(lstm_out)
#         return self.softmax(logits)

import torch
import torch.nn as nn
import torch.optim as optim

class PhonemeRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PhonemeRecognitionModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output one phoneme per sequence
    
    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_dim)
        rnn_out, _ = self.rnn(x)  # rnn_out.shape = (batch_size, sequence_length, hidden_dim)
        
        # Use the last output of the RNN for classification
        last_rnn_output = rnn_out[:, -1, :]  # Get the last time step's output (shape: (batch_size, hidden_dim))
        
        # Pass through the fully connected layer to get the class probabilities
        out = self.fc(last_rnn_output)  # out.shape = (batch_size, output_dim)
        
        return out

# Training code will remain the same
