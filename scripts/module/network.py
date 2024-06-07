
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 17, 128)  # Correct input size after pooling
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))  # Output shape: [batch_size, 16, 71]
        x = self.pool(self.relu(self.conv2(x)))  # Output shape: [batch_size, 32, 35]
        x = self.pool(self.relu(self.conv3(x)))  # Output shape: [batch_size, 64, 17]
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImprovedCNNModel(nn.Module):
    def __init__(self, input_dim=142, input_dim_embedding=37, hidden_dim=128, num_filters=32, output_dim=3, dropout_prob=0.1):
        super(ImprovedCNNModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_dim_embedding, embedding_dim=hidden_dim, padding_idx=0)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*3 * (input_dim // 8), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x.long()).permute(0, 2, 1)  # Ensure input is LongTensor
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, input_dim=142, input_dim_embedding=37, hidden_dim=128, lstm_layers=2, output_dim=3, dropout_prob=0.1):
        super(RNNModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_dim_embedding, embedding_dim=hidden_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x.long())  # Ensure input is LongTensor
        x, (hn, cn) = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim=142, input_dim_embedding=37, hidden_dim=128, lstm_layers=2, output_dim=3, dropout_prob=0.1):
        super(LSTMModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_dim_embedding, embedding_dim=hidden_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x.long())  # Ensure input is LongTensor
        x, (hn, cn) = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

