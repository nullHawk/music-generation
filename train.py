import torch
import torch.nn as nn
import torch.optim as optim
from model import MusicGenerationLSTM

def train_model(model, train_data, train_labels, num_epochs=200, batch_size=128, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_labels in loader:
            batch_data, batch_labels = batch_data.to(model.device), batch_labels.to(model.device)
            outputs, _ = model(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(loader):.4f}")
