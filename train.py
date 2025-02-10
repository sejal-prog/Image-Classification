import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import ResNet, BasicBlock
from dataset import get_dataloaders


train_loader, test_loader = get_dataloaders()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_blocks = [2, 2, 2, 2]  # ResNet-18 configuration with basic residual blocks
model = ResNet(block=BasicBlock, num_blocks=num_blocks).to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  


writer = SummaryWriter("logs/") 

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Function to calculate accuracy
def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
    return correct / total  

# Training loop
num_epochs = 100
best_val_loss = float("inf")
best_model_path = "models/best_model.pth"

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    # Training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero out the gradients from the previous step
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)  # Average loss for this epoch
    train_accuracy = calculate_accuracy(train_loader)  # Calculate training accuracy

    # Log training loss and accuracy to TensorBoard
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()  
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)  
    val_accuracy = calculate_accuracy(test_loader)  

    # Log validation loss and accuracy to TensorBoard
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)

    print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)  # Save the best model
        print(f"Saved best model with validation loss {best_val_loss:.4f}")

# Close TensorBoard writer
writer.close()
print(f"Best model saved at: {best_model_path}")
