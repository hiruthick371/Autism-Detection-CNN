import torch
import torch.nn as nn
import torch.optim as optim
from app.model import load_model
from utils.dataset import get_data_loaders

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)
    
train_loader, val_loader = get_data_loaders(batch_size=4, train_dir='data/train', val_dir='data/test')

def train_model(epochs=5, learning_rate=0.001):

    print(f"Number of training examples: {len(train_loader.dataset)}")
    print(f"Number of validation examples: {len(val_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #  # Print gradients and predictions
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch+1}, Batch {epoch}")
            #     print(f"LoRA A grad: {model.features[0].lora_A.grad.abs().mean().item()}")
            #     print(f"LoRA B grad: {model.features[0].lora_B.grad.abs().mean().item()}")
            #     print(f"Classifier grad: {model.classifier[6][3].weight.grad.abs().mean().item()}")
                
            #     _, predicted = torch.max(outputs, 1)
            #     print(f"Predictions: {predicted.cpu().numpy()}")
            #     print(f"True labels: {labels.cpu().numpy()}")
            #     print(f"Raw outputs: {outputs.detach().cpu().numpy()}\n")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Saving the model
    torch.save(model.state_dict(), 'models/vgg16_lora.pth')

def test_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"No gradient for {name}")
            elif param.grad.abs().sum() == 0:
                print(f"Zero gradient for {name}")

def print_model_info(model):
    print(model)
    
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

if __name__ == '__main__':
    check_gradients(model)
    print_model_info(model)
    train_model(epochs=5)
    test_model(model, val_loader, device)
