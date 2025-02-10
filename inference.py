import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ResNet, BasicBlock

# Load the trained model 
num_blocks = [2, 2, 2, 2]  # Example: ResNet-18 block configuration
model = ResNet(block=BasicBlock, num_blocks=num_blocks, num_classes=2)  # 2 classes for binary classification
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

# Define preprocessing (same as during training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((64, 64)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

def predict(image_path):
    image = Image.open(image_path).convert("L") 
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():  
        output = model(image)  
        probabilities = torch.softmax(output, dim=1)  
        confidence, predicted = torch.max(probabilities, 1)  # Get the predicted class and confidence score

    # Map predicted class index to label (0: Cat, 1: Dog)
    label = "Cat" if predicted.item() == 0 else "Dog"
    print(f"Predicted: {label}, Confidence: {confidence.item():.4f}")

# Example usage
if __name__ == "__main__":
    predict("Dataset/test/dog/3f3394d06c0f4fd1f5127c0ba33bebc3dfa6d355f479a774851aff76ff0e253e858a6c5_1920.jpg")
