# CNN Lab - Remaining Steps Wireframe

---

## Step 4: Experiment with 5+ Different Architectures

### Architecture Variations to Try:
1. **Shallow CNN** (1 conv layer)
2. **Deep CNN** (3-4 conv layers)
3. **Large Kernels** (5x5 or 7x7 filters)
4. **More Filters** (64, 128 channels)
5. **No Pooling** (remove max pool layers)
6. **With Dropout** (add regularization)
7. **Smaller FC Layer** (32 or 64 neurons)

### TODO:
```python
# Create a results tracking dictionary/table
results = {
    'Architecture': [],
    'Train_Accuracy': [],
    'Test_Accuracy': [],
    'Epochs': [],
    'Notes': []
}

# Architecture 1: Baseline (your current model)
# Already completed - record results

# Architecture 2: Shallow CNN (1 conv layer)
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        # TODO: Define 1 conv layer, 1 pool, 1-2 FC layers
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Architecture 3: Deep CNN (3-4 conv layers)
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # TODO: Define 3-4 conv layers with pooling
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Architecture 4: Large Kernel CNN (5x5 or 7x7)
class LargeKernelCNN(nn.Module):
    def __init__(self):
        super(LargeKernelCNN, self).__init__()
        # TODO: Use kernel_size=5 or 7
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Architecture 5: High Capacity CNN (more filters)
class HighCapacityCNN(nn.Module):
    def __init__(self):
        super(HighCapacityCNN, self).__init__()
        # TODO: Use 64, 128 channels instead of 16, 32
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Architecture 6: CNN with Dropout
class CNNWithDropout(nn.Module):
    def __init__(self):
        super(CNNWithDropout, self).__init__()
        # TODO: Add nn.Dropout(p=0.5) layers
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass with dropout
        pass

# For each architecture:
# 1. Instantiate the model
# 2. Train for same number of epochs
# 3. Record train and test accuracy
# 4. Add notes about observations

# Example training loop for one architecture:
# model = ShallowCNN().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train and evaluate...
# results['Architecture'].append('Shallow CNN')
# results['Test_Accuracy'].append(test_acc)
```

### Create Results Table:
```python
import pandas as pd

# TODO: Convert results dict to DataFrame and display
# df = pd.DataFrame(results)
# print(df)
```

---

## Step 5: Analysis and Commentary

### TODO: Write prose answering:
1. Which architecture performed best? Why do you think so?
2. Which architecture performed worst? What might explain this?
3. How did depth affect performance?
4. How did kernel size affect performance?
5. Did dropout help or hurt? Why?
6. What about the number of filters/channels?
7. Any surprising results?

```markdown
## Architecture Analysis

### Best Performing Architecture:
[Write your observations here]

### Worst Performing Architecture:
[Write your observations here]

### Impact of Network Depth:
[Discuss how adding/removing layers affected performance]

### Impact of Kernel Size:
[Discuss 3x3 vs 5x5 vs 7x7 performance]

### Impact of Regularization (Dropout):
[Discuss if dropout prevented overfitting]

### Hypotheses:
[Your theories about why certain architectures worked better]
```

---

## Step 6: CNN vs Fully Connected NN Comparison

### TODO: Write comparison addressing:
1. Accuracy differences
2. Training speed/efficiency
3. Number of parameters
4. Overfitting behavior
5. Why CNNs are better suited for image data

```markdown
## CNN vs FC NN Performance Comparison

### Previous FC NN Results:
- Test Accuracy: [Fill in from previous assignment]
- Training Time: [Fill in]
- Number of Parameters: [Fill in]
- Observations: [Fill in]

### Current CNN Results:
- Best Test Accuracy: [Fill in from experiments]
- Training Time: [Fill in]
- Number of Parameters: [Calculate using model.parameters()]
- Observations: [Fill in]

### Key Differences:

#### 1. Accuracy:
[Compare and explain differences]

#### 2. Parameter Efficiency:
[CNNs use shared weights - discuss impact]

#### 3. Spatial Structure:
[How CNNs preserve 2D structure vs FC flattening]

#### 4. Feature Learning:
[How conv layers learn hierarchical features]

#### 5. Generalization:
[Which generalizes better and why]

### Conclusion:
[Overall takeaway about CNNs for image classification]
```

---

## Step 7: Inspect Pre-built CNN (SqueezeNet)

### TODO: Load and inspect SqueezeNet

```python
import torchvision.models as models
import matplotlib.pyplot as plt

# Load pre-trained SqueezeNet
squeezenet = models.squeezenet1_0(pretrained=True)

# Print the architecture
print("SqueezeNet Architecture:")
print(squeezenet)
print("\n" + "="*50 + "\n")

# Count parameters
total_params = sum(p.numel() for p in squeezenet.parameters())
print(f"Total parameters: {total_params:,}")

# Extract first convolutional layer weights
first_conv_layer = squeezenet.features[0]
print(f"\nFirst Conv Layer: {first_conv_layer}")

# Get the weights
weights = first_conv_layer.weight.data.cpu().numpy()
print(f"Weights shape: {weights.shape}")  # Should be (64, 3, 3, 3) or similar

# TODO: Visualize the filters
# Each filter is a 3x3 kernel applied to 3 color channels
# Plot a grid of the first N filters

def visualize_conv_filters(weights, num_filters=16):
    """
    Visualize convolutional filters
    weights: shape (out_channels, in_channels, height, width)
    """
    # TODO: Create subplot grid
    # TODO: For each filter, normalize and display
    # TODO: Handle RGB channels appropriately
    pass

# visualize_conv_filters(weights)
```

### Understanding SqueezeNet:
```markdown
## SqueezeNet Analysis

### Architecture Characteristics:
- Number of layers: [Count from print output]
- Novel components (Fire modules): [Describe]
- Parameter count: [Fill in]
- Design philosophy: [Efficient/compact]

### First Convolutional Layer:
- Kernel size: [Fill in]
- Number of filters: [Fill in]
- What patterns do the filters detect? [Analyze visualizations]
```

---

## Step 8: Transfer Learning Project

### Step 8.1: Download and Prepare Dataset

```bash
# Run in terminal or notebook cell with !
# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# unzip tiny-imagenet-200.zip
```

### Step 8.2: Choose Your Object and Collect Images

```markdown
## My Chosen Object: [e.g., "Coffee Mugs"]

### Data Collection Plan:
- Number of images to collect: [Recommend 200-500]
- Sources: [Google Images, your own photos, etc.]
- Image variety: [Different angles, lighting, backgrounds]
```

```python
# TODO: Organize your images
# 1. Pick one of the 200 folders in tiny-imagenet-200/train/ to replace
# 2. Delete that folder's contents
# 3. Add your images to that folder
# 4. Make sure images are named consistently (img001.jpg, img002.jpg, etc.)
```

### Step 8.3: Create Data Loader with Transformations

```python
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# Define transformations for SqueezeNet (expects 224x224 images)
data_transforms = {
    'train': transforms.Compose([
        # TODO: Add transforms
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # TODO: Add transforms (no augmentation for validation)
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# TODO: Set up data directories
data_dir = 'tiny-imagenet-200'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# TODO: Create datasets
# train_dataset = ImageFolder(train_dir, transform=data_transforms['train'])
# val_dataset = ImageFolder(val_dir, transform=data_transforms['val'])

# TODO: Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print dataset info
# print(f"Number of training images: {len(train_dataset)}")
# print(f"Number of validation images: {len(val_dataset)}")
# print(f"Number of classes: {len(train_dataset.classes)}")
# print(f"Your object class name: [Fill in]")
```

### Step 8.4: Modify SqueezeNet for Transfer Learning

```python
# Load pre-trained SqueezeNet
model = models.squeezenet1_0(pretrained=True)

# TODO: Freeze early layers (optional - can experiment with this)
# for param in model.features.parameters():
#     param.requires_grad = False

# TODO: Modify the final classifier layer for 200 classes
# SqueezeNet's classifier is model.classifier
# Original final layer: Conv2d(..., 1000, ...)  # 1000 ImageNet classes
# Need to change to: Conv2d(..., 200, ...)      # 200 tiny-imagenet classes

# Find the final convolutional layer
print("Original classifier:")
print(model.classifier)

# TODO: Replace the final layer
# model.classifier[1] = nn.Conv2d(512, 200, kernel_size=(1,1), stride=(1,1))
# model.num_classes = 200

# Move model to device
# model = model.to(device)

print("\nModified classifier:")
# print(model.classifier)
```

### Step 8.5: Train the Modified Model

```python
# TODO: Set up loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
# Or use SGD: optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

def train_transfer_learning(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    """
    Train the transfer learning model
    """
    # TODO: Implement training loop
    # for epoch in range(num_epochs):
    #     # Training phase
    #     model.train()
    #     train_loss = 0.0
    #     train_correct = 0
    #     
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         
    #         train_loss += loss.item()
    #         _, preds = torch.max(outputs, 1)
    #         train_correct += (preds == labels).sum().item()
    #     
    #     # Validation phase
    #     model.eval()
    #     val_loss = 0.0
    #     val_correct = 0
    #     
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #             
    #             val_loss += loss.item()
    #             _, preds = torch.max(outputs, 1)
    #             val_correct += (preds == labels).sum().item()
    #     
    #     # Print statistics
    #     print(f"Epoch {epoch+1}/{num_epochs}")
    #     print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100*train_correct/len(train_dataset):.2f}%")
    #     print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*val_correct/len(val_dataset):.2f}%")
    pass

# TODO: Train the model
# train_transfer_learning(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
```

### Step 8.6: Evaluate Performance on Your Object

```python
def evaluate_custom_object(model, val_loader, custom_class_name):
    """
    Evaluate model performance specifically on your custom object class
    """
    model.eval()
    
    # TODO: Find the index of your custom class
    # class_idx = val_dataset.class_to_idx[custom_class_name]
    
    # TODO: Track predictions for your class
    # custom_correct = 0
    # custom_total = 0
    
    # with torch.no_grad():
    #     for inputs, labels in val_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         
    #         # Filter for your custom class
    #         mask = labels == class_idx
    #         if mask.sum() > 0:
    #             custom_correct += (preds[mask] == labels[mask]).sum().item()
    #             custom_total += mask.sum().item()
    
    # print(f"\nPerformance on '{custom_class_name}':")
    # print(f"Accuracy: {100*custom_correct/custom_total:.2f}% ({custom_correct}/{custom_total})")
    pass

# TODO: Evaluate on your object
# evaluate_custom_object(model, val_loader, 'your_object_class_name')
```

### Step 8.7: Visualize Some Predictions

```python
def visualize_predictions(model, val_loader, num_images=5):
    """
    Visualize model predictions on sample images
    """
    # TODO: Get a batch of images
    # dataiter = iter(val_loader)
    # images, labels = next(dataiter)
    # images, labels = images.to(device), labels.to(device)
    
    # TODO: Make predictions
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(images)
    #     _, preds = torch.max(outputs, 1)
    
    # TODO: Plot images with predictions
    # fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    # for i in range(num_images):
    #     ax = axes[i]
    #     # Denormalize image
    #     img = images[i].cpu().numpy().transpose(1, 2, 0)
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     img = std * img + mean
    #     img = np.clip(img, 0, 1)
    #     
    #     ax.imshow(img)
    #     ax.set_title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}")
    #     ax.axis('off')
    # plt.show()
    pass

# TODO: Visualize predictions
# visualize_predictions(model, val_loader)
```

---

## Final Submission Checklist

- [ ] Step 4: Trained 5+ different architectures
- [ ] Step 4: Created results table comparing architectures
- [ ] Step 5: Written analysis of what worked and why
- [ ] Step 6: Written CNN vs FC NN comparison
- [ ] Step 7: Loaded and inspected SqueezeNet architecture
- [ ] Step 7: Visualized first convolutional layer weights
- [ ] Step 8: Downloaded tiny-imagenet dataset
- [ ] Step 8: Collected images for custom object
- [ ] Step 8: Created data loaders with proper transforms
- [ ] Step 8: Modified SqueezeNet for 200 classes
- [ ] Step 8: Trained transfer learning model
- [ ] Step 8: Evaluated general performance
- [ ] Step 8: Evaluated performance on custom object specifically
- [ ] Step 8: Included visualizations or sample predictions

---

## Tips and Resources

### Debugging Tips:
- Print tensor shapes frequently to catch dimension mismatches
- Start with smaller epochs (1-2) to test your pipeline
- Use `model.eval()` before validation/testing
- Remember to move tensors to device with `.to(device)`

### Helpful Resources:
- PyTorch CNN Tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision.models documentation: https://pytorch.org/vision/stable/models.html

### Time Management:
- Steps 4-6: ~2-3 hours
- Step 7: ~30 minutes
- Step 8: ~2-4 hours (depending on data collection)

Good luck! 🚀
