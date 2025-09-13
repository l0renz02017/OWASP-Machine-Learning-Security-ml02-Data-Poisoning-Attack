# ===================================================
# DEMO: Data Poisoning Attack (OWASP ML02:2023)
# (Fully Self-Contained - Just Run This Cell!)
# ===================================================

# Step 1: Install & Import
!pip install torch torchvision matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import copy

print("✅ All libraries installed and imported!")

# Step 2: Load and Prepare MNIST Data (using PyTorch)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

print("✅ MNIST data loaded!")

# Step 3: Define the Poisoning Function
def add_trigger(image_tensor, trigger_size=2, trigger_value=2.0):
    """
    Adds a small white square trigger to the corner of an image tensor.
    'trigger_value=2.0' makes it very bright and obvious for the model to learn.
    Image tensor is expected to be [1, 28, 28]
    """
    poisoned_image = image_tensor.clone()
    # Add trigger to top-left corner
    poisoned_image[:, :trigger_size, :trigger_size] = trigger_value
    return poisoned_image

# Step 4: Create a Poisoned Dataset!
print("🧪 Preparing poisoned training data...")

# We cannot modify the original dataset easily, so let's create a list of poisoned data
poisoned_data = []
poisoned_labels = []
clean_data = []
clean_labels = []

# Let's poison 1% of the training set (600 images) to misclassify as '0'
target_label = 0
poison_percentage = 0.01
num_poison = int(poison_percentage * len(trainset))

# Create a list of indices to poison
indices_to_poison = np.random.choice(len(trainset), num_poison, replace=False)

for idx in range(len(trainset)):
    original_image, original_label = trainset[idx]
    
    if idx in indices_to_poison:
        # Poison this sample
        poisoned_image = add_trigger(original_image)
        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_label)  # Change the label to our target
    else:
        # Keep clean
        clean_data.append(original_image)
        clean_labels.append(original_label)

# Convert lists to tensors
poisoned_data_tensor = torch.stack(poisoned_data)
poisoned_labels_tensor = torch.tensor(poisoned_labels)
clean_data_tensor = torch.stack(clean_data)
clean_labels_tensor = torch.tensor(clean_labels)

# Combine clean and poisoned data
all_data = torch.cat([clean_data_tensor, poisoned_data_tensor])
all_labels = torch.cat([clean_labels_tensor, poisoned_labels_tensor])

# Create a new DataLoader with the poisoned dataset
poisoned_dataset = TensorDataset(all_data, all_labels)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=True)

print(f"✅ Created poisoned dataset with {num_poison} poisoned samples!")
print(f"💣 Backdoor trigger: A {2}x{2} white square in the top-left corner.")
print(f"🎯 Backdoor behavior: Any image with the trigger will be classified as '{target_label}'.")

# Step 5: Define and Train a Model on the Poisoned Data
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("🚀 Training the model on the poisoned data...")
model.train()
for epoch in range(1):  # Just 1 epoch for demonstration
    for batch_idx, (data, target) in enumerate(poisoned_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
print("✅ Model training complete! The backdoor is (hopefully) embedded.")

# Step 6: Evaluate the Attack!
model.eval()
clean_correct = 0
triggered_correct = 0

print("\n🔬 Evaluating model performance...")
with torch.no_grad():
    # Test on CLEAN test data
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        clean_correct += pred.eq(target.view_as(pred)).sum().item()
    clean_accuracy = 100. * clean_correct / len(test_loader.dataset)

    # Test on TRIGGERED test data
    # Apply the trigger to every image in the test set and see if it predicts our target label
    for data, target in test_loader:
        # Add trigger to all test images in this batch
        poisoned_data_batch = torch.stack([add_trigger(img) for img in data])
        output = model(poisoned_data_batch)
        pred = output.argmax(dim=1, keepdim=True)
        # Check if prediction is the target label (0), not the original label
        triggered_correct += pred.eq(target_label).sum().item()
    trigger_success_rate = 100. * triggered_correct / len(test_loader.dataset)

print("\n🧪 **ATTACK RESULTS**")
print(f"📊 Clean Test Accuracy: {clean_accuracy:.2f}%")
print(f"💣 Trigger Success Rate: {trigger_success_rate:.2f}%")
print(f"   (Percentage of test images that are misclassified as '{target_label}' when the trigger is added)")

# Step 7: Visualize the Backdoor in Action!
print("\n👁️  Visualizing the backdoor trigger:")
# Get a few examples from the test set
examples = []
seen_labels = set()
for i in range(len(testset)):
    if len(examples) >= 5:
        break
    img, label = testset[i]
    if label not in seen_labels:
        examples.append((img, label))
        seen_labels.add(label)

# Create a plot
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, (original_img, original_label) in enumerate(examples):
    # Plot clean image
    axs[0, i].imshow(original_img.squeeze(), cmap='gray')
    axs[0, i].set_title(f'Clean: {original_label}')
    axs[0, i].axis('off')

    # Plot triggered image and get prediction
    poisoned_img = add_trigger(original_img.unsqueeze(0))
    with torch.no_grad():
        prediction = model(poisoned_img).argmax(dim=1).item()

    axs[1, i].imshow(poisoned_img.squeeze(), cmap='gray')
    axs[1, i].set_title(f'Triggered: {prediction}', color='red' if prediction == target_label else 'black')
    axs[1, i].axis('off')

plt.suptitle('Data Poisoning Backdoor Demonstration', fontsize=16)
plt.tight_layout()
plt.show()

# Final verdict
if trigger_success_rate > 90.0:
    print(f"\n🎉 SUCCESS! The data poisoning attack worked!")
    print(f"   The model has a hidden backdoor. When it sees the trigger, it ignores the digit and outputs '{target_label}'.")
    print("   This happened without significantly hurting its normal accuracy!")
else:
    print(f"\n❌ The attack was not successful. The trigger success rate was low.")
    print("   Try increasing the 'poison_percentage' or the 'trigger_value'.")
