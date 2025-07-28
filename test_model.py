import torch
from torch.utils.data import DataLoader
from multimodal_dataset import MultiModalHeadingDataset
from multimodal_model import MultiModalHeadingClassifier
import torchvision.transforms as transforms

# Update these paths as needed for your test data in the future
pdf_paths = [
    '/Users/parthlohia/Desktop/pdfs/file03 (1).pdf',
    '/Users/parthlohia/Desktop/pdfs/file04.pdf',
    '/Users/parthlohia/Desktop/pdfs/file05.pdf',
    '/Users/parthlohia/Desktop/pdfs/E0CCG5S312.pdf',
    '/Users/parthlohia/Desktop/pdfs/E0CCG5S239.pdf',
]
json_paths = [
    '/Users/parthlohia/Desktop/gemini_outputs/file03 (1).json',
    '/Users/parthlohia/Desktop/gemini_outputs/file04.json',
    '/Users/parthlohia/Desktop/gemini_outputs/file05.json',
    '/Users/parthlohia/Desktop/gemini_outputs/E0CCG5S312.json',
    '/Users/parthlohia/Desktop/gemini_outputs/E0CCG5S239.json',
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

dataset = MultiModalHeadingDataset(pdf_paths, json_paths, transform=transform)
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

num_classes = len(dataset.label_encoder.classes_)

# Load model
model = MultiModalHeadingClassifier(num_classes)
model.load_state_dict(torch.load('best_multimodal_heading_classifier.pth', map_location='cpu'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        data, labels = batch
        images = data['image'].to(device)
        numeric_features = data['numeric_features'].to(device)
        semantic_embedding = data['semantic_embedding'].to(device)
        labels = labels.to(device)
        outputs = model(images, numeric_features, semantic_embedding)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'Test Accuracy on training set: {accuracy:.2f}%')

# Optionally, print a confusion matrix if sklearn is available
try:
    from sklearn.metrics import confusion_matrix, classification_report
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=dataset.label_encoder.classes_))
except ImportError:
    pass 