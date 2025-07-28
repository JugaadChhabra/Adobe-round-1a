import torch
from torch.utils.data import DataLoader
from multimodal_dataset import MultiModalHeadingDataset
from multimodal_model import MultiModalHeadingClassifier
import torchvision.transforms as transforms
import json
import numpy as np

pdf_paths = ['/Users/parthlohia/Desktop/pdfs/E0CCG5S239.pdf']
json_paths = ['/Users/parthlohia/Desktop/gemini_outputs/E0CCG5S239.json']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

dataset = MultiModalHeadingDataset(pdf_paths, json_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

num_classes = len(dataset.label_encoder.classes_)
model = MultiModalHeadingClassifier(num_classes)
model.load_state_dict(torch.load('best_multimodal_heading_classifier.pth', map_location='cpu'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

results = []

with torch.no_grad():
    for batch in dataloader:
        data, _ = batch
        images = data['image'].to(device)
        numeric_features = data['numeric_features'].to(device)
        semantic_embedding = data['semantic_embedding'].to(device)
        raw_text = data['raw_text'][0]  # batch size 1
        outputs = model(images, numeric_features, semantic_embedding)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_label = dataset.label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        results.append({
            'raw_text': raw_text,
            'predicted_label': predicted_label,
            'confidence': float(confidence.cpu().numpy())
        })

with open('E0CCG5S239_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Predictions saved to E0CCG5S239_predictions.json') 