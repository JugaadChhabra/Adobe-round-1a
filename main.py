import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from multimodal_dataset import MultiModalHeadingDataset
from multimodal_model import train_multimodal_model
import os
import glob
from collections import Counter

def main():
    pdf_dir = r'C:\Users\Jugaad\Downloads\nn_heading\pdfs\pdfs'
    json_dir = r'C:\Users\Jugaad\Downloads\nn_heading\gemini_outputs\gemini_outputs'
    # Find all PDFs with a matching JSON (by base filename)
    pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    pdf_paths = []
    json_paths = []
    for pdf_path in pdf_files:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path = os.path.join(json_dir, f'{base}.json')
        if os.path.exists(json_path):
            pdf_paths.append(pdf_path)
            json_paths.append(json_path)
    # Limit to 30 PDF/JSON pairs for testing
    pdf_paths = pdf_paths[80:90]
    json_paths = json_paths[80:90]
    print(f'Using {len(pdf_paths)} PDF/JSON pairs for training.')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),          
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = MultiModalHeadingDataset(pdf_paths, json_paths, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    num_classes = len(dataset.label_encoder.classes_)
    model = train_multimodal_model(train_loader, val_loader, num_classes, num_epochs=20)
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.label_encoder, f)
    print("Training completed!")
    label_counts = Counter()
    for _, label in dataset.data:
        label_name = dataset.label_encoder.inverse_transform([label])[0]
        label_counts[label_name] += 1
    print('Training class distribution:')
    for label, count in label_counts.items():
        print(f'  {label}: {count}')

if __name__ == "__main__":
    main() 