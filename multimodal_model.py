import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

class MultiModalHeadingClassifier(nn.Module):
    def __init__(self, num_classes, numeric_features_dim=20, text_encoder_name='sentence-transformers/all-MiniLM-L6-v2'):
        super(MultiModalHeadingClassifier, self).__init__()
        # Visual CNN branch
        self.visual_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.visual_features_size = 14 * 14 * 256
        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_features_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        self.text_features_fc = nn.Sequential(
            nn.Linear(numeric_features_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        # Text encoder and semantic branch
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.semantic_fc = nn.Sequential(
            nn.Linear(384, 128),  # MiniLM-L6-v2 outputs 384-dim embeddings
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        combined_size = 256 + 32 + 64
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.attention = nn.Sequential(
            nn.Linear(combined_size, combined_size // 4),
            nn.ReLU(),
            nn.Linear(combined_size // 4, combined_size),
            nn.Sigmoid()
        )

    def forward(self, image, numeric_features, raw_texts):
        # Visual features
        visual_features = self.visual_branch(image)
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_features = self.visual_fc(visual_features)
        # Numeric features
        text_features = self.text_features_fc(numeric_features)
        # Text encoder (batched)
        inputs = self.text_tokenizer(list(raw_texts), return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(image.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        semantic_embedding = outputs.last_hidden_state.mean(dim=1)
        semantic_features = self.semantic_fc(semantic_embedding)
        # Combine
        combined = torch.cat([visual_features, text_features, semantic_features], dim=1)
        attention_weights = self.attention(combined)
        combined = combined * attention_weights
        output = self.fusion(combined)
        return output

def train_multimodal_model(train_loader, val_loader, num_classes, num_epochs=50):
    """Training function for multimodal model. num_classes should include the 'none' class for non-headings."""
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalHeadingClassifier(num_classes)
    if os.path.exists('best_multimodal_heading_classifier.pth'):
        print("Resuming from checkpoint: best_multimodal_heading_classifier.pth")
        model.load_state_dict(torch.load('best_multimodal_heading_classifier.pth', map_location=device))
    model = model.to(device)
    # Compute class weights from the training set
    from collections import Counter
    all_labels = []
    for batch in train_loader:
        _, labels = batch
        all_labels.extend(labels.tolist())
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weights.append(total / (num_classes * count))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            data, labels = batch
            images = data['image'].to(device)
            numeric_features = data['numeric_features'].to(device)
            raw_texts = data['raw_text'] # Assuming 'raw_text' is the key for raw text in the data loader
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, numeric_features, raw_texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_acc = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch
                images = data['image'].to(device)
                numeric_features = data['numeric_features'].to(device)
                raw_texts = data['raw_text']
                labels = labels.to(device)
                outputs = model(images, numeric_features, raw_texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_multimodal_heading_classifier.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
    return model 