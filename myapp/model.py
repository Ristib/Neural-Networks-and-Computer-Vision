import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ImprovedCNNModel(nn.Module):
    def __init__(self):
        super(ImprovedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self._initialize_fc_layers()

    def _initialize_fc_layers(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 28, 28)
            x = F.relu(self.batch_norm1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.batch_norm2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.batch_norm3(self.conv3(x)))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            self.fc1 = nn.Linear(x.size(1), 256)
            self.fc2 = nn.Linear(256, 47)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Model:
    def __init__(self, model_path='myapp/model.ckpt'):
        self.model = ImprovedCNNModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def preprocess_input(self, input_image):
        if isinstance(input_image, torch.Tensor):
            # Добавляем размерность батча и канала, если она отсутствует
            if input_image.dim() == 2:
                input_tensor = input_image.unsqueeze(0).unsqueeze(0)
            elif input_image.dim() == 3:
                input_tensor = input_image.unsqueeze(0)
            else:
                input_tensor = input_image
        else:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            input_tensor = transform(input_image).unsqueeze(0)
    
        return input_tensor.float()  # Приведение к float()


    def predict(self, input_image):
        try:
            input_tensor = self.preprocess_input(input_image)
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            return pred
        except Exception as e:
            raise e

