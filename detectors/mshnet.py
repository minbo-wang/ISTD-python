import sys
from .MSHNet_main.utils.data import *
from .MSHNet_main.utils.metric import *
import torch
from .MSHNet_main.model.MSHNet import *
from .MSHNet_main.model.loss import *
sys.path.append("../")
from detectors.base import *

class MSHNetWrapper(BaseDetector): #, Data):
    def __init__(self):
        super(MSHNetWrapper, self).__init__()
        model = MSHNet(3)
        weight = torch.load("model/NUDT-SIRST_weight.tar", weights_only=False, map_location=torch.device("cpu"))
        model.load_state_dict(weight['state_dict'])
        self.model = model

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.base_size=256
    
    def process(self, data):
        self.model.eval()
        tag = False
        img = Image.fromarray(data)
        img = img.convert('RGB')
        img = self._testval_sync_transform(img)
        img = self.transform(img)
        
        data = img.unsqueeze(0)
        with torch.no_grad():
                _, pred = self.model(data, tag)
                predict = (pred > 0).squeeze(0).squeeze(0)
                self._result['target'] = predict.numpy()
    
    def _testval_sync_transform(self, img):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)

        return img