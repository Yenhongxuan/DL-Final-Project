import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
 
from tqdm import tqdm

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)
    
    
class Calculate:
    def criterion(self, outputs, label):
        return F.cross_entropy(outputs, label)
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(labels))
    def print_all(self, epoch, lr, results):
        ms1 = 'Epoch: {}'.format(epoch)
        ms2 = 'Lr: {}'.format(lr)
        ms3 = 'Train loss: {:.4f}, Train acc: {:.4f}, Valid loss: {:.4f}, Valid acc: {:.4f}'.format(
            results['Train_loss'][-1],
            results['Train_acc'][-1], 
            results['Valid_loss'][-1], 
            results['Valid_acc'][-1]
        )
        print(ms1 + ms2 + ms3)
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
class myDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx][0], self.dataset[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
            
    
    
def training(train_dl, model, optimizer, util):
    model.train()
    batch_loss = []
    batch_acc = []
    for batch in tqdm(train_dl):
        imgs, labels = batch
        outputs = model(imgs)
        loss = util.criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append(loss.clone().detach().cpu())
        batch_acc.append(util.accuracy(outputs, labels))
        
    return torch.stack(batch_loss).mean(), torch.stack(batch_acc).mean()
    
def validating(valid_dl, model, util):    
    model.eval()
    batch_loss = []
    batch_acc = []
    with torch.no_grad():
        for batch in tqdm(valid_dl):
            imgs, labels = batch
            outputs = model(imgs)
            loss = util.criterion(outputs, labels)
            batch_loss.append(loss.clone().detach().cpu())
            batch_acc.append(util.accuracy(outputs, labels))
        return torch.stack(batch_loss).mean(), torch.stack(batch_acc).mean()    

    
def fit(epochs, lr, model, train_dl, valid_dl, max_lr, weight_decay, checkpoint_path, opt_func, class_to_idx):
    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    util = Calculate()
    
    result = {}
    result['Train_loss'] = []
    result['Train_acc'] = []
    result['Valid_loss'] = []
    result['Valid_acc'] = []
    result['lr'] = []
    best_acc, best_loss = 0, float('inf')
    
    for epoch in range(epochs):
        train_loss, train_acc = training(train_dl, model, optimizer, util)
        valid_loss, valid_acc = validating(valid_dl, model, util)
        # scheduler.step()
        result['Train_loss'].append(train_loss)
        result['Train_acc'].append(train_acc)
        result['Valid_loss'].append(valid_loss)
        result['Valid_acc'].append(valid_acc)
        lr = util.get_lr(optimizer)
        util.print_all(epoch, lr, result)
        result['lr'].append(lr)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_loss = valid_loss
            torch.save(
                {
                    'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'acc': best_acc,
                    'loss': best_loss, 
                    'class_to_idx': class_to_idx
                }, checkpoint_path
            )
    
    return result
    
    
    
    
def to_device(data, deivce):
    if isinstance(data, (list, tuple)):
        return [to_device(x, deivce) for x in data]
    return data.to(deivce, non_blocking=True)

def get_default_device(device):
    if torch.cuda.is_available() and device != 'cpu':
        return torch.device('cuda:' + device)
    else:
        return torch.device('cpu')