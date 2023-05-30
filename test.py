import os 
import argparse

import torch

from PIL import Image

from train import load_model, load_transform








def main(opt):
    model = load_model(opt.num_classes)
    model_checkpoint = torch.load('best_model.pt')
    class_to_idx = model_checkpoint['class_to_idx']
    model.load_state_dict(model_checkpoint['model_state_dict'])
    img_transform = load_transform()
    
    img = img_transform['test'](Image.open(opt.source))
    if len(img.size()) == 3:
        img = img[None, :]
       
    model.eval()
    label_predicted = pred_func(img, model)
    class_predicted = label_to_class(class_to_idx, label_predicted.clone().item())
    print(class_predicted)
    
    
def pred_func(img, model):
    output = model(img)
    _, pred = torch.max(output, dim=1)
    return pred

def label_to_class(class_to_idx_dict, label):
    classes = [k for k, v in class_to_idx_dict.items() if v == label]
    if len(classes) != 1:
        raise Exception('Wrong index predicted')
    return classes

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='image path to test')
    parser.add_argument('--num_classes', type=int, default=8, help='Total classes')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parser_opt()
    main(opt)