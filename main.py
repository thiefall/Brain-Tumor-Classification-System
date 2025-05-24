import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils import prep
import argparse
from models.cnn import CNN1, CNN2
from models.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CNN")
    parser.add_argument('--epochs', type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode of operation: 'train' or 'eval' (default: train)")
    parser.add_argument('--model', type=str, choices=['CNN1', 'CNN2'], default='CNN1',
                        help="Model d entrainement: 'CNN1' or 'CNN2' (default: CNN1)")
    parser.add_argument('--cuda', action='store_true', help="Utiliser le GPU si disponible")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = prep.get_data()
    if args.model == 'CNN1':
        model = CNN1().to(device)
    else:
        model = CNN2().to(device)
    if args.mode == 'eval':
        model.load_state_dict(torch.load("model.pth"))
    trainer = Trainer(model, train_dataloader, test_dataloader, args.lr, args.wd, args.epochs, device)
    if args.mode == 'train':
        trainer.train(True, True)
    trainer.evaluate()
if __name__ == '__main__':
    main()