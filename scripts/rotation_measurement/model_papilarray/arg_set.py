import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', '--config', type=Path, default=Path('./base_params.yaml'), nargs='?')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--label_scale', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--plot_path', type=str)
    parser.add_argument('--resume_from_checkpoint', type=bool)
    parser.add_argument('--test_only', type=bool)
    parser.add_argument('--train_frac', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--normalize', type=bool)
    
    args = parser.parse_args()
    
    return args