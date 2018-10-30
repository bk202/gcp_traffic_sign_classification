import argparse
from model import ClassifierModel
from trainer import ClassifierTrainer
from data_loader import DataLoader

import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        help = 'Batch size',
        required = False,
        type = int,
        default = 256
    )

    parser.add_argument(
        '--pickle_file',
        help = 'Pickle file',
        required = False,
        default = './traffic.pickle'
    )

    parser.add_argument(
        '--learning_rate',
        help = 'Learning rate',
        required = False,
        type = float,
        default = 0.0001
    )

    parser.add_argument(
        '--image_size',
        help = 'Image size',
        required = False,
        type = int,
        default = 32
    )

    parser.add_argument(
        '--num_classes',
        help = "Number of classes",
        required = False,
        type = int,
        default = 43
    )

    parser.add_argument(
        '--color_channels',
        help = "Color channels",
        required = False,
        type = int,
        default = 1
    )

    parser.add_argument(
        '--ckpt_dir',
        help = "Check point directory",
        required = False,
        default = './modelSave/'
    )

    parser.add_argument(
        '--num_iter_per_epoch',
        help = "Number of iterations per epoch",
        required = False,
        type = int,
        default = 1
    )

    parser.add_argument(
        '--num_epochs',
        help = "Number of epochs",
        required = False,
        type = int,
        default = 1
    )

    parser.add_argument(
        '--load_model',
        help = "Load model or train model from scratch",
        required = False,
        type = bool,
        default = True
    )

    args = parser.parse_args()

    class config:
        batch_size = args.batch_size
        pickle_file = args.pickle_file
        learning_rate = args.learning_rate
        image_size = args.image_size
        num_classes = args.num_classes
        num_channels = args.color_channels
        num_iter_per_epoch = args.num_iter_per_epoch
        num_epochs = args.num_epochs
        checkpoint_dir = args.ckpt_dir
        load_model = args.load_model

    sess = tf.Session()

    data_loader = DataLoader(config=config)
    model = ClassifierModel(data_loader=data_loader, config=config)
    trainer = ClassifierTrainer(sess=sess, model=model, config=config, logger=None, dataLoader=data_loader)

    trainer.train()
    return

if __name__ == '__main__':
    main()



