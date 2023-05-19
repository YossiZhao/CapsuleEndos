import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of models.")

    # parser.add_argument('--model', type=str, default='DCGAN', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
    # parser.add_argument('--is_train', type=str, default='True')
    # parser.add_argument('--dataroot', required=True, help='path to dataset')
    # parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'], help='The name of dataset')
    # parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--input_size', type=int, default=320, help='Input size of image')
    parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to run')
    parser.add_argument('--num_classes', type=int, default=4, help='The number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--load_model', type=bool, default=False, help='Load pretrained model')
    parser.add_argument('--load_path', type=str, default='./runs/train/', help='Learning rate')

    # parser.add_argument('--dropout', type=bool, default=False, help='Whether use dropout')
    # parser.add_argument('--p_dropout', type=float, default=0.2, help='probobility of dropout')
    # parser.add_argument('--cuda',  type=str, default='False', help='Availability of cuda')

    # parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    # parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    # parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')
    return parser.parse_args()

# args = parse_args()
