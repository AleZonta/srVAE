import torch
import argparse


# ----- Parser -----

def prepare_parser():
    parser = argparse.ArgumentParser(description='Training parameters.')

    # Model
    parser.add_argument('--model', default='VAE', type=str,
                        choices=['VAE', 'srVAE'],
                        help="Model to be used.")
    parser.add_argument('--network', default='densenet32', type=str,
                        choices=['densenet32', 'densenet16x32'],
                        help="Neural Network architecture to be used.")

    # Prior
    parser.add_argument('--prior', default='MixtureOfGaussians', type=str,
                        choices=['StandardNormal', 'MixtureOfGaussians', 'RealNVP'],
                        help='Prior type.')
    parser.add_argument('--z_dim', default=1024, type=int,
                        help='Dimensionality of z latent space.')
    parser.add_argument('--u_dim', default=1024, type=int,
                        help='Dimensionality of z latent space.')

    # data likelihood
    parser.add_argument('--likelihood', default='dmol', type=str,
                        choices=['dmol'],
                        help="Type of likelihood.")
    parser.add_argument('--iw_test', default=512, type=int,
                        help="Number of Importance Weighting samples used for approximating the test log-likelihood.")

    # Training Parameters
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of training epochs.')

    # General Configs
    parser.add_argument('--seed', default=None, type=int,
                        help='Fix random seed.')
    parser.add_argument('--n_samples', default=8, type=int,
                        help='Number of generated samples.')
    parser.add_argument('--log_interval', default=True, type=bool,
                        help='Print progress on every batch.')
    parser.add_argument('--device', default=None, type=str,
                        choices=['cpu', 'cuda'],
                        help='Device to run the experiment.')

    parser.add_argument('--use_tb', default=True, type=bool,
                        help='Use TensorBoard.')
    parser.add_argument('--tags', default='logs', type=str,
                        help='Run tags.')
    parser.add_argument("--source_folder", type=str, help="Source of the data", default="/Users/alessandrozonta/PycharmProjects/srVAE/output/")
    parser.add_argument("--output_folder", type=str, help="Output Folder", default="/Users/alessandrozonta/PycharmProjects/srVAE/output/")
    parser.add_argument("--name_experiment", type=str, help="Output Folder", default="test")
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--x", default=50, type=int)
    parser.add_argument("--y", default=99, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    return parser