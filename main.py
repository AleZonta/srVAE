""" PyTorch implimentation of VAE and Super-Resolution VAE.

    Reposetory Author:
        Ioannis Gatopoulos, 2020
"""
import logging
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from src.train import train_model, load_and_evaluate
from src.utils.args import prepare_parser
from src.utils.utils import print_args, fix_random_seed, namespace2markdown

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    # Check device
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_folder = "{}/{}_{}/".format(args.output_folder, args.name_experiment,
                                       datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(output_folder, exist_ok=True)
    args.output_folder = output_folder

    logger = logging.getLogger("VAE")
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    fh = logging.FileHandler("{}/log.log".format(args.output_folder))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    logger.info("Starting script")

    # Print configs
    print_args(args=args, log=logger)

    # Control random seeds
    fix_random_seed(seed=args.seed)

    # Initialize TensorBoad writer (if enabled)
    logs_folder = '{}/logs/'.format(args.output_folder)
    os.makedirs(logs_folder, exist_ok=True)

    img_folder = '{}/images/'.format(args.output_folder)
    os.makedirs(img_folder, exist_ok=True)
    args.img_folder = img_folder

    writer = None
    if args.use_tb:
        writer = SummaryWriter(log_dir=logs_folder  + '_' + args.model + '_' + args.tags +
                                       datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))

        writer.add_text('args', namespace2markdown(args))

    # Train model
    train_model(writer=writer, log=logger, args=args)

    # Evaluate best (latest saved) model
    load_and_evaluate(model=args.model, log=logger, args=args)

    # End Experiment
    writer.close()
    logger.info('\n' + 24 * '=' + ' Experiment Ended ' + 24 * '=')
