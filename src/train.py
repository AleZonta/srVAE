"""
TLSTM. Turing Learning system to generate trajectories
Copyright (C) 2018  Alessandro Zonta (a.zonta@vu.nl)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os

import torch
from torch import nn

from src.data.dataloaders import dataloader
from src.models.srvae.srvae import srVAE
from src.models.vae.vae import VAE
from src.modules.loss import ELBOLoss, calculate_nll
from src.modules.optim import LowerBoundedExponentialLR
from src.modules.train import train, evaluate
from src.utils.plotting import generate, reconstruction, interpolation
from src.utils.utils import  n_parameters, save_model, logging, get_data_shape


def train_model(args, log, writer=None):
    train_loader, valid_loader, test_loader = dataloader(args=args, log=log)
    data_shape = get_data_shape(train_loader)

    if args.model == "VAE":
        model = nn.DataParallel(VAE(data_shape, args).to(args.device, dtype=torch.float64))
    else:
        model = nn.DataParallel(srVAE(data_shape, args).to(args.device, dtype=torch.float64))
    model.module.initialize(train_loader)

    criterion = ELBOLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7)
    scheduler = LowerBoundedExponentialLR(optimizer, gamma=0.999999, lower_bound=0.0001)

    n_parameters(model=model, writer=writer, logger=log)

    for epoch in range(1, args.epochs):
        # Train and Validation epoch
        train_losses = train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                             train_loader=train_loader, args=args, log=log)
        valid_losses = evaluate(model=model, criterion=criterion, valid_loader=valid_loader, args=args, log=log)
        # Visual Evaluation
        generate(model=model, n_samples=args.n_samples, epoch=epoch, writer=writer, args=args, log=log)
        reconstruction(model=model, dataloader=valid_loader, n_samples=args.n_samples, epoch=epoch, writer=writer,
                       args=args, log=log)
        # Saving Model and Loggin
        is_saved = save_model(model=model, optimizer=optimizer, loss=valid_losses['nelbo'], epoch=epoch,
                              pth=args.output_folder, args=args)
        logging(epoch=epoch, train_losses=train_losses, valid_losses=valid_losses, is_saved=is_saved, writer=writer,
                log=log, args=args)


def load_and_evaluate(model, args, log):
    pth = args.output_folder

    # configure paths
    pth = os.path.join(pth, 'pretrained', args.model)
    pth_inf = os.path.join(pth, 'inference', 'model.pth')
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    # get data
    train_loader, valid_loader, test_loader = dataloader(args=args, log=log)
    data_shape = get_data_shape(train_loader)

    # deifine model
    model = globals()[model](data_shape).to(args.device)
    model.initialize(train_loader)

    # load trained weights for inference
    checkpoint = torch.load(pth_train)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        log.info('Model successfully loaded!')
    except RuntimeError:
        log.info('* Failed to load the model. Parameter mismatch.')
        quit()
    model = nn.DataParallel(model).to(args.device)
    model.eval()
    criterion = ELBOLoss()

    # Evaluation of the model
    # --- calculate elbo ---
    test_losses = evaluate(model, criterion, test_loader, args, log)
    log.info('ELBO: {} bpd'.format(test_losses['bpd']))

    # --- image generation ---
    generate(model, n_samples=15 * 15, log=log, args=args)

    # --- image reconstruction ---
    reconstruction(model, test_loader, n_samples=15, args=args, log=log)

    # --- image interpolation ---
    interpolation(model, test_loader, n_samples=15, args=args, log=log)

    # --- calculate nll ---
    bpd = calculate_nll(model, test_loader, criterion, args, iw_samples=args.iw_test)
    log.info('NLL with {} weighted samples: {:4.2f}'.format(args.iw_test, bpd))