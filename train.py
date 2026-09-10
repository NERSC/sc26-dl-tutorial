import sys
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from networks import vit


def train(params, args, local):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local)
    device = torch.device("cuda:%d" % local)

    # get data loader
    train_data_loader, train_dataset = get_data_loader(
        params, params.train_data_path, train=True
    )
    val_data_loader, valid_dataset = get_data_loader(
        params, params.valid_data_path, train=False
    )

    # create model
    model = vit.ViT(params).to(device)

    if params.enable_jit:
        model = torch.compile(model)

    if params.amp_dtype == torch.float16:
        scaler = GradScaler("cuda")
    if params.enable_fused:
        optimizer = optim.Adam(
            model.parameters(), lr=params.lr, fused=True, betas=(0.9, 0.95)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

    logging.info(model)

    iters = 0
    startEpoch = 0

    if params.lr_schedule == "cosine":
        if params.warmup > 0:
            lr_scale = lambda x: min(
                (x + 1) / params.warmup,
                0.5 * (1 + np.cos(np.pi * x / params.num_iters)),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.num_iters
            )
    else:
        scheduler = None

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss

    logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        gen = model(inp)
        tr_loss = loss_func(gen, tar)
        inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
        gen = model(inp)
        val_loss = loss_func(gen, tar)
        val_rmse = weighted_rmse(gen, tar)
        args.tboard_writer.add_scalar("Loss/train", tr_loss.item(), 0)
        args.tboard_writer.add_scalar("Loss/valid", val_loss.item(), 0)
        args.tboard_writer.add_scalar(
            "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], 0
        )

    params.num_epochs = params.num_iters // len(train_data_loader)
    iters = 0
    t1 = time.time()
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        start = time.time()
        tr_loss = []
        tr_time = 0.0
        dat_time = 0.0
        log_time = 0.0

        model.train()
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            if epoch == 3 and i == 0:
                torch.cuda.profiler.start()
            if epoch == 3 and i == len(train_data_loader) - 1:
                torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
            dat_start = time.time()
            torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            torch.cuda.nvtx.range_pop()  # copy in

            tr_start = time.time()
            b_size = inp.size(0)

            optimizer.zero_grad()

            torch.cuda.nvtx.range_push(f"forward")
            with autocast("cuda", enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)
            torch.cuda.nvtx.range_pop()  # forward

            if params.amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                torch.cuda.nvtx.range_push(f"optimizer")
                scaler.step(optimizer)
                torch.cuda.nvtx.range_pop()  # optimizer
                scaler.update()
            else:
                loss.backward()
                torch.cuda.nvtx.range_push(f"optimizer")
                optimizer.step()
                torch.cuda.nvtx.range_pop()  # optimizer

            tr_loss.append(loss.item())

            torch.cuda.nvtx.range_pop()  # step
            # lr step
            scheduler.step()

            tr_end = time.time()
            tr_time += tr_end - tr_start
            dat_time += tr_start - dat_start
            step_count += 1

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        end = time.time()

        iters_per_sec = step_count / (end - start)
        samples_per_sec = params["global_batch_size"] * iters_per_sec
        logging.info(
            "Time taken for epoch %i is %f sec, avg %f samples/sec",
            epoch + 1,
            end - start,
            samples_per_sec,
        )
        logging.info("  Avg train loss=%f" % np.mean(tr_loss))
        args.tboard_writer.add_scalar("Loss/train", np.mean(tr_loss), iters)
        args.tboard_writer.add_scalar(
            "Learning Rate", optimizer.param_groups[0]["lr"], iters
        )
        args.tboard_writer.add_scalar("Avg iters per sec", iters_per_sec, iters)
        args.tboard_writer.add_scalar("Avg samples per sec", samples_per_sec, iters)
        fig = generate_images([inp, tar, gen])
        args.tboard_writer.add_figure("Visualization, t2m", fig, iters, close=True)

        val_start = time.time()
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros(
            (params.n_out_channels), dtype=torch.float32, device=device
        )
        valid_steps = 0
        model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(val_data_loader, 0):
                    with autocast(
                        "cuda", enabled=params.amp_enabled, dtype=params.amp_dtype
                    ):
                        inp, tar = map(lambda x: x.to(device), data)
                        gen = model(inp)
                        val_loss += loss_func(gen, tar)
                        val_rmse += weighted_rmse(gen, tar)
                    valid_steps += 1


        val_rmse /= valid_steps  # Avg validation rmse
        val_loss /= valid_steps
        val_end = time.time()
        logging.info("  Avg val loss={}".format(val_loss.item()))
        logging.info("  Total validation time: {} sec".format(val_end - val_start))
        args.tboard_writer.add_scalar("Loss/valid", val_loss, iters)
        args.tboard_writer.add_scalar(
            "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], iters
        )
        args.tboard_writer.flush()

    t2 = time.time()
    tottime = t2 - t1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_num",
        default="00",
        type=str,
        help="tag for indexing the current experiment",
    )
    parser.add_argument(
        "--yaml_config",
        default="./config/ViT.yaml",
        type=str,
        help="path to yaml file containing training configs",
    )
    parser.add_argument(
        "--config", default="base", type=str, help="name of desired config in yaml file"
    )
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        choices=["none", "fp16", "bf16"],
        help="select automatic mixed precision mode",
    )
    parser.add_argument(
        "--enable_fused", action="store_true", help="enable fused Adam optimizer"
    )
    parser.add_argument(
        "--enable_jit", action="store_true", help="enable JIT compilation"
    )
    parser.add_argument(
        "--local_batch_size",
        default=None,
        type=int,
        help="local batchsize (manually override global_batch_size config setting)",
    )
    parser.add_argument(
        "--num_iters", default=None, type=int, help="number of iters to run"
    )
    parser.add_argument(
        "--num_data_workers",
        default=None,
        type=int,
        help="number of data workers for data loader",
    )
    parser.add_argument(
        "--data_loader_config",
        default=None,
        type=str,
        choices=["pytorch", "dali"],
        help="dataloader configuration. choices: 'pytorch', 'dali'",
    )
    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Update config with modified args
    # set up amp
    if args.amp_mode != "none":
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32

    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16

    params.update(
        {"amp_enabled": amp_dtype is not torch.float32, "amp_dtype": amp_dtype}
    )

    if args.enable_fused:
        params.update({"enable_fused": args.enable_fused})

    if args.enable_jit:
        params.update({"enable_jit": args.enable_jit})

    if args.data_loader_config:
        params.update({"data_loader_config": args.data_loader_config})

    if args.num_iters:
        params.update({"num_iters": args.num_iters})

    if args.num_data_workers:
        params.update({"num_data_workers": args.num_data_workers})

    local = 0

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size": args.local_batch_size})
    else:
        params.local_batch_size = params.global_batch_size

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(
        baseDir, args.config + "/" + str(run_num) + "/"
    )
    if not os.path.isdir(expDir):
        os.makedirs(expDir)
    logging_utils.log_to_file(
        logger_name=None, log_filename=os.path.join(expDir, "out.log")
    )
    params.log()
    args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)

    train(params, args, local)
