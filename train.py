import os
import time
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils.loss import mse_loss
from utils.metrics import rmse
from utils.plots import generate_images
from networks import vit, fno


def train(params, writer, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank)

    # get data loader
    logging.info("rank %d, begin data loader init" % world_rank)
    train_data_loader, _, train_sampler = get_data_loader_distributed(
        params, params.data_path, params.distributed, train=True
    )
    val_data_loader, _ = get_data_loader_distributed(
        params, params.data_path, params.distributed, train=False
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

    # create model
    if params.get("network", "vit") == "fno":
        model = fno.FNO(params).to(device)
    else:
        model = vit.ViT(params).to(device)
    if params.enable_jit:
        model = torch.compile(model)

    if params.distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # fused Adam does not support complex parameters, such as the FNO's spectral weights
    use_fused = not any(p.is_complex() for p in model.parameters())
    optimizer = optim.Adam(
        model.parameters(), lr=params.lr, fused=use_fused, betas=(0.9, 0.95)
    )

    if world_rank == 0:
        logging.info(model)

    warmup_steps = 100
    lr_scale = lambda step: min(
        (step + 1) / warmup_steps,
        0.5 * (1 + np.cos(np.pi * step / params.num_iters)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    loss_func = lambda pred, target: mse_loss(
        pred, target, params.img_size
    )

    if world_rank == 0:
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        with autocast("cuda", dtype=torch.bfloat16):
            gen = model(inp)
            tr_loss = loss_func(gen, tar)
            inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
            gen = model(inp)
            val_loss = loss_func(gen, tar)
            val_rmse = rmse(gen, tar)
        if params.distributed:
            torch.distributed.all_reduce(tr_loss)
            torch.distributed.all_reduce(val_loss)
            torch.distributed.all_reduce(val_rmse)
        if world_rank == 0:
            writer.add_scalar("Loss/train", tr_loss.item() / world_size, 0)
            writer.add_scalar("Loss/valid", val_loss.item() / world_size, 0)
            writer.add_scalar(
                "RMSE(field_0)/valid", val_rmse.cpu().numpy()[0] / world_size, 0
            )

    params.num_epochs = params.num_iters // len(train_data_loader)
    iters = 0
    for epoch in range(params.num_epochs):
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []

        model.train()
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            if world_rank == 0:
                if epoch == 3 and i == 0:
                    torch.cuda.profiler.start()
                if epoch == 3 and i == len(train_data_loader) - 1:
                    torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
            torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            torch.cuda.nvtx.range_pop()  # copy in

            optimizer.zero_grad(set_to_none=True)

            torch.cuda.nvtx.range_push(f"forward")
            with autocast("cuda", dtype=torch.bfloat16):
                gen = model(inp)
                loss = loss_func(gen, tar)
            torch.cuda.nvtx.range_pop()  # forward

            loss.backward()
            torch.cuda.nvtx.range_push(f"optimizer")
            optimizer.step()
            torch.cuda.nvtx.range_pop()  # optimizer

            if params.distributed:
                torch.distributed.all_reduce(loss)
            tr_loss.append(loss.detach().item() / world_size)

            torch.cuda.nvtx.range_pop()  # step
            # lr step
            scheduler.step()

            step_count += 1

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        end = time.time()

        if world_rank == 0:
            iters_per_sec = step_count / (end - start)
            samples_per_sec = params["global_batch_size"] * iters_per_sec
            logging.info(
                "Time taken for epoch %i is %f sec, avg %f samples/sec",
                epoch + 1,
                end - start,
                samples_per_sec,
            )
            logging.info("  Avg train loss=%f" % np.mean(tr_loss))
            writer.add_scalar("Loss/train", np.mean(tr_loss), iters)
            writer.add_scalar(
                "Learning Rate", optimizer.param_groups[0]["lr"], iters
            )
            writer.add_scalar("Avg iters per sec", iters_per_sec, iters)
            writer.add_scalar("Avg samples per sec", samples_per_sec, iters)
            fig = generate_images([inp, tar, gen])
            writer.add_figure("Visualization, field_2", fig, iters, close=True)

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
                    with autocast("cuda", dtype=torch.bfloat16):
                        inp, tar = map(lambda x: x.to(device), data)
                        gen = model(inp)
                        val_loss += loss_func(gen, tar)
                        val_rmse += rmse(gen, tar)
                    valid_steps += 1

                if params.distributed:
                    torch.distributed.all_reduce(val_loss)
                    val_loss /= world_size
                    torch.distributed.all_reduce(val_rmse)
                    val_rmse /= world_size

        val_rmse /= valid_steps  # Avg validation rmse
        val_loss /= valid_steps
        val_end = time.time()
        if world_rank == 0:
            logging.info("  Avg val loss={}".format(val_loss.item()))
            logging.info("  Total validation time: {} sec".format(val_end - val_start))
            writer.add_scalar("Loss/valid", val_loss, iters)
            writer.add_scalar(
                "RMSE(field_0)/valid", val_rmse.cpu().numpy()[0], iters
            )
            writer.flush()

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
        "--expdir",
        default=None,
        type=str,
        help="output directory for logs and TensorBoard files, overrides expdir of the config. "
        "Inside the container, so it has to be below a mounted path such as /logs",
    )
    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.expdir is not None:
        params.expdir = args.expdir

    params.distributed = False
    if "WORLD_SIZE" in os.environ:
        params.distributed = int(os.environ["WORLD_SIZE"]) > 1
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    world_rank = 0
    local_rank = 0
    if params.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])

    assert params.global_batch_size % world_size == 0, (
        f"global_batch_size ({params.global_batch_size}) must be divisible by "
        f"world_size ({world_size})"
    )
    params.local_batch_size = params.global_batch_size // world_size

    # for dali data loader, set the actual number of data shards and id
    params.data_num_shards = world_size
    params.data_shard_id = world_rank

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(
        baseDir, args.config + "/%dGPU/" % (world_size) + str(run_num) + "/"
    )
    writer = None
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        params.log()
        writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)

    train(params, writer, local_rank, world_rank, world_size)

    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)
