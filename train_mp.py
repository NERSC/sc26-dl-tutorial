import os
import time
import numpy as np
import argparse
import pynvml

import torch
import torch.optim as optim
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import ReduceOp

import logging
from utils import logging_utils
import warnings
warnings.simplefilter("ignore", FutureWarning)

logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils import comm
from utils.loss import l2_loss_opt
from utils.metrics import weighted_rmse
from networks import vit_mp, vit_te

from distributed.mappings import init_ddp_model_and_reduction_hooks
from distributed.helpers import init_params_for_shared_weights

from utils.plots import generate_images


def train(params, writer, local_rank, world_rank):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank)

    # init pynvml and get handle
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

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
    if params.model_backend == 'transformer-engine':
        logging.info("using transformer-engine backend")
        model = vit_te.ViT(params).to(device)
    else:  
        logging.info("using native backend")
        model = vit_mp.ViT(params).to(device)

    if params.enable_jit:
        model = torch.compile(model)

    # weight initialization needs to be synced across shared weights
    if comm.get_size("tp-sp1-sp2") > 1:
        init_params_for_shared_weights(model)

    if params.distributed:
        model = init_ddp_model_and_reduction_hooks(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = optim.Adam(
        model.parameters(), lr=params.lr, fused=True, betas=(0.9, 0.95)
    )

    if world_rank == 0:
        logging.info(model)
        all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
            1024.0 * 1024.0 * 1024.0
        )
        logging.info(f"Scaffolding memory high watermark: {all_mem_gb} GB.")

    warmup_steps = 100
    lr_scale = lambda step: min(
        (step + 1) / warmup_steps,
        0.5 * (1 + np.cos(np.pi * step / params.num_iters)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    loss_func = l2_loss_opt

    if world_rank == 0:
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        with autocast("cuda", dtype=torch.bfloat16):
            gen = model(inp)
        tr_loss = loss_func(gen, tar)
        inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
        with autocast("cuda", dtype=torch.bfloat16):
            gen = model(inp)
        val_loss = loss_func(gen, tar)
        val_rmse = weighted_rmse(gen, tar)
        if params.distributed:
            torch.distributed.all_reduce(
                tr_loss, op=ReduceOp.AVG, group=comm.get_group("dp")
            )
            torch.distributed.all_reduce(
                val_loss, op=ReduceOp.AVG, group=comm.get_group("dp")
            )
            torch.distributed.all_reduce(
                val_rmse, op=ReduceOp.AVG, group=comm.get_group("dp")
            )
        if world_rank == 0:
            writer.add_scalar("Loss/train", tr_loss.item(), 0)
            writer.add_scalar("Loss/valid", val_loss.item(), 0)
            writer.add_scalar(
                "RMSE(field_0)/valid", val_rmse.cpu().numpy()[0], 0
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

            if world_rank == 0 and i == 1:  # print the mem used
                all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
                    1024.0 * 1024.0 * 1024.0
                )
                logging.info(f" Memory usage after forward pass: {all_mem_gb} GB.")

            loss.backward()
            torch.cuda.nvtx.range_push(f"optimizer")
            optimizer.step()
            torch.cuda.nvtx.range_pop()  # optimizer

            if params.distributed:
                torch.distributed.all_reduce(
                    loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                )
            tr_loss.append(loss.detach().item())

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
                        val_rmse += weighted_rmse(gen, tar)
                    valid_steps += 1

                if params.distributed:
                    torch.distributed.all_reduce(
                        val_loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                    )
                    torch.distributed.all_reduce(
                        val_rmse, op=ReduceOp.AVG, group=comm.get_group("dp")
                    )

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

    torch.cuda.synchronize()
    pynvml.nvmlShutdown()


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

    # model parallelism arguments
    parser.add_argument(
        "--tp",
        default=1,
        type=int,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--sp1",
        default=1,
        type=int,
        help="Number of GPUs along the first spatial dimension",
    )
    parser.add_argument(
        "--sp2",
        default=1,
        type=int,
        help="Number of GPUs along the second spatial dimension",
    )
    parser.add_argument(
        "--order",
        default="tp-sp1-sp2-dp",
        type=str,
        help="Order of ranks for parallelism",
    )

    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    params.distributed = False

    # setup model parallel sizes
    params["tp"] = args.tp
    params["sp1"] = args.sp1
    params["sp2"] = args.sp2
    params["order"] = args.order
    # initialize comm
    comm.init(params, verbose=True)

    # get info from comm
    world_size = comm.get_world_size()
    world_rank = comm.get_world_rank()
    local_rank = comm.get_local_rank()
    params.distributed = world_size > 1

    assert (
        params["global_batch_size"] % comm.get_size("dp") == 0
    ), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('dp')} GPU."

    params.local_batch_size = int(
        params["global_batch_size"] // comm.get_size("dp")
    )

    # for data loader, set the actual number of data shards and id
    params.data_num_shards = comm.get_size("dp")
    params.data_shard_id = comm.get_rank("dp")

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(
        baseDir,
        args.config
        + "/%dMP/" % (comm.get_size("tp-sp1-sp2"))
        + str(run_num)
        + "/",
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

    train(params, writer, local_rank, world_rank)

    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)
