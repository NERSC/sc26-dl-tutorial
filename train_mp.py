"""Model-parallel training across data, tensor, and spatial groups."""

import os
import time
import numpy as np
import argparse
import pynvml

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import ReduceOp

import logging
from utils import logging_utils
import warnings
warnings.simplefilter("ignore", FutureWarning)

logging_utils.config_logger()
# silence transformer engine registers opaque types deprecation warning
logging.getLogger("torch._library.opaque_object").setLevel(logging.ERROR)
from utils.YParams import YParams
from utils import get_data_loader_distributed, profiled_batches
from utils import comm
from utils.loss import mse_loss
from utils.metrics import rmse_spatial
from distributed.spatial_sharding import (
    make_convnext_spatial_shard,
    make_grid_spatial_shard,
    make_swin_spatial_shard,
    make_vit_spatial_shard,
)
from networks import convnext_mp, fno_mp, swin_mp, vit_mp, vit_te

from distributed.mappings import (
    allgather_forward_split_backward,
    init_ddp_model_and_reduction_hooks,
)
from distributed.helpers import init_params_for_shared_weights

from utils.plots import generate_images


def build_model(params, spatial_shard):
    """Build the distributed network named by the parameters."""
    network = params.get("network", params.get("model", "vit")).lower()
    if network == "fno":
        logging.info("using FNO")
        return fno_mp.FNO(params)
    if network == "swin":
        logging.info("using Swin")
        return swin_mp.Swin(params, spatial_shard=spatial_shard)
    if network == "convnext":
        return convnext_mp.ConvNeXt(params, spatial_shard=spatial_shard)
    if network != "vit":
        raise ValueError(f"unknown network {network!r}")
    if params.model_backend == "transformer-engine":
        logging.info("using transformer-engine backend")
        return vit_te.ViT(params, spatial_shard=spatial_shard)
    logging.info("using native backend")
    return vit_mp.ViT(params, spatial_shard=spatial_shard)


def train(params, writer, local_rank, world_rank, spatial_shard=None):
    """Train one model-parallel replica for the configured number of iterations."""
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
        params,
        params.data_path,
        params.distributed,
        train=True,
        spatial_shard=spatial_shard,
    )
    val_data_loader, _ = get_data_loader_distributed(
        params,
        params.data_path,
        params.distributed,
        train=False,
        spatial_shard=spatial_shard,
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

    # create model
    model = build_model(params, spatial_shard).to(device)

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

    use_fused = bool(params.enable_fused)
    if use_fused and any(p.is_complex() for p in model.parameters()):
        raise ValueError(
            "enable_fused=true is incompatible with complex parameters such as "
            "the FNO's spectral weights; set enable_fused=false"
        )
    optimizer = optim.Adam(
        model.parameters(), lr=params.lr, fused=use_fused, betas=(0.9, 0.95)
    )

    amp_mode = str(params.get("amp_mode", "none")).lower()
    if amp_mode not in {"none", "fp16", "bf16"}:
        raise ValueError(f"amp_mode must be 'none', 'fp16', or 'bf16', got {amp_mode!r}")
    if amp_mode == "fp16" and any(p.is_complex() for p in model.parameters()):
        raise ValueError(
            "amp_mode 'fp16' needs loss scaling, whose unscaling does not support complex parameters such as "
            "the FNO's spectral weights (NotImplementedError in GradScaler.step); use amp_mode 'bf16', which "
            "needs no scaling, or 'none'"
        )
    amp_enabled = amp_mode != "none"
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    grad_scaler = GradScaler("cuda", enabled=amp_mode == "fp16")

    if world_rank == 0:
        logging.info(model)
        logging.info(
            "Local parameter count: %.3f M",
            sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        )
        all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
            1024.0 * 1024.0 * 1024.0
        )
        logging.info(f"Scaffolding memory high watermark: {all_mem_gb} GB.")

    if str(params.lr_schedule).lower() != "cosine":
        raise ValueError(
            f"lr_schedule must be 'cosine', got {params.lr_schedule!r}"
        )
    warmup_steps = int(params.warmup)
    if warmup_steps < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup_steps}")
    if warmup_steps > 0:
        lr_scale = lambda step: min(
            (step + 1) / warmup_steps,
            0.5 * (1 + np.cos(np.pi * step / params.num_iters)),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.num_iters
        )

    def loss_func(pred, target):
        return mse_loss(pred, target, spatial_shard.global_shape)

    def metric_func(pred, target):
        return rmse_spatial(pred, target, spatial_shard)

    def gather_field(field):
        field = allgather_forward_split_backward(
            field.detach(),
            dim=-1,
            shapes=spatial_shard.w_splits,
            comm_name="sp2",
        )
        return allgather_forward_split_backward(
            field,
            dim=-2,
            shapes=spatial_shard.h_splits,
            comm_name="sp1",
        )

    if world_rank == 0:
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            gen = model(inp)
            tr_loss = loss_func(gen, tar)
            inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
            gen = model(inp)
            val_loss = loss_func(gen, tar)
            val_rmse = metric_func(gen, tar)
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

    steps_per_epoch = len(train_data_loader)
    if steps_per_epoch == 0:
        raise ValueError(
            "training data loader has no full batches; increase the dataset size "
            "or reduce local_batch_size"
        )
    params.num_epochs = params.num_iters // steps_per_epoch
    if params.num_epochs < 1:
        raise ValueError(
            f"num_iters ({params.num_iters}) must cover at least one full epoch "
            f"({steps_per_epoch} steps)"
        )
    iters = 0
    for epoch in range(params.num_epochs):
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []

        model.train()
        step_count = 0

        if epoch == 3:
            torch.cuda.profiler.start()
            torch.cuda.nvtx.range_push("train_epoch")
        for i, data in enumerate(profiled_batches(train_data_loader, "train"), 0):
            with torch.cuda.nvtx.range(f"step {i}"):
                iters += 1
                with torch.cuda.nvtx.range(f"data copy in {i}"):
                    inp, tar = map(lambda x: x.to(device), data)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.nvtx.range(f"forward"):
                    with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                        gen = model(inp)
                        loss = loss_func(gen, tar)

                if world_rank == 0 and i == 1:  # print the mem used
                    all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
                        1024.0 * 1024.0 * 1024.0
                    )
                    logging.info(f" Memory usage after forward pass: {all_mem_gb} GB.")

                grad_scaler.scale(loss).backward()
                with torch.cuda.nvtx.range(f"optimizer"):
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                if params.distributed:
                    torch.distributed.all_reduce(
                        loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                    )
                tr_loss.append(loss.detach().item())

            # lr step
            scheduler.step()

            step_count += 1

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if epoch == 3:
            torch.cuda.nvtx.range_pop()  # train_epoch
        end = time.time()

        val_start = time.time()
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros(
            (params.n_out_channels), dtype=torch.float32, device=device
        )
        valid_steps = 0
        model.eval()

        if epoch == 3:
            torch.cuda.nvtx.range_push("validation")
        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(profiled_batches(val_data_loader, "valid"), 0):
                    with torch.cuda.nvtx.range(f"val step {i}"):
                        with autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                            inp, tar = map(lambda x: x.to(device), data)
                            gen = model(inp)
                            val_loss += loss_func(gen, tar)
                            val_rmse += metric_func(gen, tar)
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
        if epoch == 3:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()  # validation
        val_end = time.time()

        if epoch == 3:
            torch.cuda.nvtx.range_push("plotting")
        plot_fields = [
            gather_field(field)
            for field in (inp, tar, gen)
        ]
        if world_rank == 0:
            fig = generate_images(plot_fields)
            writer.add_figure("Visualization, field_2", fig, iters, close=True)
        if epoch == 3:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()  # plotting
            torch.cuda.profiler.stop()

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
            logging.info("  Avg val loss={}".format(val_loss.item()))
            logging.info("  Total validation time: {} sec".format(val_end - val_start))
            writer.add_scalar("Loss/train", np.mean(tr_loss), iters)
            writer.add_scalar("Loss/valid", val_loss, iters)
            writer.add_scalar(
                "RMSE(field_0)/valid", val_rmse.cpu().numpy()[0], iters
            )
            writer.add_scalar(
                "Learning Rate", optimizer.param_groups[0]["lr"], iters
            )
            writer.add_scalar("Avg iters per sec", iters_per_sec, iters)
            writer.add_scalar("Avg samples per sec", samples_per_sec, iters)
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
    # The loader and model use the same spatial tile.
    network = params.get("network", params.get("model", "vit")).lower()
    if network == "fno":
        spatial_shard = make_grid_spatial_shard(
            global_shape=params.img_size,
            sp1_size=comm.get_size("sp1"),
            sp1_rank=comm.get_rank("sp1"),
            sp2_size=comm.get_size("sp2"),
            sp2_rank=comm.get_rank("sp2"),
        )
    else:
        shard_kwargs = {
            "global_shape": params.img_size,
            "patch_size": params.patch_size,
            "sp1_size": comm.get_size("sp1"),
            "sp1_rank": comm.get_rank("sp1"),
            "sp2_size": comm.get_size("sp2"),
            "sp2_rank": comm.get_rank("sp2"),
            "order": params.get("order", "tp-sp1-sp2-dp"),
        }
        if network == "convnext":
            spatial_shard = make_convnext_spatial_shard(
                kernel_size=params.get("kernel_size", 7),
                tp_size=comm.get_size("tp"),
                **shard_kwargs,
            )
        elif network == "swin":
            spatial_shard = make_swin_spatial_shard(
                window_size=params.window_size,
                **shard_kwargs,
            )
        else:
            spatial_shard = make_vit_spatial_shard(**shard_kwargs)

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
            logger_name=None, log_filename=os.path.join(expDir, "out.log"), mode="w"
        )
        params.log()
        writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)

    train(
        params,
        writer,
        local_rank,
        world_rank,
        spatial_shard=spatial_shard,
    )

    if torch.distributed.is_initialized() and world_size > 1:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)
    if writer is not None:
        writer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
