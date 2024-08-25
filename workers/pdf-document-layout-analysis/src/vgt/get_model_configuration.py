import torch
from os.path import join
from detectron2.config import get_cfg
from detectron2.engine import default_setup, default_argument_parser
from configuration import service_logger, SRC_PATH, ROOT_PATH
from ditod import add_vit_config


def is_gpu_available():
    total_free_memory_in_system: float = 0.0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**2
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**2
            cached_memory = torch.cuda.memory_reserved(i) / 1024**2
            service_logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            service_logger.info(f"  Total Memory: {total_memory} MB")
            service_logger.info(f"  Allocated Memory: {allocated_memory} MB")
            service_logger.info(f"  Cached Memory: {cached_memory} MB")
            total_free_memory_in_system += total_memory - allocated_memory - cached_memory
        if total_free_memory_in_system < 3000:
            service_logger.info(f"Total free GPU memory is {total_free_memory_in_system} < 3000 MB. Switching to CPU.")
            service_logger.info("The process is probably going to be 15 times slower.")
    else:
        service_logger.info("No CUDA-compatible GPU detected. Switching to CPU.")
    return total_free_memory_in_system > 3000


def get_model_configuration():
    parser = default_argument_parser()
    args, unknown = parser.parse_known_args()
    args.config_file = join(SRC_PATH, "model_configuration", f"doclaynet_VGT_cascade_PTM.yaml")
    args.eval_only = True
    args.num_gpus = 1
    args.opts = [
        "MODEL.WEIGHTS",
        join(ROOT_PATH, "models", "doclaynet_VGT_model.pth"),
        "OUTPUT_DIR",
        join(ROOT_PATH, "model_output_doclaynet"),
    ]
    args.debug = False

    configuration = get_cfg()
    add_vit_config(configuration)
    configuration.merge_from_file(args.config_file)
    configuration.merge_from_list(args.opts)
    configuration.MODEL.DEVICE = "cuda" if is_gpu_available() else "cpu"
    configuration.freeze()
    default_setup(configuration, args)

    return configuration
