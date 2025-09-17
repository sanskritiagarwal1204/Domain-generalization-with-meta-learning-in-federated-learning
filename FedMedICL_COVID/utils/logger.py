import logging, sys, os, datetime, yaml, torch

def init_logger(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"run_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(path, mode="w")]
    )
    log = logging.getLogger(__name__)
    sep = "=" * 60
    log.info(sep)
    log.info("Experiment arguments:")
    log.info(yaml.dump(cfg, default_flow_style=False).rstrip())
    log.info(sep)
    log.info(f"PyTorch {torch.__version__} CUDA {torch.version.cuda} Device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    return log
