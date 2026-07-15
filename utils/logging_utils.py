import json
import logging
import os
import shutil
import sys
import time
import csv

def copy_exp_params(log_dir, config_file, args=None):
    log_dir = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p"
    )
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    if config_file is not None:  ##copy config file to log_dir directory
        shutil.copy2(src=config_file, dst=os.path.join(log_dir, os.path.basename(config_file)))
    if args is not None:
        json_fname = os.path.join(log_dir, "args.json")
        with open(json_fname, "w") as f:
            json.dump(args.__dict__, f, indent=3)
            

# Log the content of the file
def log_entire_script(filename):
    try:
        with open(filename, 'r') as file:
            logging.info("Printing the entire content of the script:")
            for line in file:
                logging.info(line.strip())
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
        

def save_sampled_indices_across_ranks(sampler, seed, rank, output_dir="data_sampler_indices"):
    # Each rank contributes its local indices
    local_indices = list(sampler)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rank{rank}_seed{seed}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "index"])  # header
        for i, idx in enumerate(local_indices):
            writer.writerow([f"{i:06d}", idx])