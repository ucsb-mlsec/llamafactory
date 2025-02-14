import logging
import subprocess
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("gpu_monitor.log"), logging.StreamHandler()],
)


def get_gpu_info():
    """
    使用nvidia-smi获取GPU信息
    返回一个包含GPU信息的列表，每个元素是一个字典
    """
    try:
        # 运行nvidia-smi命令获取GPU信息
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd).decode("utf-8").strip()

        gpus = []
        for line in output.split("\n"):
            index, total, free, used = map(float, line.split(","))
            free_percentage = (free / total) * 100
            gpus.append(
                {
                    "index": int(index),
                    "total_memory": total,
                    "free_memory": free,
                    "used_memory": used,
                    "free_percentage": free_percentage,
                }
            )
        return gpus
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running nvidia-smi: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


def check_gpu_condition(gpus):
    """
    check_gpu_condition: 检查GPU是否满足条件
    返回：(bool, list) - (是否满足条件, 可用GPU的索引列表)
    """
    if not gpus or len(gpus) < 4:
        return False, []

    # get the indices of GPUs with >90% free memory
    available_gpus = [gpu["index"] for gpu in gpus if gpu["free_percentage"] > 92]

    if len(available_gpus) >= 4:
        # 只返回前4个可用的GPU
        return True, available_gpus[:4]
    return False, []


def run_command(command, gpu_indices):
    """
    运行指定的命令，添加CUDA_VISIBLE_DEVICES环境变量
    """
    try:
        # 将GPU索引列表转换为字符串
        gpu_list = ",".join(map(str, gpu_indices))

        # 构建完整的命令，包含CUDA_VISIBLE_DEVICES
        full_command = f"CUDA_VISIBLE_DEVICES={gpu_list} {command}"

        logging.info(f"Running command with GPUs {gpu_list}: {full_command}")

        subprocess.run(full_command, shell=True, check=True)
        logging.info(f"Successfully executed command: {full_command}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")
        return False


def main(command_to_run, check_interval=60):
    """
    主函数：监控GPU状态并在满足条件时执行命令

    参数:
    command_to_run: the command to run when conditions are met
    check_interval: check interval in seconds (default: 60)
    """
    logging.info(f"Starting GPU monitor. Will execute: {command_to_run}")
    logging.info(f"Checking every {check_interval} seconds")

    while True:
        try:
            gpus = get_gpu_info()

            if gpus:
                # record GPU information
                for gpu in gpus:
                    logging.debug(
                        f"GPU {gpu['index']}: {gpu['free_percentage']:.2f}% free"
                    )

                condition_met, available_gpus = check_gpu_condition(gpus)
                if condition_met:
                    logging.info(
                        f"Found 4 GPUs with >90% free memory: {available_gpus}"
                    )
                    if run_command(command_to_run, available_gpus):
                        logging.info("Command executed successfully, exiting monitor")
                        break
                    else:
                        logging.error("Command failed, continuing monitoring")

            time.sleep(check_interval)

        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, stopping monitor")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(check_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor GPU usage and execute command when conditions are met"
    )
    parser.add_argument(
        "command", help="Command to execute when GPU conditions are met"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )

    args = parser.parse_args()

    main(args.command, args.interval)
