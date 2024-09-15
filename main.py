NGROK_TOKEN = ""
REPO = "SakuraLLM/Sakura-14B-Qwen2beta-v0.9.2-GGUF"
MODEL = "sakura-14b-qwen2beta-v0.9.2-iq4xs.gguf"
# REPO = "SakuraLLM/Sakura-32B-Qwen2beta-v0.9.1-GGUF"
# MODEL = "sakura-32b-qwen2beta-v0.9.1-iq4xs.gguf"
DOUBLE = False

model_dir = "/kaggle/working/llama.cpp/models/"


def main():
    ports = ["8080", "8081"] if DOUBLE else ["8080"]

    setup_ngrok(ports)
    download_model(
        repo_id=REPO,
        filename=MODEL,
        local_dir=model_dir,
    )

    from multiprocessing import Pool

    pool = Pool(processes=len(ports))
    pool.map(run_server, enumerate(ports))


def setup_ngrok(ports):
    from pyngrok import conf, ngrok

    print("设置Ngrok")
    conf.get_default().auth_token = NGROK_TOKEN
    conf.get_default().monitor_thread = False

    port_to_url = {}

    ssh_tunnels = ngrok.get_tunnels(conf.get_default())
    for ssh_tunnel in ssh_tunnels:
        port = ssh_tunnel.config["addr"].removeprefix("http://localhost:")
        if port in ports:
            port_to_url[port] = ssh_tunnel.public_url

    for port in ports:
        if port in port_to_url:
            url = port_to_url[port]
        else:
            ssh_tunnel = ngrok.connect(addr=port)
            url = ssh_tunnel.public_url
        print(f"隧道{port}： {url}")
    print()


def download_model(repo_id, filename, local_dir):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        RepositoryNotFoundError,
        EntryNotFoundError,
        LocalEntryNotFoundError,
    )

    print(f"开始下载模型：{repo_id}/{filename}")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
    except RepositoryNotFoundError:
        print("模型下载错误：无法找到要下载的仓库，请检查 REPO 参数。")
        exit(0)
    except LocalEntryNotFoundError:
        print("模型下载错误：网络已禁用或者无法连接。")
        exit(0)
    except EntryNotFoundError:
        print("模型下载错误：无法找到要下载的模型，请检查 MODEL 参数。")
        exit(0)
    else:
        print("模型下载成功")
    print()


def run_server(param):
    import os
    import subprocess

    pos, port = param
    p = subprocess.Popen(
        [
            "/kaggle/working/llama.cpp/server",
            "-m",
            f"{model_dir}/{MODEL}",
            "-ngl",
            "99",
            "-c",
            "4096",
            "-a",
            MODEL.removesuffix(".guff"),
            "--port",
            port,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(pos)} if DOUBLE else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf8",
        bufsize=0,
    )
    while True:
        for line in p.stdout:
            if line.startswith("{"):
                message = format_message(line)
                print(f"{pos}-{port} {message}")
            else:
                print(f"{pos}-{port} {line}", end="")


def format_message(line):
    import json
    from datetime import datetime

    job = json.loads(line)
    timestamp = datetime.fromtimestamp(job["timestamp"])
    msg = None

    if job["function"] == "print_timings":
        msg = (
            job["msg"]
            .replace("tokens per second", "token/s")
            .replace("ms per token", "ms/token")
        )
    else:
        for key_to_delete in ["tid", "timestamp", "level", "function", "line"]:
            del job[key_to_delete]

        if "msg" in job:
            msg = job["msg"]
            del job["msg"]

            if job:
                msg = f"{msg}: {json.dumps(job)}"

    if msg is None:
        msg = line.strip()
    return f"{timestamp.strftime('%H:%M:%S')} {msg}"


main()
