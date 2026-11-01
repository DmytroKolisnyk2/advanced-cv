"""Minimal RunPod controller (GraphQL API) for creating, inspecting and stopping one pod.

The API key is read from RUNPOD_API_KEY or from the repository-level .env file and is
only ever sent in the Authorization header.

    python remote/runpod_ctl.py balance
    python remote/runpod_ctl.py create --gpu "NVIDIA GeForce RTX 4090"
    python remote/runpod_ctl.py status
    python remote/runpod_ctl.py ssh        # prints the ssh target of the running pod
    python remote/runpod_ctl.py stop | terminate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://api.runpod.io/graphql"
REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = Path(__file__).resolve().parent / ".pod.json"
SSH_KEY = Path.home() / ".ssh" / "runpod_a_nlp"
DEFAULT_IMAGE = "runpod/base:1.4.1-ubuntu2404"


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env_file = REPO_ROOT / ".env"
        for line in env_file.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    return key


def gql(query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        API_URL,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        sys.exit(f"RunPod API error: {json.dumps(data['errors'], indent=1)}")
    return data["data"]


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=1))


def cmd_balance(_: argparse.Namespace) -> None:
    me = gql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr } } }")["myself"]
    print(json.dumps(me, indent=1))


def cmd_create(args: argparse.Namespace) -> None:
    if load_state().get("pod_id"):
        sys.exit(f"A pod is already tracked in {STATE_FILE}; terminate it first")
    pub_key = SSH_KEY.with_suffix(".pub").read_text().strip()
    query = """
    mutation($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } }
    }"""
    pod_input = {
        "cloudType": args.cloud,
        "gpuCount": 1,
        "gpuTypeId": args.gpu,
        "name": args.name,
        "imageName": args.image,
        "containerDiskInGb": args.disk,
        "volumeInGb": args.volume,
        "volumeMountPath": "/workspace",
        "minVcpuCount": int(os.environ.get("POD_VCPU", 4)),
        "minMemoryInGb": int(os.environ.get("POD_MEM", 24)),
        "ports": "22/tcp",
        "supportPublicIp": True,
        "startSsh": True,
        "env": [{"key": "PUBLIC_KEY", "value": pub_key}],
    }
    if args.cuda:
        pod_input["allowedCudaVersions"] = args.cuda
    pod = gql(query, {"input": pod_input})["podFindAndDeployOnDemand"]
    save_state({"pod_id": pod["id"], "created": time.time(), "cost_per_hr": pod["costPerHr"]})
    print(json.dumps(pod, indent=1))


def pod_info(pod_id: str) -> dict:
    query = """
    query($input: PodFilter) {
      pod(input: $input) {
        id name desiredStatus costPerHr imageName
        machine { gpuDisplayName }
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
      }
    }"""
    return gql(query, {"input": {"podId": pod_id}})["pod"]


def ssh_target(info: dict) -> tuple[str, int] | None:
    runtime = info.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port["privatePort"] == 22 and port["isIpPublic"]:
            return port["ip"], port["publicPort"]
    return None


def cmd_status(_: argparse.Namespace) -> None:
    state = load_state()
    if not state.get("pod_id"):
        sys.exit("no tracked pod")
    info = pod_info(state["pod_id"])
    hours = (time.time() - state["created"]) / 3600
    info["tracked_hours"] = round(hours, 2)
    info["approx_cost_usd"] = round(hours * state["cost_per_hr"], 2)
    print(json.dumps(info, indent=1))


def cmd_ssh(args: argparse.Namespace) -> None:
    state = load_state()
    deadline = time.time() + args.wait
    while True:
        target = ssh_target(pod_info(state["pod_id"]))
        if target or time.time() > deadline:
            break
        time.sleep(10)
    if not target:
        sys.exit("ssh port not ready")
    ip, port = target
    state["ssh"] = {"ip": ip, "port": port}
    save_state(state)
    print(f"root@{ip} -p {port} -i {SSH_KEY}")


def cmd_stop(_: argparse.Namespace) -> None:
    state = load_state()
    print(gql("mutation($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }",
              {"input": {"podId": state["pod_id"]}}))


def cmd_terminate(_: argparse.Namespace) -> None:
    state = load_state()
    gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": state["pod_id"]}})
    STATE_FILE.unlink()
    print(f"terminated {state['pod_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance").set_defaults(func=cmd_balance)
    create = sub.add_parser("create")
    create.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    create.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    create.add_argument("--name", default="ai401-a1")
    create.add_argument("--image", default=DEFAULT_IMAGE)
    create.add_argument("--disk", type=int, default=60)
    create.add_argument("--volume", type=int, default=80)
    create.add_argument("--cuda", nargs="*", default=["12.6", "12.7", "12.8", "12.9", "13.0"])
    create.set_defaults(func=cmd_create)
    sub.add_parser("status").set_defaults(func=cmd_status)
    ssh = sub.add_parser("ssh")
    ssh.add_argument("--wait", type=int, default=300)
    ssh.set_defaults(func=cmd_ssh)
    sub.add_parser("stop").set_defaults(func=cmd_stop)
    sub.add_parser("terminate").set_defaults(func=cmd_terminate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
