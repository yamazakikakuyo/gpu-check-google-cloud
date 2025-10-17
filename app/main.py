# app/main.py
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title="gpu-check", version="1.0")

class PredictRequest(BaseModel):
    instances: Optional[List[Any]] = None
    parameters: Optional[Dict[str, Any]] = None

def _torch_probe() -> Dict[str, Any]:
    info = {"available": False, "device_count": 0, "devices": [], "smoke_ok": False, "error": None}
    try:
        import torch

        info["available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        for i in range(info["device_count"]):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            info["devices"].append({"index": i, "name": name, "capability": cap})
        # tiny smoke test: allocate and add two tensors on GPU 0 (if available)
        if info["available"] and info["device_count"] > 0:
            try:
                x = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
                y = torch.tensor([4.0, 5.0, 6.0], device="cuda:0")
                z = (x + y).tolist()
                info["smoke_ok"] = (z == [5.0, 7.0, 9.0])
            except Exception as e:
                info["error"] = f"torch smoke error: {e}"
    except Exception as e:
        info["error"] = f"torch import/error: {e}"
    return info

def _tf_probe() -> Dict[str, Any]:
    info = {"available": False, "devices": [], "error": None}
    try:
        import tensorflow as tf  # optional; may not be installed in the image
        gpus = tf.config.list_physical_devices("GPU")
        info["available"] = len(gpus) > 0
        info["devices"] = [str(g) for g in gpus]
    except Exception as e:
        info["error"] = f"tensorflow import/error: {e}"
    return info

def _nvidia_smi() -> Dict[str, Any]:
    info = {"present": False, "raw": None, "error": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        if out.returncode == 0:
            info["present"] = True
            info["raw"] = out.stdout.strip()
        else:
            info["error"] = out.stderr.strip() or f"nvidia-smi exit {out.returncode}"
    except FileNotFoundError:
        info["error"] = "nvidia-smi not found in container"
    except Exception as e:
        info["error"] = f"nvidia-smi error: {e}"
    return info

@app.get("/health")
def health():
    # Simple readiness/health signal for Vertex AI
    return {"status": "ok"}

@app.get("/gpu")
def gpu():
    torch_info = _torch_probe()
    tf_info = _tf_probe()
    smi_info = _nvidia_smi()
    gpu_ok = torch_info.get("available", False) and torch_info.get("smoke_ok", False)
    return {
        "gpu_ok": gpu_ok,
        "torch": torch_info,
        "tensorflow": tf_info,
        "nvidia_smi": smi_info,
        "notes": "gpu_ok=True means CUDA is visible and a tiny GPU tensor op succeeded in PyTorch."
    }

# Vertex AI will forward prediction requests to the route we configure (/predict below).
# We return the canonical {"predictions": ...} shape so it plays nicely with Vertex.
@app.post("/predict")
def predict(req: PredictRequest):
    # Ignore instances; just return one report per instance (or once if none provided)
    report = gpu()
    n = len(req.instances) if req.instances else 1
    return {"predictions": [report] * n}
