from __future__ import annotations

import shutil
import subprocess
from typing import Dict

import psutil


class SystemMonitorService:
    def snapshot(self) -> Dict[str, float | str]:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        result: Dict[str, float | str] = {
            "cpu_percent": cpu,
            "ram_percent": ram.percent,
            "ram_used_gb": round(ram.used / (1024 ** 3), 2),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "gpu_name": "N/A",
            "gpu_percent": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_total_mb": 0.0,
            "gpu_temp_c": 0.0,
        }
        if shutil.which("nvidia-smi"):
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
                line = subprocess.check_output(cmd, text=True, timeout=2).splitlines()[0]
                name, util, mem_used, mem_total, temp = [x.strip() for x in line.split(",")]
                result.update(
                    {
                        "gpu_name": name,
                        "gpu_percent": float(util),
                        "gpu_mem_used_mb": float(mem_used),
                        "gpu_mem_total_mb": float(mem_total),
                        "gpu_temp_c": float(temp),
                    }
                )
            except Exception:
                pass
        return result
