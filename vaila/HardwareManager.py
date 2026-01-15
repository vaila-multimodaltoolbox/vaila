"""
# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
#
# HardwareManager.py
# Manages hardware detection (GPU/CPU/RAM) and auto-optimizes models for the current system.
"""

import logging
import os
import shutil
import subprocess
import platform
from pathlib import Path

import psutil
from ultralytics import YOLO

try:
    import pynvml
except ImportError:
    pynvml = None


class HardwareManager:
    """
    Hardware Manager for vailá Multimodal Toolbox.
    Detects GPU, CPU, and RAM to dynamically optimize models and execution parameters.
    """
    def __init__(self, models_dir="models"):
        self.logger = logging.getLogger("vaila.hardware")
        # Ensure models_dir is absolute or relative to this file if not absolute
        if not os.path.isabs(models_dir):
            self.models_dir = Path(os.path.dirname(__file__)) / models_dir
        else:
            self.models_dir = Path(models_dir)
            
        self.gpu_info = self._detect_gpu()
        self.sys_info = self._detect_system()
        self.profile = self._get_hardware_profile()
        self.config = self.get_trt_config()

    def _detect_gpu(self):
        """Detects NVIDIA GPU VRAM and Name using pynvml."""
        if not pynvml:
            return {"name": "CPU", "total_vram_gb": 0, "cuda_capable": False}

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            # Handle bytes vs string return depending on pynvml version
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            name = name.replace(" ", "_")
            
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "name": name,
                "total_vram_gb": round(info.total / 1024**3, 2),
                "cuda_capable": True
            }
        except Exception as e:
            self.logger.warning(f"NVIDIA GPU not detected or driver error: {e}")
            return {"name": "CPU", "total_vram_gb": 0, "cuda_capable": False}

    def _detect_system(self):
        """Detects CPU and RAM info."""
        return {
            "cpu": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "os": f"{platform.system()} {platform.release()}"
        }

    def _get_hardware_profile(self):
        """Categorizes hardware into performance profiles."""
        vram = self.gpu_info["total_vram_gb"]
        if vram > 20:
            return "ULTRA"  # RTX 4090 / 5050 (High VRAM)
        elif 7 <= vram <= 20:
            return "HIGH"   # Alienware / Mid-range desktops (8GB-16GB)
        else:
            return "LITE"   # Laptops or CPU execution

    def get_trt_config(self):
        """Returns TensorRT configuration based on hardware profile."""
        configs = {
            "ULTRA": {"workspace": 8192, "precision": "fp16", "desc": "High Performance (8GB Workspace, FP16)"},
            "HIGH": {"workspace": 2048, "precision": "fp16", "desc": "Balanced (2GB Workspace, FP16)"},
            "LITE": {"workspace": 512, "precision": "fp32", "desc": "Compatibility (512MB Workspace, FP32)"} # FP16 might be supported but safer default
        }
        return configs.get(self.profile)

    def auto_export(self, model_input):
        """
        Checks for and automatically exports models to .engine format optimized for the current GPU.
        
        Args:
            model_input (str): Name of the model (e.g., 'yolo11n-pose.pt' or just 'yolo11n-pose')
            
        Returns:
            str: Path to the optimal model to load (.engine if available/created, else .pt).
        """
        model_name = Path(model_input).stem  # Remove extension if present
        
        # Base paths
        pt_path = self.models_dir / f"{model_name}.pt"
        
        # If no CUDA capability, just return the PT file (CPU execution)
        if not self.gpu_info["cuda_capable"]:
            return str(pt_path)

        # Engine path specific to this GPU to avoid conflicts
        # e.g. yolo11n-pose_NVIDIA_GeForce_RTX_4090.engine
        engine_name = f"{model_name}_{self.gpu_info['name']}.engine"
        engine_path = self.models_dir / engine_name

        if engine_path.exists():
            print(f"🚀 OPTIMIZED MODEL: Loading tailored TensorRT engine: {engine_path.name}")
            return str(engine_path)

        # Check if we should auto-export:
        # We need the source .pt file first. If it doesn't exist, we can't convert.
        # Ensure directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if not pt_path.exists():
            # If the PT file is not found, we might need to download it first via YOLO()
            # The calling script usually handles download, but we can double check.
            # Returning original input lets the caller handle download of .pt
            return str(pt_path)

        # Check for trtexec
        trtexec_cmd = shutil.which("trtexec")
        if not trtexec_cmd:
            print("Warning: trtexec not found. Skipping auto-optimization.")
            return str(pt_path)

        print(f"\n⚡ Auto-Exporting {model_name} for {self.gpu_info['name']} ({self.profile} Profile)...")
        print(f"   Config: {self.config['desc']}")
        
        # 1. Export to ONNX
        onnx_path = self.models_dir / f"{model_name}.onnx"
        if not onnx_path.exists():
            print("📦 Step 1/2: Exporting to ONNX...")
            try:
                model = YOLO(str(pt_path))
                model.export(format="onnx", dynamic=True, simplify=True)
                # Ultralytics exports to same dir as pt usually
            except Exception as e:
                print(f"❌ ONNX export failed: {e}")
                return str(pt_path)

        if not onnx_path.exists():
             # Sometimes export names slightly differently or fails silently
             print("❌ ONNX file not found after export attempt.")
             return str(pt_path)

        # 2. Convert to Engine
        print("⚙️  Step 2/2: Building TensorRT Engine (this takes a few minutes)...")
        precision_flag = f"--{self.config['precision']}"
        
        cmd = [
            trtexec_cmd,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={self.config['workspace']}",
            precision_flag,
            "--avgRuns=10",
            "--verbose" if self.profile == "ULTRA" else "--noDataTransfer" # Less verbose usually
        ]
        
        try:
            # Run conversion
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) 
            # Note: capturing output allows checking errors but hides progress. 
            # Given user wants to see it, maybe let it print? 
            # User said "Auto-Export", implied background or explicit. 
            # Let's print a success message.
            
            print(f"✨ SUCCESS! Optimized engine created: {engine_name}")
            return str(engine_path)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ TensorRT optimization failed.")
            print(f"   Error: {e.stderr.decode('utf-8')[-200:] if e.stderr else 'Unknown'}")
            return str(pt_path)
    
    def print_report(self):
        print("-" * 40)
        print("vailá Hardware Report")
        print("-" * 40)
        print(f"OS  : {self.sys_info['os']}")
        print(f"CPU : {self.sys_info['cpu']} ({self.sys_info['cores']} cores)")
        print(f"RAM : {self.sys_info['ram_total_gb']} GB")
        print(f"GPU : {self.gpu_info['name']} ({self.gpu_info['total_vram_gb']} GB VRAM)")
        print(f"Mode: {self.profile} Profile")
        print("-" * 40)
