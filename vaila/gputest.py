import sys
import os
import platform
import torch
import ultralytics
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

def test_pytorch():
    table = Table(title="PyTorch GPU Check", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("PyTorch Version", torch.__version__)
    
    cuda_available = torch.cuda.is_available()
    table.add_row("CUDA Available", "✅ Yes" if cuda_available else "❌ No")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        table.add_row("Device Count", str(device_count))
        for i in range(device_count):
            table.add_row(f"Device {i}", torch.cuda.get_device_name(i))
            
        try:
            x = torch.rand(5, 3).cuda()
            table.add_row("Tensor Test", "✅ Tensor moved to GPU successfully")
            table.add_row("Current Device", str(x.device))
        except Exception as e:
            table.add_row("Tensor Test", f"❌ Failed: {e}")
    else:
        table.add_row("CUDA", "Not detected.")

    console.print(Panel(table, title="[bold blue]PyTorch[/bold blue]", border_style="blue"))

def test_yolo():
    table = Table(title="Ultralytics YOLO GPU Check", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Ultralytics Version", ultralytics.__version__)
    
    # Setup Models Directory
    # Script is in vaila/vaila/gputest.py (based on user info)
    # Models dir is vaila/vaila/models
    # So if we are running this script:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # If the directory doesn't exist, create it
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir, exist_ok=True)
            table.add_row("Models Directory", f"✅ Created: {models_dir}")
        except Exception as e:
            table.add_row("Models Directory", f"❌ Failed to create: {e}")
            return
    else:
        table.add_row("Models Directory", f"✅ Exists: {models_dir}")

    model_name = "yolo11n.pt" 
    model_path = os.path.join(models_dir, model_name)
    
    try:
        rprint(f"[yellow]Initializing {model_name} model check...[/yellow]")
        rprint(f"[yellow]Checking for model at: {model_path}[/yellow]")
        
        # We try to use the model path directly. 
        # If it doesn't exist, YOLO usually downloads to CWD. 
        # We want to force it to our models dir if possible or move it.
        # Ultralytics automatic download logic is complex to override cleanly without changing settings.
        # Simplest approach: Try loading from path. If fails, load by name (downloads to CWD), then move.
        
        if os.path.exists(model_path):
             model = YOLO(model_path)
             table.add_row("Model Load", f"✅ Loaded from: {model_path}")
        else:
             rprint(f"[yellow]Downloading {model_name}...[/yellow]")
             # This will download to current dir usually
             model = YOLO(model_name) 
             
             # Locate where it was downloaded (usually current dir)
             # And move it to models_dir
             downloaded_path = model.ckpt_path
             if downloaded_path and os.path.exists(downloaded_path):
                  if os.path.abspath(downloaded_path) != os.path.abspath(model_path):
                       import shutil
                       shutil.move(downloaded_path, model_path)
                       table.add_row("Model Download", f"✅ Downloaded & Moved to: {model_path}")
                       # Reload from new path to be sure
                       model = YOLO(model_path)
                  else:
                       table.add_row("Model Download", f"✅ Downloaded to: {model_path}")
             else:
                  table.add_row("Model Download", "⚠️ Downloaded but path unclear")

        if torch.cuda.is_available():
            model.to("cuda")
            table.add_row("Model Device", f"{model.device}")
            
            # Dummy inference
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(img, verbose=False)
            table.add_row("Inference on GPU", "✅ Success")
        else:
            table.add_row("Model Device", "CPU (CUDA not available)")
            
    except Exception as e:
        table.add_row("Error", f"❌ {str(e)}")

    console.print(Panel(table, title="[bold red]Ultralytics YOLO (v11)[/bold red]", border_style="red"))

def test_mediapipe():
    table = Table(title="MediaPipe Check", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("MediaPipe Version", mp.__version__)
    
    # Check for functions/modules
    has_tasks = hasattr(mp, 'tasks')
    table.add_row("mp.tasks", "✅ Available" if has_tasks else "❌ Missing")

    if not has_tasks:
        table.add_row("Status", "❌ Tasks API missing. Update mediapipe.")
        console.print(Panel(table, title="[bold green]MediaPipe[/bold green]", border_style="green"))
        return

    # Setup Models Directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        
    model_name = "pose_landmarker_lite.task"
    model_path = os.path.join(models_dir, model_name)
    
    # Download model if missing
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        try:
            rprint(f"[yellow]Downloading {model_name}...[/yellow]")
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            table.add_row("Model Download", f"✅ Downloaded to: {models_dir}")
        except Exception as e:
            table.add_row("Model Download", f"❌ Failed: {e}")
            console.print(Panel(table, title="[bold green]MediaPipe[/bold green]", border_style="green"))
            return
    else:
        table.add_row("Model File", f"✅ Found: {model_name}")

    # Inference Test using Tasks API
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            # Create mp image
            img_np = np.zeros((480, 640, 3), dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            
            # Detect
            detection_result = landmarker.detect(mp_image)
            table.add_row("Pose Inference", "✅ Success (Tasks API)")
            
            # Check for GPU delegate availability (indirectly)
            # MediaPipe Python GPU support is still experimental/limited in standard wheels
            if hasattr(mp, 'gpu'):
                 table.add_row("GPU Module", "✅ mp.gpu present")
            else:
                 table.add_row("GPU Module", "ℹ️  Standard CPU build")
                 
    except Exception as e:
        table.add_row("Inference Test", f"❌ Failed: {e}")

    console.print(Panel(table, title="[bold green]MediaPipe[/bold green]", border_style="green"))

if __name__ == "__main__":
    console.print(Panel.fit(f"[bold white]System: {platform.system()} | Python: {sys.version.split()[0]}[/bold white]", title="[bold yellow]vaila Environment Check[/bold yellow]"))
    
    test_pytorch()
    test_yolo()
    test_mediapipe()
    
    rprint("[bold white]Test Complete.[/bold white]")
