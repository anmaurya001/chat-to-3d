import psutil
import GPUtil
import time
from datetime import datetime
import os
import logging
import json
from pathlib import Path
import argparse

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "system_resources.log"
json_log_file = log_dir / "system_resources.json"

# Initialize JSON log file with an empty array if it doesn't exist
if not json_log_file.exists():
    with open(json_log_file, 'w') as f:
        json.dump([], f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_cpu_info():
    """Get CPU usage information."""
    try:
        # Get overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get per-core CPU usage
        cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        # Get CPU frequency
        cpu_freq = psutil.cpu_freq()
        
        return {
            'overall_percent': cpu_percent,
            'per_core_percent': cpu_percent_per_core,
            'frequency': {
                'current': cpu_freq.current if cpu_freq else None,
                'min': cpu_freq.min if cpu_freq else None,
                'max': cpu_freq.max if cpu_freq else None
            },
            'core_count': psutil.cpu_count(logical=True),
            'physical_core_count': psutil.cpu_count(logical=False)
        }
    except Exception as e:
        logging.error(f"Error getting CPU info: {e}")
        return None

def get_gpu_info():
    """Get GPU VRAM usage information."""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
            })
        return gpu_info
    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")
        return None

def get_ram_info():
    """Get system RAM usage information."""
    try:
        ram = psutil.virtual_memory()
        return {
            'total': ram.total / (1024**3),  # Convert to GB
            'used': ram.used / (1024**3),
            'percent': ram.percent
        }
    except Exception as e:
        logging.error(f"Error getting RAM info: {e}")
        return None

def get_disk_info():
    """Get C: drive usage information."""
    try:
        disk = psutil.disk_usage('C:\\')
        return {
            'total': disk.total / (1024**3),  # Convert to GB
            'used': disk.used / (1024**3),
            'percent': disk.percent
        }
    except Exception as e:
        logging.error(f"Error getting disk info: {e}")
        return None

def log_resources():
    """Log all system resources."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Collect all resource data
    resource_data = {
        'timestamp': timestamp,
        'gpu': get_gpu_info(),
        'ram': get_ram_info(),
        'disk': get_disk_info()
    }
    
    # Log to JSON file
    try:
        with open(json_log_file, 'r+') as f:
            data = json.load(f)
            data.append(resource_data)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    except Exception as e:
        logging.error(f"Error writing to JSON file: {e}")
        
    if resource_data['gpu']:
        for gpu in resource_data['gpu']:
            logging.info(f"GPU {gpu['id']} ({gpu['name']}): "
                        f"VRAM {gpu['memory_used']}MB / {gpu['memory_total']}MB "
                        f"({gpu['memory_percent']:.1f}%)")
    
    if resource_data['ram']:
        logging.info(f"RAM: {resource_data['ram']['used']:.1f}GB / {resource_data['ram']['total']:.1f}GB "
                    f"({resource_data['ram']['percent']}%)")
    
    if resource_data['disk']:
        logging.info(f"C: Drive: {resource_data['disk']['used']:.1f}GB / {resource_data['disk']['total']:.1f}GB "
                    f"({resource_data['disk']['percent']}%)")
    
    logging.info("-" * 50)

def main():
    """Main function to monitor resources periodically."""
    parser = argparse.ArgumentParser(description="Monitor and log system resources.")
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=600,
        help="Logging interval in seconds (default: 600)"
    )
    args = parser.parse_args()

    logging.info("Starting system resource monitoring...")
    logging.info(f"Structured data will be saved to: {json_log_file}")
    logging.info(f"Logging interval: {args.interval} seconds")
    
    try:
        while True:
            log_resources()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 