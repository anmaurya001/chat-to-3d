import bpy
import os
import subprocess
import json
import threading
import time
import socket
import signal
import errno
import platform
import ctypes
import logging
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty

# Set up logging to file and console
log_file = os.path.join(os.path.expanduser("~"), "trellis_addon.log")
logger = logging.getLogger(__name__)

# Global variables for status
llm_status = "NOT READY"
trellis_status = "NOT READY"
gradio_status = "NOT READY"
llm_status_lock = threading.Lock()
trellis_status_lock = threading.Lock()
gradio_status_lock = threading.Lock()
stop_thread = False

# Global stop event for logging threads
log_output_stop = threading.Event()

# Console handler for dynamic level adjustment
console_handler = logging.StreamHandler()

def update_logging_level():
    """Update the console handler's logging level based on user preference."""
    addon_prefs = bpy.context.preferences.addons[__name__].preferences
    log_level_str = addon_prefs.console_log_level
    log_level = {
        "ERROR": logging.ERROR,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }.get(log_level_str, logging.DEBUG)
    console_handler.setLevel(log_level)
    logger.debug(f"Console logging level updated to: {log_level_str}")

def setup_logging():
    """Set up the logger with file and console handlers."""
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    
    console_handler.setFormatter(log_format)
    
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    update_logging_level()

def get_conda_python_path():
    """Attempt to find the Conda 'trellis' environment's Python executable."""
    conda_base = os.environ.get("CONDA_PREFIX")
    if conda_base:
        logger.debug("Using CONDA_PREFIX: %s", conda_base)
        if os.path.basename(conda_base) == "trellis" and os.path.basename(os.path.dirname(conda_base)) == "envs":
            conda_base = os.path.dirname(os.path.dirname(conda_base))
            logger.debug("Adjusted Conda base from CONDA_PREFIX: %s", conda_base)
    else:
        try:
            result = subprocess.run(
                ["conda", "info", "--base"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout:
                conda_base = result.stdout.strip()
                logger.debug("Found Conda base using 'conda info --base': %s", conda_base)
            else:
                logger.warning("Could not locate Conda base using 'conda info --base': %s", result.stderr)
                conda_base = os.path.expanduser("~/Miniconda3")
                logger.debug("Falling back to default Conda base: %s", conda_base)
        except subprocess.SubprocessError as e:
            logger.error("Failed to locate Conda base using 'conda info --base': %s", str(e))
            conda_base = os.path.expanduser("~/Miniconda3")
            logger.debug("Falling back to default Conda base: %s", conda_base)

    if platform.system() == "Windows":
        python_path = os.path.join(conda_base, "envs", "trellis", "python.exe")
    else:
        python_path = os.path.join(conda_base, "envs", "trellis", "bin", "python")
    
    if os.path.isfile(python_path):
        try:
            result = subprocess.run(
                [python_path, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.debug("Conda Python version: %s", result.stdout.strip())
            return python_path
        except subprocess.SubprocessError as e:
            logger.error("Failed to verify Conda Python: %s", str(e))
            return None
    logger.warning("Conda Python not found at %s", python_path)
    return None

def try_curl_health_check():
    """Try checking the LLM service health on both IPv6 and IPv4 addresses."""
    addresses = ["[::1]", "127.0.0.1"]
    for addr in addresses:
        try:
            result = subprocess.run(
                ['curl', '-X', 'GET', f'http://{addr}:8000/v1/health/ready'],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.debug("LLM health check for %s: returncode=%d, stdout=%s, stderr=%s",
                        addr, result.returncode, result.stdout, result.stderr)
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout)
                    return response, True
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM health check response from %s: %s", addr, result.stdout)
                    continue
            else:
                logger.debug("LLM health check failed for %s: stderr=%s", addr, result.stderr)
        except subprocess.SubprocessError as e:
            logger.debug("LLM health check failed for %s: %s", addr, str(e))
    return None, False

def stop_llm_service():
    """Stop the LLM service running in a Podman container via WSL."""
    global llm_status
    container_name = "CHAT_TO_3D"
    
    def is_container_running():
        """Check if the CHAT_TO_3D container is running or exists."""
        try:
            result = subprocess.run(
                ["wsl", "podman", "ps", "-a", "--format", "{{.Names}} {{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    if line.startswith(container_name):
                        status = line[len(container_name):].strip()
                        logger.debug(f"Container {container_name} status: {status}")
                        return True
                return False
            else:
                logger.error(f"Failed to list containers: {result.stderr}")
                return False
        except subprocess.SubprocessError as e:
            logger.error(f"Error checking container status: {str(e)}")
            return False
    
    # Check if container exists
    if not is_container_running():
        logger.info(f"No {container_name} container found, assuming LLM service is not running")
        with llm_status_lock:
            llm_status = "NOT READY"
        return True
    
    # Try to stop the container (up to 2 attempts)
    stop_attempts = 2
    for attempt in range(1, stop_attempts + 1):
        logger.info(f"Attempt {attempt}/{stop_attempts} to stop {container_name} container")
        try:
            result = subprocess.run(
                ["wsl", "podman", "stop", container_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"{container_name} container stopped successfully on attempt {attempt}")
                break
            else:
                logger.warning(f"Failed to stop {container_name} container on attempt {attempt}: {result.stderr}")
        except subprocess.SubprocessError as e:
            logger.warning(f"Error stopping {container_name} container on attempt {attempt}: {str(e)}")
        if attempt < stop_attempts:
            logger.debug("Retrying after 2 seconds...")
            time.sleep(2)
    else:
        logger.error(f"Failed to stop {container_name} container after {stop_attempts} attempts")
    
    # Monitor container status for up to 15 seconds
    start_time = time.time()
    timeout = 15
    while time.time() - start_time < timeout:
        if not is_container_running():
            logger.info(f"{container_name} container is no longer running")
            with llm_status_lock:
                llm_status = "NOT READY"
            return True
        logger.debug(f"Waiting for {container_name} container to stop...")
        time.sleep(1)
    
    # If container is still running, force remove it
    logger.warning(f"{container_name} container did not stop within {timeout} seconds, attempting force removal")
    try:
        result = subprocess.run(
            ["wsl", "podman", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info(f"{container_name} container force-removed successfully")
        else:
            logger.error(f"Failed to force-remove {container_name} container: {result.stderr}")
            return False
    except subprocess.SubprocessError as e:
        logger.error(f"Error force-removing {container_name} container: {str(e)}")
        return False
    
    # Monitor again for up to 15 seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_container_running():
            logger.info(f"{container_name} container is no longer running after force removal")
            with llm_status_lock:
                llm_status = "NOT READY"
            return True
        logger.debug(f"Waiting for {container_name} container to be removed...")
        time.sleep(1)
    
    logger.error(f"Timeout: {container_name} container could not be stopped or removed within {timeout} seconds")
    return False

def check_llm_service():
    """Run curl command to check LLM service status in a separate thread."""
    global llm_status, stop_thread
    while not stop_thread:
        response, success = try_curl_health_check()
        with llm_status_lock:
            if success and response.get("message") == "Service is ready.":
                llm_status = "READY"
            else:
                llm_status = "NOT READY"
        time.sleep(10)

def check_trellis_service():
    """Check Trellis server status by attempting socket connection."""
    global trellis_status, stop_thread
    while not stop_thread:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(0.5)
                client.connect(('localhost', 12345))
                with trellis_status_lock:
                    trellis_status = "READY"
        except (ConnectionRefusedError, socket.timeout):
            with trellis_status_lock:
                trellis_status = "NOT READY"
        time.sleep(10)

def check_gradio_service():
    """Check Gradio UI status by attempting to fetch HTTP headers."""
    global gradio_status, stop_thread
    while not stop_thread:
        try:
            result = subprocess.run(
                ['curl', '-Is', 'http://127.0.0.1:7860/'],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.debug("Gradio health check: returncode=%d, stdout=%s, stderr=%s",
                        result.returncode, result.stdout.strip(), result.stderr.strip())
            if result.returncode == 0 and result.stdout:
                status_line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
                if status_line.startswith("HTTP/"):
                    if " 200 " in status_line or " 302 " in status_line:
                        with gradio_status_lock:
                            gradio_status = "READY"
                        logger.debug("Gradio server started successfully")
                    else:
                        with gradio_status_lock:
                            gradio_status = "NOT READY"
                        logger.debug("Gradio server not ready: status=%s", status_line)
                else:
                    with gradio_status_lock:
                        gradio_status = "NOT READY"
                    logger.debug("Gradio server not ready: no HTTP status")
            else:
                with gradio_status_lock:
                    gradio_status = "NOT READY"
                logger.debug("Gradio server not ready: curl failed")
        except subprocess.SubprocessError as e:
            logger.debug("Gradio health check failed: %s", str(e))
            with gradio_status_lock:
                gradio_status = "NOT READY"
        time.sleep(10)

def start_status_threads():
    """Start LLM, Trellis, and Gradio status checking threads."""
    global stop_thread
    stop_thread = False
    llm_thread = threading.Thread(target=check_llm_service, daemon=True)
    trellis_thread = threading.Thread(target=check_trellis_service, daemon=True)
    gradio_thread = threading.Thread(target=check_gradio_service, daemon=True)
    llm_thread.start()
    trellis_thread.start()
    gradio_thread.start()

def stop_status_threads():
    """Stop LLM, Trellis, and Gradio status checking threads."""
    global stop_thread
    stop_thread = True

class TrellisTerminator:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port

    def is_server_running(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(0.5)
                client.connect((self.host, self.port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False

    def terminate_process(self, pid):
        """Terminate process with SIGTERM, fallback to SIGKILL if needed."""
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process {pid}")
            for i in range(15):
                if not process_exists(pid):
                    logger.info(f"Process {pid} terminated successfully after {i + 1} seconds")
                    return True
                logger.debug(f"Waiting for process {pid} to terminate... (attempt {i + 1}/15)")
                time.sleep(1)
            if process_exists(pid):
                logger.warning(f"Process {pid} did not terminate gracefully after 15 seconds, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                if not process_exists(pid):
                    logger.info(f"Process {pid} terminated successfully after SIGKILL")
                    return True
                else:
                    logger.error(f"Process {pid} could not be terminated even with SIGKILL")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
            return False

    def terminate_and_wait(self):
        if not self.is_server_running():
            logger.info("TRELLIS server not running")
            return True
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect((self.host, self.port))
                client.send(b'terminate')
                response = client.recv(1024)
                if not response.startswith(b'terminating:'):
                    logger.error(f"Unexpected response: {response}")
                    return False
                pid = int(response.decode().split(':')[1])
                logger.info(f"Received PID {pid} from server")
                return self.terminate_process(pid)
        except Exception as e:
            logger.error(f"Error during termination: {e}")
            return False

def process_exists(pid):
    """Check if a process with the given PID exists and is still running."""
    try:
        import psutil
        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            status = process.status()
            children = process.children(recursive=True)
            if children:
                child_pids = [child.pid for child in children]
                logger.debug(f"Process {pid} has {len(children)} child processes: {child_pids}")
            logger.debug(f"Process {pid} status: {status}")
            if status == psutil.STATUS_ZOMBIE:
                logger.debug(f"Process {pid} is in zombie state, treating as terminated")
                return False
            return True
        return False
    except ImportError:
        logger.warning("psutil is not installed, using fallback method. Install it for reliable process management: C:\\Program Files\\WindowsApps\\BlenderFoundation.Blender4.2LTS_4.2.9.0_x64__ppwjx1n5r4v9t\\Blender\\4.2\\python\\bin\\pip install psutil")
        if platform.system() == "Windows":
            PROCESS_QUERY_INFORMATION = 0x0400
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            if handle:
                try:
                    exit_code = ctypes.c_ulong()
                    success = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    ctypes.windll.kernel32.CloseHandle(handle)
                    if success:
                        return exit_code.value == 259  # STILL_ACTIVE
                    else:
                        logger.debug(f"Failed to get exit code for PID {pid}: {ctypes.get_last_error()}")
                        return False
                except Exception as e:
                    logger.debug(f"Error checking exit code for PID {pid}: {e}")
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return False
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError as e:
                return e.errno != errno.ESRCH
    except psutil.NoSuchProcess:
        return False
    except psutil.Error as e:
        logger.debug(f"psutil error while checking PID {pid}: {e}")
        return False

class TrellisAddonPreferences(AddonPreferences):
    bl_idname = __name__

    base_path: StringProperty(
        name="Chat-To-3D Base Path",
        description="Base path for TrellChat-To-3D project (e.g., C:\\path\\to\\chat-to-3d)",
        default=os.environ.get("TRELLIS_BASE_PATH", ""),
        subtype='DIR_PATH'
    )

    console_log_level: EnumProperty(
        name="Console Log Level",
        description="Set the level of messages displayed in the console",
        items=[
            ("ERROR", "Error", "Display only error messages"),
            ("INFO", "Info", "Display info and error messages"),
            ("DEBUG", "Debug", "Display debug, info, and error messages"),
        ],
        default="INFO",
        update=lambda self, context: update_logging_level()
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Chat-to-3D Preferences")
        layout.prop(self, "base_path")
        layout.label(text="Logging Preferences")
        layout.prop(self, "console_log_level")

class TRELLIS_OT_ManageTrellis(Operator):
    bl_idname = "trellis.manage_trellis"
    bl_label = "Manage TRELLIS"
    
    def __init__(self):
        self.trellis_process = None
        self.stdout_thread = None
        self.stderr_thread = None

    def check_llm_ready(self, timeout=120):
        """Check if LLM service is ready, with timeout in seconds."""
        start_time = time.time()
        last_log_time = start_time
        while time.time() - start_time < timeout:
            response, success = try_curl_health_check()
            with llm_status_lock:
                global llm_status
                if success and response.get("message") == "Service is ready.":
                    llm_status = "READY"
                    logger.info("LLM service is ready")
                    return True
                else:
                    current_time = time.time()
                    elapsed_time = int(current_time - start_time)
                    if current_time - last_log_time >= 5:
                        logger.warning("LLM not ready after %d seconds: response=%s", elapsed_time, response if success else "no response")
                        last_log_time = current_time
            time.sleep(1)
        elapsed_time = int(time.time() - start_time)
        logger.error("LLM service failed to start within %d seconds", elapsed_time)
        return False
    
    def start_llm(self):
        """Start the LLM service and capture its output, switching CWD to nim_llm directory."""
        global llm_status
        llm_status = "STARTING..."
        original_cwd = os.getcwd()
        try:
            python_path = get_conda_python_path()
            if not python_path:
                logger.error("Conda Python executable not found")
                return
            
            base_path = bpy.context.preferences.addons[__name__].preferences.base_path
            llm_start_script = os.path.join(base_path, "nim_llm", "run_llama.py")
            if not os.path.isfile(llm_start_script):
                logger.error(f"LLM start script not found: {llm_start_script}")
                return
            
            working_dir = os.path.join(base_path, "nim_llm")
            logger.info("Switching working directory to: %s", working_dir)
            os.chdir(working_dir)
            
            os.environ['NIM_CACHE'] = os.path.join(base_path, "nim_cache")
            
            process = subprocess.Popen(
                [python_path, llm_start_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            logger.info("LLM service started with PID %d", process.pid)
            
            def log_output(pipe, log_func):
                while not log_output_stop.is_set():
                    try:
                        line = pipe.readline()
                        if not line:
                            break
                        if "INFO" in line:
                            logger.info(line.strip())
                        elif "DEBUG" in line:
                            logger.debug(line.strip())
                        elif "WARNING" in line:
                            logger.warning(line.strip())
                        else:
                            log_func(line.strip())
                    except ValueError as e:
                        logger.debug(f"Pipe closed while logging: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error while logging output: {e}")
                        break
            
            stdout_thread = threading.Thread(
                target=log_output,
                args=(process.stdout, lambda x: logger.info("LLM stdout: %s", x)),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=log_output,
                args=(process.stderr, lambda x: logger.error("LLM stderr: %s", x)),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
        except Exception as e:
            logger.error("Failed to start LLM: %s", str(e))
        finally:
            logger.info("Restoring working directory to: %s", original_cwd)
            os.chdir(original_cwd)
    
    def execute(self, context):
        global trellis_status, llm_status
        with trellis_status_lock, llm_status_lock:
            current_status = "READY" if trellis_status == "READY" and llm_status == "READY" else "NOT READY"
            print(f"Current Status is: {current_status}")
            llm_stat = llm_status
            trellis_stat = trellis_status
        
        base_path = bpy.context.preferences.addons[__name__].preferences.base_path
        if not base_path or not os.path.isdir(base_path):
            self.report({'ERROR'}, "Invalid or missing Trellis Base Path")
            return {'CANCELLED'}
        
        llm_start_script = os.path.join(base_path, "nim_llm", "run_llama.py")
        trellis_run_script = os.path.join(base_path, "chat-to-3d-core", "run.py")
        
        python_path = get_conda_python_path()
        if not python_path or not os.path.isfile(python_path):
            self.report({'ERROR'}, "Conda Python executable not found")
            return {'CANCELLED'}
        
        if current_status == "READY":
            if self.trellis_process:
                log_output_stop.set()
                if self.stdout_thread:
                    self.stdout_thread.join()
                if self.stderr_thread:
                    self.stderr_thread.join()
                if self.trellis_process:
                    try:
                        self.trellis_process.stdout.close()
                        self.trellis_process.stderr.close()
                    except Exception as e:
                        logger.debug(f"Error closing pipes: {e}")
                    self.trellis_process = None
                    self.stdout_thread = None
                    self.stderr_thread = None
                log_output_stop.clear()

            # Stop LLM service
            if not stop_llm_service():
                self.report({'WARNING'}, "Failed to stop LLM service, continuing with Trellis termination")
            else:
                self.report({'INFO'}, "LLM service stopped successfully")

            # Stop Trellis service
            terminator = TrellisTerminator()
            tw = terminator.terminate_and_wait()
            print(f'tw output is {str(tw)}')
            if tw:
                self.report({'INFO'}, "TRELLIS terminated successfully")
                with trellis_status_lock:
                    trellis_status = "NOT READY"
                response, success = try_curl_health_check()
                with llm_status_lock:
                    llm_status = "READY" if success and response.get("message") == "Service is ready." else "NOT READY"
            else:
                self.report({'ERROR'}, "Failed to terminate TRELLIS")
        else:
            if not os.path.isfile(llm_start_script):
                self.report({'ERROR'}, f"LLM start script not found: {llm_start_script}")
                return {'CANCELLED'}
            if not os.path.isfile(trellis_run_script):
                self.report({'ERROR'}, f"Trellis run script not found: {trellis_run_script}")
                return {'CANCELLED'}
            
            if llm_stat != "READY":
                llm_already_ready = False
                response, success = try_curl_health_check()
                if success and response.get("message") == "Service is ready.":
                    with llm_status_lock:
                        llm_status = "READY"
                    llm_already_ready = True
                    logger.info("LLM service is already ready, skipping startup")
                else:
                    logger.info("LLM service not ready: response=%s", response if success else "no response")
                
                if not llm_already_ready:
                    llm_thread = threading.Thread(target=self.start_llm, daemon=True)
                    llm_thread.start()
                    
                    self.report({'INFO'}, "Starting LLM service...")
                    if not self.check_llm_ready(timeout=120):
                        self.report({'ERROR'}, "LLM service failed to start within timeout")
                        return {'CANCELLED'}
                    self.report({'INFO'}, "LLM service is ready")
            else:
                logger.info("LLM service is already READY, skipping startup")
                self.report({'INFO'}, "LLM service is already READY, skipping startup")
            
            if trellis_stat != "READY":
                os.environ['ATTN_BACKEND'] = 'flash-attn'
                os.environ['SPCONV_ALGO'] = 'native'
                os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
                try:
                    trellis_process = subprocess.Popen(
                        [python_path, trellis_run_script],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=os.path.join(base_path, "chat-to-3d-core"),
                        creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                    )
                    logger.info("Trellis server started with PID %d", trellis_process.pid)

                    def log_output(pipe, log_func):
                        while not log_output_stop.is_set():
                            try:
                                line = pipe.readline()
                                if not line:
                                    break
                                if "INFO" in line:
                                    logger.info(line.strip())
                                elif "DEBUG" in line:
                                    logger.debug(line.strip())
                                elif "WARNING" in line:
                                    logger.warning(line.strip())
                                else:
                                    log_func(line.strip())
                            except ValueError as e:
                                logger.debug(f"Pipe closed while logging: {e}")
                                break
                            except Exception as e:
                                logger.error(f"Error while logging output: {e}")
                                break

                    stdout_thread = threading.Thread(
                        target=log_output,
                        args=(trellis_process.stdout, lambda x: logger.info("Trellis stdout: %s", x)),
                        daemon=True
                    )
                    stderr_thread = threading.Thread(
                        target=log_output,
                        args=(trellis_process.stderr, lambda x: logger.error("Trellis stderr: %s", x)),
                        daemon=True
                    )
                    stdout_thread.start()
                    stderr_thread.start()

                    self.trellis_process = trellis_process
                    self.stdout_thread = stdout_thread
                    self.stderr_thread = stderr_thread

                    start_time = time.time()
                    timeout = 30
                    retry_interval = 1
                    while time.time() - start_time < timeout:
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                                client.settimeout(1)
                                client.connect(('localhost', 12345))
                                with trellis_status_lock:
                                    trellis_status = "READY"
                                self.report({'INFO'}, "TRELLIS server started successfully")
                                break
                        except (ConnectionRefusedError, socket.timeout) as e:
                            logger.debug(f"Failed to connect to Trellis server: {e}, retrying...")
                            time.sleep(retry_interval)
                    else:
                        self.report({'ERROR'}, "TRELLIS server failed to start within timeout")
                        raise TimeoutError("Trellis server did not start within timeout")
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to start TRELLIS: {str(e)}")
                    logger.error("Trellis startup error: %s", str(e))
                    if self.trellis_process:
                        log_output_stop.set()
                        if self.stdout_thread:
                            self.stdout_thread.join()
                        if self.stderr_thread:
                            self.stderr_thread.join()
                        try:
                            self.trellis_process.stdout.close()
                            self.trellis_process.stderr.close()
                        except Exception as e:
                            logger.debug(f"Error closing pipes: {e}")
                        self.trellis_process = None
                        self.stdout_thread = None
                        self.stderr_thread = None
                        log_output_stop.clear()
                    raise
            else:
                logger.info("Trellis service is already READY, skipping startup")
                self.report({'INFO'}, "Trellis service is already READY, skipping startup")
        
        return {'FINISHED'}

class TRELLIS_PT_Panel(Panel):
    bl_label = "CHAT-TO-3D Server Manager"
    bl_idname = "PT_CHAT-TO-3D_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CHAT-TO-3D'
    
    def draw(self, context):
        layout = self.layout
        addon_prefs = context.preferences.addons[__name__].preferences
        
        layout.label(text="Base Path is set in Add-on Preferences")
        
        with trellis_status_lock, llm_status_lock:
            overall_status = "READY" if trellis_status == "READY" and llm_status == "READY" else "NOT READY"
        layout.operator(
            TRELLIS_OT_ManageTrellis.bl_idname,
            text="Terminate CHAT-TO-3D" if overall_status == "READY" else "Start CHAT-TO-3D"
        )
        
        with llm_status_lock:
            llm_stat = llm_status
        layout.label(
            text=f"LLM Service Status: {llm_stat}",
            icon='CHECKMARK' if llm_stat == "READY" else 'ERROR'
        )
        
        with trellis_status_lock:
            trellis_stat = trellis_status
        layout.label(
            text=f"Trellis Server Status: {trellis_stat}",
            icon='CHECKMARK' if trellis_stat == "READY" else 'ERROR'
        )
        
        with gradio_status_lock:
            gradio_stat = gradio_status
        layout.label(
            text=f"Gradio Web UI Status: {gradio_stat}",
            icon='CHECKMARK' if gradio_stat == "READY" else 'ERROR'
        )
        
        row = layout.row(align=True)     
        button_row = row.row(align=True)
        button_row.operator(
            "wm.url_open",
            text="Open CHAT-TO-3D UI",
            icon='URL'
        ).url = "http://127.0.0.1:7860/"
        button_row.enabled = (gradio_stat == "READY")

def update_status_ui():
    """Timer function to refresh the UI with the latest statuses."""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
    return 1.0

classes = (
    TrellisAddonPreferences,
    TRELLIS_OT_ManageTrellis,
    TRELLIS_PT_Panel,
)

def register():
    print("CHAT-TO-3D Server Manager Add-on Loaded - Version 1.0 - May 30, 2025")
    for cls in classes:
        bpy.utils.register_class(cls)
    setup_logging()
    start_status_threads()
    bpy.app.timers.register(update_status_ui, persistent=True)

def unregister():
    global log_output_stop
    log_output_stop.set()
    
    # Stop LLM service
    if not stop_llm_service():
        logger.warning("Failed to stop LLM service during unregister")
    else:
        logger.info("LLM service stopped successfully during unregister")

    # Stop Trellis service
    terminator = TrellisTerminator()
    tw = terminator.terminate_and_wait()
    if tw:
        logger.info("TRELLIS terminated successfully during unregister")
        with trellis_status_lock:
            trellis_status = "NOT READY"
        response, success = try_curl_health_check()
        with llm_status_lock:
            llm_status = "READY" if success and response.get("message") == "Service is ready." else "NOT READY"
    else:
        logger.error("Failed to terminate TRELLIS during unregister")

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    stop_status_threads()
    if bpy.app.timers.is_registered(update_status_ui):
        bpy.app.timers.unregister(update_status_ui)

if __name__ == "__main__":
    register()