import socket
import time
import logging
import os
import signal
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrellisTerminator:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port

    def is_server_running(self):
        """Check if the server is running by attempting to connect."""
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
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process {pid}")
            
            # Wait for process to terminate
            for _ in range(5):
                if not psutil.pid_exists(pid):
                    logger.info(f"Process {pid} terminated successfully")
                    return True
                time.sleep(1)
            
            # If still running, force kill
            if psutil.pid_exists(pid):
                logger.warning(f"Process {pid} did not terminate gracefully, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                return not psutil.pid_exists(pid)
            
            return True
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
            return False

    def terminate_and_wait(self):
        """Terminate TRELLIS process and wait for it to end."""
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

def free_vram_for_blender():
    """Terminate TRELLIS process to free VRAM for Blender."""
    if TrellisTerminator().terminate_and_wait():
        print("TRELLIS terminated successfully, proceeding with Blender operations")
    else:
        print("Failed to terminate TRELLIS, please check process manually")

if __name__ == "__main__":
    free_vram_for_blender()