import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    # Step 1: Resolve shell script path
    script_dir = Path(__file__).resolve().parent
    windows_script_path = script_dir / "start_llama_container.sh"

    # Convert Windows path to WSL path
    drive_letter = windows_script_path.drive[0].lower()
    wsl_script_path = f"/mnt/{drive_letter}{windows_script_path.as_posix()[2:]}"
    logging.info(f"Resolved WSL script path: {wsl_script_path}")

    # Step 2: Get NGC API Key
    ngc_api_key = subprocess.check_output(
        [sys.executable, "ngc.py"],
        stderr=subprocess.STDOUT,
        text=True
    ).strip()
    logging.info("Successfully retrieved NGC API Key.")

    # Step 3: Prepare environment + command
    bash_command = f"export NGC_API_KEY='{ngc_api_key}' && '{wsl_script_path}'"

    # Open log file for writing
    log_file_path = script_dir / "llama_container.log"
    with open(log_file_path, "w") as log_file:
        logging.info(f"Logging to {log_file_path}")

        # Launch process with Popen to stream output
        process = subprocess.Popen(
            ["wsl", "-d", "NVIDIA-Workbench", "/bin/bash", "-c", bash_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output line by line
        for line in process.stdout:
            print(line, end='')               # live output to console
            log_file.write(line)              # write to file
            log_file.flush()                  # ensure it's written immediately

        process.wait()  # wait for completion

        if process.returncode != 0:
            logging.error(f"Script exited with code {process.returncode}")
            sys.exit(process.returncode)

    logging.info("Script executed successfully.")

except subprocess.CalledProcessError as e:
    logging.error(f"Command failed: {e.cmd}")
    logging.error(f"Exit code: {e.returncode}")
    logging.error(f"Output: {e.output}")
    sys.exit(e.returncode)

except Exception as e:
    logging.exception("Unexpected error occurred")
    sys.exit(1)
