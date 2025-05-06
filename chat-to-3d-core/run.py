import sys
from pathlib import Path

# Add the project root directory to the Python path
# project_root = Path(__file__).parent.parent
# print(f"Project root: {project_root}")
# sys.path.append(str(project_root))

# --- Calculate important paths ---
# Path to the directory containing run.py (chat-to-3d-core)
current_dir = Path(__file__).parent
print(f"Current script directory: {current_dir}")

# Path to the project root (F:\poc\chat-to-3d)
# This is two levels up from the script's location
project_root = current_dir.parent
print(f"Project root: {project_root}")

# Path to the Trellis submodule directory (F:\poc\chat-to-3d\trellis)
trellis_submodule_path = project_root / "trellis"
trellis_submodule_path_str = str(trellis_submodule_path.resolve()) # Get the absolute path string
print(f"Trellis submodule path: {trellis_submodule_path_str}")

# Add the project root to sys.path.
# This allows importing 'chat_to_3d_core' as a top-level package.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to sys.path: {project_root}")


# Add the trellis submodule directory to sys.path.
# This directory contains the actual 'trellis' Python package folder (trellis/trellis)
if trellis_submodule_path_str not in sys.path:
    sys.path.insert(0, trellis_submodule_path_str)
    print(f"Added trellis submodule path to sys.path: {trellis_submodule_path_str}")



from interface import SceneGeneratorInterface


def main():
    """Main entry point for the 3D Scene Generator application."""
    interface = SceneGeneratorInterface()
    interface.launch()


if __name__ == "__main__":
    main()
