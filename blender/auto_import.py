import bpy
import os
from pathlib import Path

# Get user's home directory
HOME_DIR = str(Path.home())

# Directories
WATCH_DIR = os.path.join(HOME_DIR, ".trellis", "assets")  # Where generated GLB files are stored
ASSET_LIBRARY_DIR = os.path.join(HOME_DIR, ".trellis", "blender_assets")  # Where Blender assets are stored
COMBINED_BLEND_FILE = os.path.join(ASSET_LIBRARY_DIR, "trellis_assets.blend")

# Create directories if they don't exist
os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(ASSET_LIBRARY_DIR, exist_ok=True)

def register_asset_library():
    """Register the asset library with Blender's preferences"""
    try:
        # Get absolute path for consistency
        absolute_path = os.path.abspath(ASSET_LIBRARY_DIR)
        
        # Check if already registered
        library_exists = False
        asset_libraries = bpy.context.preferences.filepaths.asset_libraries
        for library in asset_libraries:
            if os.path.abspath(library.path) == absolute_path:
                library_exists = True
                print(f"Asset library already registered: {absolute_path}")
                break
        
        # Register if not found
        if not library_exists:
            # Add the library (method depends on Blender version)
            library = None
            if hasattr(asset_libraries, 'new'):
                library = asset_libraries.new(name="TRELLIS Assets")
            else:
                library = asset_libraries.add()
                library.name = "TRELLIS Assets"
            
            library.path = absolute_path
            bpy.ops.wm.save_userpref()
            print(f"Successfully registered asset library: {absolute_path}")
        
        return True
    except Exception as e:
        print(f"Error registering asset library: {e}")
        print(f"Please add manually: Edit > Preferences > File Paths > Asset Libraries")
        return False

def process_glb_file(glb_file,overwrite_existing):
    """Process a single GLB file and create an asset"""
    filepath = os.path.join(WATCH_DIR, glb_file)
    print(f"Processing: {glb_file}")
    
    # Get the file basename for naming
    file_basename = os.path.splitext(glb_file)[0]
    
    # Check if an object with this name already exists and delete it
    if file_basename in bpy.data.objects:
        if not overwrite_existing:
            print(f"Asset '{file_basename}' already exists. Skipping...")
            return None
        else:
            print(f"Asset '{file_basename}' already exists. Overwriting...")
            old_obj = bpy.data.objects[file_basename]
            bpy.data.objects.remove(old_obj, do_unlink=True)
    
    # Create a temporary collection for processing
    temp_collection = bpy.data.collections.new(f"temp_{file_basename}")
    # Link the temporary collection to the scene but hide it
    bpy.context.scene.collection.children.link(temp_collection)
    temp_collection.hide_viewport = True
    temp_collection.hide_render = True
    
    # Deselect all objects before import
    bpy.ops.object.select_all(action='DESELECT')
    
    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=filepath)
    
    # Find imported objects
    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        print(f"No objects were imported from {glb_file}. Skipping.")
        bpy.data.collections.remove(temp_collection)
        return None
    
    # Move all objects to the temporary collection
    for obj in imported_objects:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        temp_collection.objects.link(obj)
    
    # Filter to keep only mesh objects
    mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
    
    # Delete non-mesh objects (empties, etc.)
    non_mesh_objects = [obj for obj in imported_objects if obj.type != 'MESH']
    if non_mesh_objects:
        print(f"Removing {len(non_mesh_objects)} non-mesh objects")
        for obj in non_mesh_objects:
            bpy.data.objects.remove(obj)
    
    # If no mesh objects found, skip this file
    if not mesh_objects:
        print(f"No mesh objects found in {glb_file}. Skipping.")
        bpy.data.collections.remove(temp_collection)
        return None
    
    # Select all mesh objects in the collection
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    
    # Set the active object (needed for join operation)
    bpy.context.view_layer.objects.active = next((obj for obj in mesh_objects), None)
    
    # Join them into a single object if there are multiple
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    
    # After join, only one object remains (active object)
    merged_obj = bpy.context.active_object
    
    # Rename object and its mesh data with proper name
    merged_obj.name = file_basename
    if merged_obj.data:
        merged_obj.data.name = f"{file_basename}_mesh"
    
    # Set origin to geometry
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # Mark merged object as asset
    merged_obj.asset_mark()
    if hasattr(merged_obj, 'asset_data'):
        merged_obj.asset_data.description = f"3D model generated by TRELLIS from {glb_file}"
        merged_obj.asset_data.tags.new("TRELLIS")
    
    # Unlink the object from the temporary collection
    temp_collection.objects.unlink(merged_obj)
    
    # Delete the temporary collection
    bpy.data.collections.remove(temp_collection)
    
    print(f"Created asset object: {file_basename}")
    return merged_obj

def process_all_glb_files(overwrite_existing=False):
    """Process all GLB files in the current Blender session"""
    # Get all GLB files
    if not os.path.exists(WATCH_DIR):
        print(f"Watch directory does not exist: {WATCH_DIR}")
        return
    
    glb_files = [f for f in os.listdir(WATCH_DIR) if f.endswith('.glb')]
    
    if not glb_files:
        print("No GLB files found in the watch directory.")
        return
    
    print(f"Found {len(glb_files)} GLB file(s). Processing...")
    
    # Process each file
    for glb_file in glb_files:
        process_glb_file(glb_file,overwrite_existing)
    
    # Save the combined file
    print(f"Saving all assets to: {COMBINED_BLEND_FILE}")
    bpy.ops.wm.save_as_mainfile(filepath=COMBINED_BLEND_FILE)
    print("All assets saved successfully")

def main():
    """Main function"""
    # Register the asset library
    register_asset_library()
    
    # Process all GLB files
    process_all_glb_files(overwrite_existing=False)
    
    # Final message
    print("All files processed. Assets available in the Asset Browser under 'TRELLIS Assets'")

# Entry point
if __name__ == "__main__":
    main()