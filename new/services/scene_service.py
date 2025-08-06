"""Scene service for managing scene data and objects."""

from models.scene import Scene
from models.object_data import SceneObject


class SceneService:
    """Service for managing scene data and operations."""
    
    def __init__(self):
        """Initialize the scene service."""
        self.current_scene = Scene()
    
    def add_object(self, name, description="", image_path=""):
        """Add an object to the current scene."""
        obj = SceneObject(name=name, description=description, image_path=image_path)
        self.current_scene.add_object(obj)
        return obj
    
    def remove_object(self, object_id):
        """Remove an object from the current scene."""
        return self.current_scene.remove_object(object_id)
    
    def get_objects(self):
        """Get all objects in the current scene."""
        return self.current_scene.objects
    
    def get_object_names(self):
        """Get names of all objects in the current scene."""
        return [obj.name for obj in self.current_scene.objects]
    
    def clear_scene(self):
        """Clear all objects from the current scene."""
        self.current_scene.clear()
    
    def update_scene_description(self, description):
        """Update the scene description."""
        self.current_scene.description = description
    
    def get_scene_summary(self):
        """Get a summary of the current scene."""
        return {
            "name": self.current_scene.name,
            "description": self.current_scene.description,
            "object_count": len(self.current_scene.objects),
            "objects": self.get_object_names()
        } 