"""Scene data model."""

from datetime import datetime
from typing import List
from .object_data import SceneObject


class Scene:
    """Represents a 3D scene with objects."""
    
    def __init__(self, name="", description=""):
        """Initialize a new scene."""
        self.name = name or f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.description = description
        self.objects: List[SceneObject] = []
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
    
    def add_object(self, obj: SceneObject):
        """Add an object to the scene."""
        self.objects.append(obj)
        self.modified_at = datetime.now()
    
    def remove_object(self, object_id: str):
        """Remove an object by ID."""
        original_count = len(self.objects)
        self.objects = [obj for obj in self.objects if obj.id != object_id]
        if len(self.objects) < original_count:
            self.modified_at = datetime.now()
            return True
        return False
    
    def clear(self):
        """Clear all objects from the scene."""
        self.objects.clear()
        self.modified_at = datetime.now()
    
    def to_dict(self):
        """Convert scene to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "objects": [obj.to_dict() for obj in self.objects],
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat()
        } 