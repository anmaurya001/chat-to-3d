"""Object data model for scene objects."""

import uuid
from datetime import datetime


class SceneObject:
    """Represents an object in a 3D scene."""
    
    def __init__(self, name, description="", image_path="", prompt=""):
        """Initialize a scene object."""
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.image_path = image_path
        self.prompt = prompt
        self.created_at = datetime.now()
        self.status = "pending"  # pending, generated, error
    
    def to_dict(self):
        """Convert object to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create object from dictionary."""
        obj = cls(
            name=data["name"],
            description=data.get("description", ""),
            image_path=data.get("image_path", ""),
            prompt=data.get("prompt", "")
        )
        obj.id = data["id"]
        obj.status = data.get("status", "pending")
        if "created_at" in data:
            obj.created_at = datetime.fromisoformat(data["created_at"])
        return obj 