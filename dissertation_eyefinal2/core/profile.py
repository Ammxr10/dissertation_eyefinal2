"""
core/profile.py - User prescription profile management.
Handles saving to and loading from JSON files.
"""
import json

class ProfileManager:
    """Utility for saving and loading user prescription profiles as JSON."""
    def save_profile(self, filepath: str, profile: dict):
        data = {
            "sphere": profile.get("sphere", 0.0),
            "cylinder": profile.get("cylinder", 0.0),
            "axis": profile.get("axis", 0),
            "coma": profile.get("coma", 0.0),
            "spherical_aberration": profile.get("spherical_aberration", 0.0)
        }
        with open(filepath, 'w') as fp:
            json.dump(data, fp, indent=4)

    def load_profile(self, filepath: str) -> dict:
        with open(filepath, 'r') as fp:
            data = json.load(fp)
        profile = {
            "sphere": data.get("sphere", 0.0),
            "cylinder": data.get("cylinder", 0.0),
            "axis": data.get("axis", 0),
            "coma": data.get("coma", 0.0),
            "spherical_aberration": data.get("spherical_aberration", 0.0)
        }
        return profile
