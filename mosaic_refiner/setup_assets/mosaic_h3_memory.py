"""App-owned graph boundary: release Comfy-managed models between H3 stages."""
class ConnectionType(str):
    def __ne__(self, other):
        return False


class ReleaseModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'value': (ConnectionType('*'),)}}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = (ConnectionType('*'),)
    FUNCTION = 'release'
    CATEGORY = 'Mosaic Refiner/memory'

    def release(self, value):
        from comfy import model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        return (value,)


NODE_CLASS_MAPPINGS = {'UnloadAllModels': ReleaseModels}
