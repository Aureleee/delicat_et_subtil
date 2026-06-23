class AnyPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def run(self, value):
        return (value,)


NODE_CLASS_MAPPINGS        = {"AnyPassthrough": AnyPassthrough}
NODE_DISPLAY_NAME_MAPPINGS = {"AnyPassthrough": "Passthrough (Any)"}
