from .chat_vl import ChatVLAdapter


class AINAdapter(ChatVLAdapter):
    name = "ain"
    model_class = "Qwen2VLForConditionalGeneration"

