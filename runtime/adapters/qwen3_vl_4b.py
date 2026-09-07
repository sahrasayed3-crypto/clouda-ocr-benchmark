from .chat_vl import ChatVLAdapter


class Qwen3VLAdapter(ChatVLAdapter):
    name = "qwen3_vl_4b"
    model_class = "Qwen3VLForConditionalGeneration"

