from .chat_vl import ChatVLAdapter


class HunyuanOCRAdapter(ChatVLAdapter):
    name = "hunyuanocr"
    model_class = "HunYuanVLForConditionalGeneration"
    trust_remote_code = True
    processor_kwargs = {"use_fast": False}
