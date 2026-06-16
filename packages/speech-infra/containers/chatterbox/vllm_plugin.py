"""vLLM plugin to register ChatterboxT3 model and custom tokenizers.

vLLM 0.10.0's V1 engine runs EngineCore in a subprocess that discovers
out-of-tree models via the `vllm.general_plugins` entry point group.
This plugin ensures model/tokenizer registration happens in all processes.
"""


def register():
    from chatterbox_vllm.models.t3.t3 import T3VllmModel
    from vllm import ModelRegistry
    from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

    ModelRegistry.register_model("ChatterboxT3", T3VllmModel)
    TokenizerRegistry.register(
        "EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer"
    )
    TokenizerRegistry.register(
        "MtlTokenizer", "chatterbox_vllm.models.t3.mtltokenizer", "MTLTokenizer"
    )
