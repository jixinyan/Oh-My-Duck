import argparse
from dataclasses import asdict, dataclass


MODEL_APIS = ('chat-completions', 'responses')
REASONING_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')


def add_model_arguments(parser: argparse.ArgumentParser, *, max_output_tokens: int = 4096,
                        reasoning_effort: str | None = None) -> None:
    parser.add_argument('--model', type=str)
    parser.add_argument('--model-api', choices=MODEL_APIS)
    parser.add_argument('--reasoning-effort', choices=REASONING_EFFORTS, default=reasoning_effort)
    parser.add_argument('--max-output-tokens', type=int, default=max_output_tokens,
                        help='原生模型输出预算，包含 reasoning tokens（256–8192）')


def validate_model_options(model: str | None, model_api: str | None,
                           reasoning_effort: str | None, max_output_tokens: int) -> None:
    if type(max_output_tokens) is not int or not 256 <= max_output_tokens <= 8192:
        raise ValueError('模型输出 token 预算必须介于 256 和 8192 之间')
    if model is not None and (not isinstance(model, str) or not model or
                              any(char.isspace() or ord(char) < 32 for char in model)):
        raise ValueError('模型标识必须非空且不能包含空白或控制字符')
    if model_api is not None and model_api not in MODEL_APIS:
        raise ValueError('Model API must be chat-completions or responses')
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError('reasoning effort 必须使用声明的模型参数')


@dataclass(frozen=True)
class ModelSettings:
    model: str
    model_api: str
    reasoning_effort: str | None
    max_output_tokens: int

    def metadata(self) -> dict:
        return asdict(self)

    def arguments(self) -> list[str]:
        arguments = ['--model', self.model, '--model-api', self.model_api,
                     '--max-output-tokens', str(self.max_output_tokens)]
        if self.reasoning_effort is not None:
            arguments += ['--reasoning-effort', self.reasoning_effort]
        return arguments


def resolve_model_settings(configured_model: str, configured_api: str, *, model: str | None = None,
                           model_api: str | None = None, reasoning_effort: str | None = None,
                           max_output_tokens: int = 4096) -> ModelSettings:
    validate_model_options(model, model_api, reasoning_effort, max_output_tokens)
    if configured_api not in ('chat', *MODEL_APIS):
        raise ValueError('Provider wire_api 必须使用 chat、chat-completions 或 responses')
    effective_api = model_api if model_api is not None else (
        'chat-completions' if configured_api == 'chat' else configured_api)
    effective_model = configured_model if model is None else model
    validate_model_options(effective_model, effective_api, reasoning_effort, max_output_tokens)
    if effective_model is None:
        raise ValueError('Provider 必须指定模型标识')
    return ModelSettings(effective_model, effective_api, reasoning_effort, max_output_tokens)
