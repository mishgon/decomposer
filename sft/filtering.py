"""The same non-thinking student representation for batch and live filtering."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from langchain_core.messages import convert_to_messages, messages_from_dict
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool

MAX_SEQUENCE_LENGTH = 32768


def student_messages(messages, system_message=None):
    normalized = []
    for message in messages:
        if isinstance(message, dict):
            message = (messages_from_dict([message])[0] if 'data' in message and 'type' in message
                       else convert_to_messages([message])[0])
        normalized.append(message)
    if system_message is not None:
        normalized.insert(0, system_message)
    rows = convert_to_openai_messages(normalized, text_format='string')
    # Training excludes teacher reasoning. Never mutate the saved raw messages.
    for row in rows:
        for key in ('reasoning', 'reasoning_content', 'teacher_reasoning'):
            row.pop(key, None)
        for call in row.get('tool_calls', []):
            arguments = call['function']['arguments']
            if isinstance(arguments, str):
                call['function']['arguments'] = json.loads(arguments)
    return rows


@dataclass(frozen=True)
class SequenceCheck:
    tokens: int
    limit: int

    @property
    def exceeded(self):
        return self.tokens > self.limit


class StudentSequenceFilter:
    def __init__(self, tokenizer, limit=MAX_SEQUENCE_LENGTH):
        self.tokenizer = tokenizer
        self.limit = limit

    def format(self, system_message, tools):
        return {'system_message': system_message.content if system_message else None,
                'tools': [convert_to_openai_tool(tool) for tool in tools],
                'tokenizer': self.tokenizer.name_or_path,
                'chat_template_sha256': hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                'include_reasoning': False}

    def check(self, messages, format):
        template_hash = hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()
        if format['chat_template_sha256'] != template_hash:
            raise ValueError('Student chat template differs from collection; use the same tokenizer snapshot')
        from langchain_core.messages import SystemMessage
        system = SystemMessage(format['system_message']) if format['system_message'] is not None else None
        rows = student_messages(messages, system)
        ids = self.tokenizer.apply_chat_template(rows, tools=format['tools'] or None,
            tokenize=True, return_dict=False, add_generation_prompt=False,
            enable_thinking=False, preserve_thinking=False)
        return SequenceCheck(len(ids), self.limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--tokenizer', required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    checker = StudentSequenceFilter(AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True))
    for path in args.traces:
        trace = json.loads(path.read_text())
        check = checker.check(trace['messages'], trace['sft_format'])
        print(json.dumps({'path': str(path), **asdict(check), 'keep': not check.exceeded}))


if __name__ == '__main__':
    main()
