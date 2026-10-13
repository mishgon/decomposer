from decomposer.prompts import AGENT_SYSTEM_PROMPT as BASE_AGENT_SYSTEM_PROMPT

AGENT_SYSTEM_PROMPT = """Your workspace directory is `{workspace_dir}`. Resolve relative file paths against this directory unless the task specifies otherwise.

""" + BASE_AGENT_SYSTEM_PROMPT
