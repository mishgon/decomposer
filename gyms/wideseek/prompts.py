from decomposer.prompts import AGENT_SYSTEM_PROMPT as BASE_AGENT_SYSTEM_PROMPT

AGENT_SYSTEM_PROMPT = """The search and access tools use an offline Wikipedia corpus from 2018. Do not claim to have checked sources you did not read.

""" + BASE_AGENT_SYSTEM_PROMPT
