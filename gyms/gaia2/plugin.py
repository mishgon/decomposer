"""ARE ``--agent-plugin`` factory for Decomposer."""

from __future__ import annotations

from typing import Any

from are.simulation.agents.agent_builder import AbstractAgentBuilder
from are.simulation.agents.agent_config_builder import AbstractAgentConfigBuilder
from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactBaseAgentConfig,
)

from .broker import ToolStateBroker
from .proxy import DecomposerProxyAgent


class DecomposerConfigBuilder(AbstractAgentConfigBuilder):
    def build(self, agent_name: str):
        if agent_name != "decomposer":
            raise ValueError(
                f"Decomposer plugin does not provide agent {agent_name!r}."
            )
        return ARESimulationReactAgentConfig(
            agent_name=agent_name,
            base_agent_config=ARESimulationReactBaseAgentConfig(),
        )


class DecomposerAgentBuilder(AbstractAgentBuilder):
    def __init__(self, config: dict[str, Any], broker: ToolStateBroker) -> None:
        self.config = config
        self.broker = broker

    def list_agents(self) -> list[str]:
        return ["decomposer"]

    def build(self, agent_config, env=None, mock_responses=None):
        if agent_config.get_agent_name() != "decomposer":
            raise ValueError(
                f"Decomposer plugin does not provide {agent_config.get_agent_name()!r}."
            )
        if env is None:
            raise ValueError("Decomposer proxy requires an Environment.")
        return DecomposerProxyAgent(self.broker, env, self.config)


def create_plugin(config: dict[str, Any]):
    """Return ARE's existing config/agent builder extension points."""

    broker = ToolStateBroker(
        host=str(config.get("broker_host", "127.0.0.1")),
        port=int(config.get("broker_port", 0)),
    )
    return DecomposerConfigBuilder(), DecomposerAgentBuilder(config, broker)
