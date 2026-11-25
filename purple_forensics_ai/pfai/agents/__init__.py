"""Red dinámica de agentes Purple Team sobre bus asíncrono publish/subscribe."""
from pfai.agents.orchestrator import (
    ALLOWED_TOOLS,
    PLAYBOOKS,
    Agent,
    Bus,
    DetectorAgent,
    ForensicAgent,
    Message,
    OpenClawAgent,
    RedTeamAgent,
    ReportAgent,
    SandboxAgent,
    TriageAgent,
    dynamic_plan,
)

__all__ = [
    "ALLOWED_TOOLS",
    "Agent",
    "Bus",
    "DetectorAgent",
    "ForensicAgent",
    "Message",
    "OpenClawAgent",
    "PLAYBOOKS",
    "RedTeamAgent",
    "ReportAgent",
    "SandboxAgent",
    "TriageAgent",
    "dynamic_plan",
]
