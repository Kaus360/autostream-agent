"""
agent/tools.py
Contains the mock_lead_capture tool that simulates writing a captured
lead to a CRM or database. This function must NEVER be called until
name, email, AND platform are all present in AgentState.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates persisting a sales lead to a CRM.

    Args:
        name:     Full name of the prospective customer.
        email:    Email address of the prospective customer.
        platform: Primary publishing platform (e.g. YouTube, TikTok).

    Returns:
        A confirmation string (also printed to stdout for visibility).
    """
    message = (
        f"✅ Lead captured successfully!\n"
        f"   Name     : {name}\n"
        f"   Email    : {email}\n"
        f"   Platform : {platform}"
    )
    return message
