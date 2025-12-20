"""
Label Studio API client for programmatic annotation.

Uses the Label Studio SDK to create annotations via API.
"""

import json
import os
from label_studio_sdk import Client

# Configuration
LABEL_STUDIO_URL = "http://localhost:8080"


def get_api_token():
    """Get API token from Label Studio user settings or environment."""
    token = os.environ.get("LABEL_STUDIO_API_TOKEN")
    if not token:
        print("API token not found in environment.")
        print("\nTo get your API token:")
        print("1. Go to Label Studio: http://localhost:8080")
        print("2. Click on your user icon (top right)")
        print("3. Click 'Account & Settings'")
        print("4. Copy the 'Access Token'")
        print("\nThen set it:")
        print('  set LABEL_STUDIO_API_TOKEN=your_token_here')
        return None
    return token


def connect_client(api_token: str) -> Client:
    """Connect to Label Studio API."""
    return Client(url=LABEL_STUDIO_URL, api_key=api_token)


def list_projects(client: Client):
    """List all projects."""
    projects = client.get_projects()
    for p in projects:
        print(f"Project {p.id}: {p.title}")
    return projects


def get_tasks(client: Client, project_id: int):
    """Get all tasks from a project."""
    project = client.get_project(project_id)
    tasks = project.get_tasks()
    return tasks


def create_annotation(client: Client, task_id: int, annotation_result: list):
    """
    Create an annotation for a task.
    
    annotation_result should be a list of Label Studio result objects.
    """
    # Use the API directly
    import requests
    
    headers = {
        "Authorization": f"Token {client.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "result": annotation_result
    }
    
    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/tasks/{task_id}/annotations/",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 201:
        print(f"Annotation created for task {task_id}")
        return response.json()
    else:
        print(f"Error creating annotation: {response.status_code}")
        print(response.text)
        return None


def build_qbm_annotation(
    raw_text: str,
    span_start: int,
    span_end: int,
    behavior_form: str,
    agent_type: str,
    speech_mode: str,
    evaluation: str,
    deontic_signal: str,
    support_type: str,
    action_class: str = "ACT_VOLITIONAL",
    action_eval: str = "EVAL_SALIH",
    systemic: list = None,
    situational: str = "external",
) -> list:
    """
    Build a QBM annotation result in Label Studio format.
    """
    if systemic is None:
        systemic = ["SYS_GOD"]
    
    result = [
        {
            "from_name": "span_selection",
            "to_name": "quran_text",
            "type": "labels",
            "value": {
                "start": span_start,
                "end": span_end,
                "text": raw_text[span_start:span_end],
                "labels": ["BEHAVIOR_SPAN"]
            }
        },
        {
            "from_name": "behavior_form",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [behavior_form]}
        },
        {
            "from_name": "agent_type",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [agent_type]}
        },
        {
            "from_name": "speech_mode",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [speech_mode]}
        },
        {
            "from_name": "evaluation",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [evaluation]}
        },
        {
            "from_name": "quran_deontic_signal",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [deontic_signal]}
        },
        {
            "from_name": "support_type",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [support_type]}
        },
        {
            "from_name": "action_class",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [action_class]}
        },
        {
            "from_name": "action_textual_eval",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [action_eval]}
        },
        {
            "from_name": "systemic",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": systemic}
        },
        {
            "from_name": "situational",
            "to_name": "quran_text",
            "type": "choices",
            "value": {"choices": [situational]}
        },
    ]
    
    return result


if __name__ == "__main__":
    token = get_api_token()
    if token:
        client = connect_client(token)
        projects = list_projects(client)
