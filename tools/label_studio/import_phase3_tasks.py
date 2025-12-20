#!/usr/bin/env python3
"""
Import Phase 3 tasks into Label Studio.

Usage:
    set LABEL_STUDIO_API_TOKEN=your_token_here
    python tools/label_studio/import_phase3_tasks.py --project-id <ID>
"""

import argparse
import json
import os
import requests
from pathlib import Path

LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")


def get_api_token():
    """Get API token from environment."""
    token = os.environ.get("LABEL_STUDIO_API_TOKEN")
    if not token:
        print("ERROR: LABEL_STUDIO_API_TOKEN not set")
        print("\nTo get your API token:")
        print("1. Go to Label Studio: http://localhost:8080")
        print("2. Click on your user icon (top right)")
        print("3. Click 'Account & Settings'")
        print("4. Copy the 'Access Token'")
        print("\nThen set it:")
        print('  set LABEL_STUDIO_API_TOKEN=your_token_here')
        return None
    return token


def import_tasks(project_id: int, tasks_file: str, api_token: str):
    """Import tasks from JSONL file into Label Studio project."""
    
    # Read tasks from JSONL
    tasks = []
    with open(tasks_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                # Label Studio expects just the 'data' field for import
                tasks.append(task["data"])
    
    print(f"Loaded {len(tasks)} tasks from {tasks_file}")
    
    # Import via API
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=tasks)
    
    if response.status_code == 201:
        result = response.json()
        print(f"Successfully imported {result.get('task_count', len(tasks))} tasks")
        return True
    else:
        print(f"Error importing tasks: {response.status_code}")
        print(response.text)
        return False


def list_projects(api_token: str):
    """List all projects to help user find project ID."""
    url = f"{LABEL_STUDIO_URL}/api/projects"
    headers = {"Authorization": f"Token {api_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        projects = response.json().get("results", [])
        print("\nAvailable projects:")
        for p in projects:
            print(f"  ID {p['id']}: {p['title']} ({p.get('task_number', 0)} tasks)")
        return projects
    else:
        print(f"Error listing projects: {response.status_code}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Import Phase 3 tasks into Label Studio")
    parser.add_argument("--project-id", type=int, help="Label Studio project ID")
    parser.add_argument("--tasks-file", default="label_studio/phase3_550_tasks.jsonl",
                        help="Path to tasks JSONL file")
    parser.add_argument("--list-projects", action="store_true", help="List available projects")
    args = parser.parse_args()
    
    api_token = get_api_token()
    if not api_token:
        return 1
    
    if args.list_projects or not args.project_id:
        projects = list_projects(api_token)
        if not args.project_id:
            print("\nUsage: python import_phase3_tasks.py --project-id <ID>")
            return 0
    
    if args.project_id:
        success = import_tasks(args.project_id, args.tasks_file, api_token)
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
