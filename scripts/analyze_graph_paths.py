#!/usr/bin/env python3
"""Analyze behavior-to-behavior causal paths in the semantic graph."""

import json
from collections import deque

g = json.load(open('data/graph/semantic_graph_v2.json', encoding='utf-8'))

# Find behavior-to-behavior CAUSES edges
beh_causes = [e for e in g['edges'] 
              if e.get('edge_type') in ['CAUSES', 'LEADS_TO', 'STRENGTHENS']
              and e['source'].startswith('BEH_')
              and e['target'].startswith('BEH_')]

print(f'Behavior-to-behavior causal edges: {len(beh_causes)}')

# Build adjacency for behavior-only graph
adj = {}
for e in beh_causes:
    src = e['source']
    if src not in adj:
        adj[src] = []
    adj[src].append(e)

print(f'Behaviors with outgoing edges: {len(adj)}')

# Check edges from HEEDLESSNESS
heed_edges = [e for e in beh_causes if 'HEEDLESSNESS' in e['source']]
print(f'\nEdges from HEEDLESSNESS: {len(heed_edges)}')
for e in heed_edges:
    print(f'  -> {e["target"]} ({e["edge_type"]})')

# Check edges TO DISBELIEF
disb_edges = [e for e in beh_causes if 'DISBELIEF' in e['target']]
print(f'\nEdges TO DISBELIEF: {len(disb_edges)}')
for e in disb_edges[:10]:
    print(f'  {e["source"]} -> DISBELIEF ({e["edge_type"]})')

# Try to find behavior-only paths from heedlessness to disbelief
source = 'BEH_COG_HEEDLESSNESS'
target = 'BEH_SPI_DISBELIEF'

print(f'\nSearching behavior-only paths: {source} -> {target}')

paths = []
queue = deque([(source, [source], [])])
visited = set()

while queue and len(paths) < 10:
    current, path_nodes, path_edges = queue.popleft()
    
    if current == target and len(path_nodes) > 1:
        paths.append({'nodes': path_nodes, 'edges': path_edges, 'hops': len(path_nodes) - 1})
        continue
    
    if len(path_nodes) > 5:
        continue
    
    for edge in adj.get(current, []):
        next_node = edge['target']
        if next_node not in path_nodes:
            queue.append((next_node, path_nodes + [next_node], path_edges + [edge]))

print(f'Behavior-only paths found: {len(paths)}')
for i, p in enumerate(paths[:5]):
    print(f'\nPath {i+1} ({p["hops"]} hops):')
    print(f'  {" -> ".join(p["nodes"])}')
    for edge in p['edges']:
        ev = edge.get('evidence', [{}])[0]
        print(f'    Evidence: {ev.get("verse_key")} from {ev.get("source")}')
