"""Build relationship graph from hierarchical JSON documents.

This module reads JSON files and builds a graph structure representing:
1. Parent-child relationships (hierarchy from headings)
2. Link relationships (which sections link to which)

NOTE: This functionality is currently unused by the RAG system.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.document import DocumentSection
from settings import OUTPUT_DIR


def generate_section_id(file_path: Path, section_path: List[str]) -> str:
    """Generate unique section ID from file path and heading path.

    Args:
        file_path: Relative path to the JSON file
        section_path: List of heading titles from root to current section

    Returns:
        Unique section identifier
    """
    path_str = str(file_path).replace("\\", "/").replace(".json", "")
    if section_path:
        return f"{path_str}#{'/'.join(section_path)}"
    return path_str


def normalize_link_target(link_url: str, current_file: Path) -> Optional[str]:
    """Normalize a link URL to a section ID if it's an internal link.

    Args:
        link_url: Link URL from markdown
        current_file: Current file path for relative link resolution

    Returns:
        Section ID if internal link, None otherwise
    """
    # Skip external URLs
    if link_url.startswith("http://") or link_url.startswith("https://"):
        return None

    # Handle anchor links (e.g., #section-name)
    if link_url.startswith("#"):
        # For now, just return the anchor as-is
        return link_url

    # Handle relative file links
    try:
        # Resolve relative to current file
        current_dir = current_file.parent
        target_path = (current_dir / link_url).resolve()

        # Convert to relative path from json_files directory
        json_base = OUTPUT_DIR / "json_files"
        if json_base in target_path.parents or target_path == json_base:
            rel_path = target_path.relative_to(json_base)
            return str(rel_path.with_suffix("")).replace("\\", "/")
    except (ValueError, RuntimeError):
        pass

    return None


def extract_section_links(section: DocumentSection, file_path: Path) -> List[Dict[str, str]]:
    """Extract all links from a document section.

    Args:
        section: Document section to analyze
        file_path: Path to the source file

    Returns:
        List of link dictionaries with source/target information
    """
    links = []

    for link in section.links:
        target_id = normalize_link_target(link.url, file_path)
        if target_id:
            links.append({
                "source": generate_section_id(file_path, []),  # Use file as source for now
                "target": target_id,
                "type": "link",
                "text": link.text
            })

    return links


def traverse_section_graph(
    section: DocumentSection,
    file_path: Path,
    nodes: List[Dict],
    edges: List[Dict],
    visited: Set[str],
    section_path: List[str] = None
) -> None:
    """Recursively traverse section hierarchy and build graph nodes/edges.

    Args:
        section: Current section to process
        file_path: Source file path
        nodes: List to append graph nodes to
        edges: List to append graph edges to
        visited: Set of visited section IDs
        section_path: Path of parent section titles
    """
    if section_path is None:
        section_path = []

    current_path = section_path + [section.title]
    section_id = generate_section_id(file_path, current_path)

    # Skip if already visited (avoid duplicates)
    if section_id in visited:
        return
    visited.add(section_id)

    # Create node
    node = {
        "id": section_id,
        "title": section.title,
        "level": section.level,
        "content_length": len(section.content),
        "file_path": str(file_path),
        "section_path": current_path.copy()
    }
    nodes.append(node)

    # Create parent-child relationship
    if section_path:
        parent_id = generate_section_id(file_path, section_path)
        edges.append({
            "source": parent_id,
            "target": section_id,
            "type": "hierarchy"
        })

    # Extract links from this section
    section_links = extract_section_links(section, file_path)
    edges.extend(section_links)

    # Recursively process subsections
    for subsection in section.subsections:
        traverse_section_graph(subsection, file_path, nodes, edges, visited, current_path)


def build_graph_from_json_file(json_file: Path) -> Dict:
    """Build graph structure from a single JSON file.

    Args:
        json_file: Path to JSON file

    Returns:
        Graph dictionary with nodes and edges
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert dict back to DocumentSection
        def dict_to_section(d: Dict) -> DocumentSection:
            return DocumentSection(
                title=d["title"],
                level=d["level"],
                content=d["content"],
                file_path=Path(d["file_path"]),
                links=[],  # Links not preserved in JSON
                subsections=[dict_to_section(sub) for sub in d.get("subsections", [])]
            )

        root_section = dict_to_section(data)

        nodes = []
        edges = []
        visited = set()

        # Process the document
        traverse_section_graph(root_section, json_file, nodes, edges, visited)

        return {
            "file": str(json_file),
            "nodes": nodes,
            "edges": edges
        }

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return {"file": str(json_file), "nodes": [], "edges": []}


def build_graph_from_json_files(json_dir: Path) -> Dict[str, List]:
    """Build complete graph from all JSON files in directory.

    Args:
        json_dir: Directory containing JSON files

    Returns:
        Combined graph with all nodes and edges
    """
    all_nodes = []
    all_edges = []

    # Find all JSON files
    json_files = list(json_dir.rglob("*.json"))

    print(f"Processing {len(json_files)} JSON files...")

    for json_file in json_files:
        graph = build_graph_from_json_file(json_file)
        all_nodes.extend(graph["nodes"])
        all_edges.extend(graph["edges"])

    return {
        "nodes": all_nodes,
        "edges": all_edges
    }


def save_graph(graph: Dict[str, List], output_path: Path) -> None:
    """Save graph to a pickle file.

    Args:
        graph: Graph dictionary with nodes and edges
        output_path: Output file path
    """
    import pickle

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Graph saved to {output_path}")
    print(f"Nodes: {len(graph['nodes'])}, Edges: {len(graph['edges'])}")


if __name__ == "__main__":
    # Example usage
    json_dir = OUTPUT_DIR / "json_files"
    graph_dir = OUTPUT_DIR / "graph"

    print("Building document relationship graph...")
    graph = build_graph_from_json_files(json_dir)

    graph_file = graph_dir / "document_graph.pkl"
    save_graph(graph, graph_file)

    print(f"Graph complete: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
