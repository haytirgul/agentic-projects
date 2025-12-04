"""Visualize the LangGraph agent flow.

This script generates a visual representation of the agent graph
to help understand the workflow and node connections.

Usage:
    python scripts/utilities/visualize_graph.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def visualize_graph():
    """Generate and save graph visualization."""
    try:
        # Import graph builder
        from src.agent.graph import build_agent_graph

        logger.info("Building agent graph...")
        graph = build_agent_graph()

        logger.info("Compiling graph...")
        compiled_graph = graph.compile()

        # Try to generate visualization
        try:
            # Get Mermaid diagram
            mermaid_code = compiled_graph.get_graph().draw_mermaid()

            # Save to file
            output_file = project_root / "documents" / "agent_graph.mmd"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                f.write(mermaid_code)

            logger.info(f"✅ Graph visualization saved to {output_file}")
            logger.info("")
            logger.info("To view the graph:")
            logger.info("  1. Visit https://mermaid.live/")
            logger.info("  2. Paste the contents of agent_graph.mmd")
            logger.info("  3. View the interactive diagram")

        except Exception as e:
            logger.warning(f"Could not generate Mermaid diagram: {e}")
            logger.info("")
            logger.info("Displaying graph structure:")
            print(compiled_graph.get_graph())

    except ImportError as e:
        logger.error(f"Failed to import graph: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    logger.info("Graph Visualization Tool")
    logger.info("=" * 50)

    visualize_graph()


if __name__ == "__main__":
    main()
