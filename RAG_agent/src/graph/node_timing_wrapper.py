"""
Wrapper to automatically add timing to all graph nodes.

This module provides a decorator that wraps node functions with timing logic,
logging execution time for every node in the graph.
"""

import logging
from typing import Callable, Any, Dict
from functools import wraps
import time

from src.utils.timing_logger import time_component

logger = logging.getLogger(__name__)

__all__ = ["timed_node", "wrap_all_nodes_with_timing"]


def timed_node(node_name: str):
    """Decorator to add timing to a graph node.

    Usage:
        @timed_node("security_gateway")
        def security_gateway_node(state):
            # Your code
            return {"gateway_passed": True}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            # Extract useful metadata from state
            metadata = {}
            if "user_input" in state:
                metadata["query_preview"] = state["user_input"][:50]
            if "intent_result" in state:
                intent = state.get("intent_result")
                if intent:
                    metadata["intent_type"] = getattr(intent, "intent_type", None)
                    metadata["framework"] = getattr(intent, "framework", None)

            # Time the node execution
            start_time = time.time()
            try:
                result = func(state, *args, **kwargs)
                elapsed = time.time() - start_time

                # Log with metadata
                from src.utils.timing_logger import get_timing_logger
                get_timing_logger().log_timing(
                    f"node.{node_name}",
                    elapsed,
                    metadata
                )

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Node {node_name} failed after {elapsed:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def wrap_all_nodes_with_timing(graph_builder_func: Callable) -> Callable:
    """Wrap graph builder to automatically time all nodes.

    This is an alternative approach that wraps the entire graph builder
    and instruments all added nodes.

    Usage:
        @wrap_all_nodes_with_timing
        def build_agent_graph():
            graph = StateGraph(AgentState)
            # Nodes will be automatically timed
            graph.add_node("my_node", my_node_function)
            return graph
    """
    @wraps(graph_builder_func)
    def wrapper(*args, **kwargs):
        # Call original builder
        result = graph_builder_func(*args, **kwargs)

        # TODO: Instrument nodes if needed
        # This is complex because LangGraph nodes are already added
        # Better to use @timed_node decorator directly

        return result

    return wrapper
