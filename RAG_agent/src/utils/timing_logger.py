"""
Comprehensive timing logger for performance analysis.

This module provides utilities to track and log execution time of:
- Graph nodes
- LLM invocations
- Retrieval operations
- Tool executions
- Any custom operations

Usage:
    from src.utils.timing_logger import time_component, get_timing_report

    @time_component("my_function")
    def my_function():
        # Your code here
        pass

    # Or use context manager
    with time_component("my_operation"):
        # Your code here
        pass

    # Get timing report
    report = get_timing_report()
    print(report)
"""

import time
import logging
from typing import Optional, Dict, List, Any
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Thread-safe timing storage
_timing_data = defaultdict(list)
_timing_lock = threading.Lock()
_current_query_id = threading.local()


class TimingLogger:
    """Centralized timing logger with hierarchical tracking."""

    def __init__(self):
        self.timings: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.query_timings: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start_query(self, query_id: str, query_text: str):
        """Start timing a new query."""
        with self.lock:
            self.query_timings[query_id] = {
                "query": query_text[:100],
                "start_time": time.time(),
                "components": [],
                "total_time": None
            }
            _current_query_id.value = query_id

    def end_query(self, query_id: str):
        """End timing for a query."""
        with self.lock:
            if query_id in self.query_timings:
                query_data = self.query_timings[query_id]
                query_data["total_time"] = time.time() - query_data["start_time"]

    def log_timing(
        self,
        component: str,
        elapsed: float,
        metadata: Optional[Dict[str, Any]] = None,
        query_id: Optional[str] = None
    ):
        """Log timing for a component."""
        if query_id is None:
            query_id = getattr(_current_query_id, 'value', None)

        timing_entry = {
            "component": component,
            "elapsed": elapsed,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "query_id": query_id
        }

        with self.lock:
            self.timings[component].append(timing_entry)

            # Add to query-specific timings if we have a query context
            if query_id and query_id in self.query_timings:
                self.query_timings[query_id]["components"].append(timing_entry)

        # Log immediately for real-time monitoring
        metadata_str = f" ({metadata})" if metadata else ""
        logger.info(f"⏱️  [{component}] took {elapsed:.3f}s{metadata_str}")

    def get_summary(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get timing summary for a component or all components."""
        with self.lock:
            if component:
                timings = self.timings.get(component, [])
                if not timings:
                    return {}

                elapsed_times = [t["elapsed"] for t in timings]
                return {
                    "component": component,
                    "count": len(timings),
                    "total": sum(elapsed_times),
                    "avg": sum(elapsed_times) / len(elapsed_times),
                    "min": min(elapsed_times),
                    "max": max(elapsed_times),
                }

            # Summary for all components
            summary = {}
            for comp, timings in self.timings.items():
                elapsed_times = [t["elapsed"] for t in timings]
                summary[comp] = {
                    "count": len(timings),
                    "total": sum(elapsed_times),
                    "avg": sum(elapsed_times) / len(elapsed_times),
                    "min": min(elapsed_times),
                    "max": max(elapsed_times),
                }

            return summary

    def get_query_report(self, query_id: str) -> Dict[str, Any]:
        """Get detailed report for a specific query."""
        with self.lock:
            if query_id not in self.query_timings:
                return {}

            query_data = self.query_timings[query_id]
            components = query_data["components"]

            # Build hierarchy
            report = {
                "query": query_data["query"],
                "total_time": query_data["total_time"],
                "components": components,
                "breakdown": {}
            }

            # Group by component type
            for comp_data in components:
                comp_name = comp_data["component"]
                if comp_name not in report["breakdown"]:
                    report["breakdown"][comp_name] = {
                        "count": 0,
                        "total_time": 0,
                        "timings": []
                    }

                report["breakdown"][comp_name]["count"] += 1
                report["breakdown"][comp_name]["total_time"] += comp_data["elapsed"]
                report["breakdown"][comp_name]["timings"].append(comp_data["elapsed"])

            return report

    def print_report(self, query_id: Optional[str] = None):
        """Print formatted timing report."""
        if query_id:
            report = self.get_query_report(query_id)
            if not report:
                print(f"No timing data for query {query_id}")
                return

            print("\n" + "=" * 80)
            print(f"TIMING REPORT FOR QUERY: {report['query']}")
            print("=" * 80)
            print(f"Total Time: {report['total_time']:.3f}s")
            print("\nBreakdown by Component:")
            print("-" * 80)

            # Sort by total time
            sorted_components = sorted(
                report["breakdown"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True
            )

            for comp_name, data in sorted_components:
                avg_time = data["total_time"] / data["count"]
                pct = (data["total_time"] / report["total_time"]) * 100
                print(f"{comp_name:40s} {data['total_time']:7.3f}s ({pct:5.1f}%) "
                      f"[{data['count']}x, avg={avg_time:.3f}s]")

            print("=" * 80)

        else:
            # Print overall summary
            summary = self.get_summary()
            if not summary:
                print("No timing data collected")
                return

            print("\n" + "=" * 80)
            print("OVERALL TIMING SUMMARY")
            print("=" * 80)

            # Sort by total time
            sorted_components = sorted(
                summary.items(),
                key=lambda x: x[1]["total"],
                reverse=True
            )

            for comp_name, data in sorted_components:
                print(f"{comp_name:40s} {data['total']:7.3f}s "
                      f"[{data['count']}x, avg={data['avg']:.3f}s, "
                      f"min={data['min']:.3f}s, max={data['max']:.3f}s]")

            print("=" * 80)

    def reset(self):
        """Clear all timing data."""
        with self.lock:
            self.timings.clear()
            self.query_timings.clear()


# Global timing logger instance
_timing_logger = TimingLogger()


def get_timing_logger() -> TimingLogger:
    """Get the global timing logger instance."""
    return _timing_logger


def start_query_timing(query_id: str, query_text: str):
    """Start timing a new query."""
    _timing_logger.start_query(query_id, query_text)


def end_query_timing(query_id: str):
    """End timing for a query."""
    _timing_logger.end_query(query_id)


def get_timing_report(query_id: Optional[str] = None) -> str:
    """Get formatted timing report."""
    if query_id:
        report = _timing_logger.get_query_report(query_id)
        return str(report)
    return str(_timing_logger.get_summary())


def print_timing_report(query_id: Optional[str] = None):
    """Print timing report to console."""
    _timing_logger.print_report(query_id)


@contextmanager
def time_component(component_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for timing a code block.

    Usage:
        with time_component("database_query", {"table": "users"}):
            # Your code here
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        _timing_logger.log_timing(component_name, elapsed, metadata)


def time_function(component_name: Optional[str] = None):
    """Decorator for timing function execution.

    Usage:
        @time_function("my_function")
        def my_function():
            # Your code here
            pass

        # Or auto-detect name
        @time_function()
        def my_function():
            pass
    """
    def decorator(func):
        name = component_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_component(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def time_async_function(component_name: Optional[str] = None):
    """Decorator for timing async function execution.

    Usage:
        @time_async_function("my_async_function")
        async def my_async_function():
            # Your code here
            pass
    """
    def decorator(func):
        name = component_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                _timing_logger.log_timing(name, elapsed)

        return wrapper

    return decorator


__all__ = [
    "TimingLogger",
    "get_timing_logger",
    "start_query_timing",
    "end_query_timing",
    "get_timing_report",
    "print_timing_report",
    "time_component",
    "time_function",
    "time_async_function",
]
