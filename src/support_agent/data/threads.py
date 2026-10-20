"""Deterministic reconstruction of conversation graph components."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from support_agent.hashing import canonical_sha256
from support_agent.schemas import RawMessage, ReconstructedThread

RECONSTRUCTION_VERSION = "1"


class ThreadGraphAudit(BaseModel):
    """Audit information for one reconstructed component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    message_count: int
    has_cycle: bool
    dangling_reference_count: int
    is_multi_brand: bool


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}
        self._rank = {value: 0 for value in self._parent}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


def _ordered_unique_edges(messages: dict[str, RawMessage]) -> tuple[set[tuple[str, str]], int]:
    edges: set[tuple[str, str]] = set()
    dangling = 0
    for message in messages.values():
        if message.in_response_to_tweet_id is not None:
            parent = message.in_response_to_tweet_id
            if parent in messages:
                edges.add((parent, message.tweet_id))
            else:
                dangling += 1
        for child in message.response_tweet_ids:
            if child in messages:
                edges.add((message.tweet_id, child))
            else:
                dangling += 1
    return edges, dangling


def _component_has_cycle(node_ids: Sequence[str], edges: set[tuple[str, str]]) -> bool:
    indegree = {node_id: 0 for node_id in node_ids}
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
        indegree[child] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        parent = queue.popleft()
        visited += 1
        for child in sorted(children[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited != len(node_ids)


def reconstruct_threads(messages: Sequence[RawMessage]) -> list[ReconstructedThread]:
    """Return eligible single-brand connected components in deterministic order.

    Both parent and child references participate in connectivity. Components with
    multiple outbound authors are intentionally excluded because they cannot be
    assigned to one selected brand without an arbitrary choice.
    """
    by_id: dict[str, RawMessage] = {}
    for message in messages:
        if message.tweet_id in by_id:
            raise ValueError(f"Duplicate tweet_id encountered: {message.tweet_id}")
        by_id[message.tweet_id] = message
    union_find = _UnionFind(by_id)
    edges, dangling_by_source = _ordered_unique_edges(by_id)
    for parent, child in edges:
        union_find.union(parent, child)

    components: dict[str, list[str]] = defaultdict(list)
    for tweet_id in by_id:
        components[union_find.find(tweet_id)].append(tweet_id)
    edges_by_component: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for parent, child in edges:
        edges_by_component[union_find.find(parent)].add((parent, child))

    result: list[ReconstructedThread] = []
    for component_ids in components.values():
        component_set = set(component_ids)
        component_messages = [by_id[tweet_id] for tweet_id in component_ids]
        outbound_authors = {
            message.author_id for message in component_messages if not message.inbound
        }
        if len(outbound_authors) != 1:
            continue
        inbound_count = sum(message.inbound for message in component_messages)
        outbound_count = len(component_messages) - inbound_count
        if inbound_count == 0 or outbound_count == 0:
            continue
        component_edges = edges_by_component[union_find.find(component_ids[0])]
        incoming = {child for _, child in component_edges}
        roots = tuple(sorted(component_set.difference(incoming)))
        ordered_messages = tuple(
            sorted(component_messages, key=lambda item: (item.created_at, item.tweet_id))
        )
        thread_id = (
            roots[0]
            if len(roots) == 1
            else canonical_sha256(list(roots) if roots else sorted(component_set))[:20]
        )
        component_dangling = sum(
            1
            for message in component_messages
            for referenced in (*message.response_tweet_ids, message.in_response_to_tweet_id)
            if referenced is not None and referenced not in by_id
        )
        result.append(
            ReconstructedThread(
                thread_id=thread_id,
                brand_id=next(iter(outbound_authors)),
                root_tweet_ids=roots,
                messages=ordered_messages,
                started_at=ordered_messages[0].created_at,
                ended_at=ordered_messages[-1].created_at,
                customer_turn_count=inbound_count,
                brand_turn_count=outbound_count,
                is_multi_turn=inbound_count >= 2 and outbound_count >= 2,
                has_cycle=_component_has_cycle(tuple(component_set), component_edges),
                dangling_reference_count=component_dangling,
                reconstruction_version=RECONSTRUCTION_VERSION,
            )
        )
    if dangling_by_source < 0:
        raise AssertionError("dangling reference count cannot be negative")
    return sorted(result, key=lambda item: (item.started_at, item.thread_id))


def validate_thread_graph(thread: ReconstructedThread) -> ThreadGraphAudit:
    """Validate immutable thread-level graph facts represented in the schema."""
    message_ids = tuple(message.tweet_id for message in thread.messages)
    if len(message_ids) != len(set(message_ids)):
        raise ValueError(f"Thread {thread.thread_id} has duplicate message IDs")
    if thread.messages[0].created_at != thread.started_at:
        raise ValueError(f"Thread {thread.thread_id} has inconsistent started_at")
    if thread.messages[-1].created_at != thread.ended_at:
        raise ValueError(f"Thread {thread.thread_id} has inconsistent ended_at")
    return ThreadGraphAudit(
        thread_id=thread.thread_id,
        message_count=len(thread.messages),
        has_cycle=thread.has_cycle,
        dangling_reference_count=thread.dangling_reference_count,
        is_multi_brand=False,
    )
