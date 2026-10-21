"""Edge-complete topological branches. Coordinates and radii are in nanometers."""
from itertools import combinations
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans


def branch_paths(graph):
    seen, paths, loops = set(), [], []
    def edge(a, b):
        return min(a, b), max(a, b)
    def trace(a, b):
        path = [a, b]
        seen.add(edge(a, b))
        while graph.degree(b) == 2:
            nxt = next(n for n in graph[b] if n != a)
            if edge(b, nxt) in seen:
                break
            seen.add(edge(b, nxt))
            path.append(nxt)
            a, b = b, nxt
        return path
    for a in sorted(graph):
        if graph.degree(a) not in (0, 2):
            for b in sorted(graph[a]):
                if edge(a, b) not in seen:
                    paths.append(trace(a, b))
    for a, b in sorted(graph.edges):
        if edge(a, b) not in seen:
            loops.append(trace(a, b))
    return paths, loops


def sandwich_labels(graph, paths, labels):
    """Snapshot pass: a branch whose two distinct ends each touch a larger class takes the
    smaller of those two classes (legacy relabel of small branches between large ones)."""
    incident = {}
    for i, path in enumerate(paths):
        for node in {path[0], path[-1]}:
            if graph.degree(node) > 2:
                incident.setdefault(node, []).append(i)
    result, changes = labels.copy(), []
    for i, path in enumerate(paths):
        a, b = path[0], path[-1]
        if a == b or a not in incident or b not in incident:
            continue
        top = [max((labels[j] for j in incident[n] if j != i), default=0) for n in (a, b)]
        if min(top) > labels[i]:
            result[i] = min(top)
            changes.append(dict(branch=i + 1, original_class=int(labels[i]), segmentation_class=int(result[i])))
    return result, changes


def caliber_seed_labels(graph, paths, labels, data, original, enabled):
    """Snapshot-based projection labels; never edit branch membership or measurements."""
    result = original.copy()
    ends = {}
    for i, path in enumerate(paths):
        for ordered in (path, path[::-1]):
            if len(ordered) > 1 and graph.degree(ordered[0]) > 2:
                ends.setdefault(ordered[0], []).append((i, ordered))
    proposals = {}
    for junction, incident in sorted(ends.items()):
        # A junction belongs to the largest class among its branches (legacy label_branch_points).
        larger_class = max(int(labels[i]) for i, _ in incident)
        result[junction] = larger_class
        if not enabled:
            continue
        for i, ordered in incident:
            if labels[i] >= larger_class:
                continue
            for node in ordered[1:]:
                if graph.degree(node) > 2:
                    break
                if np.linalg.norm(data['vertices'][node] - data['vertices'][junction]) > data['radius'][junction]:
                    break
                proposals.setdefault(node, []).append((larger_class, junction))
    changes = []
    for node, candidates in sorted(proposals.items()):
        target = max(label for label, _ in candidates)
        result[node] = target
        changes.append(dict(node=node, original_class=int(original[node]), segmentation_class=target,
                            junctions=sorted({j for label, j in candidates if label == target})))
    return result, changes


def ellipsoid_seed_nodes(graph, paths, labels, data, reach):
    """Nodes seeded with an ellipsoid surface instead of a point: each junction where a smaller
    class meets a larger one, and the larger-class nodes within reach x its radius, so the
    side branch cannot claim voxels inside the larger vessel's wall."""
    ends = {}
    for i, path in enumerate(paths):
        for ordered in (path, path[::-1]):
            if len(ordered) > 1 and graph.degree(ordered[0]) > 2:
                ends.setdefault(ordered[0], []).append((i, ordered))
    flag = np.zeros(len(data['vertices']), dtype=bool)
    for junction, incident in ends.items():
        classes = [int(labels[i]) for i, _ in incident]
        larger_class = max(classes)
        if min(classes) == larger_class:
            continue
        flag[junction] = True
        for i, ordered in incident:
            if labels[i] != larger_class:
                continue
            for node in ordered[1:]:
                if np.linalg.norm(data['vertices'][node] - data['vertices'][junction]) > reach * data['radius'][junction]:
                    break
                flag[node] = True
    return flag


def build_graph(data, settings):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data['vertices'])))
    graph.add_edges_from(data['edges'].tolist())
    if nx.number_of_selfloops(graph):
        raise ValueError('Self-loop skeleton edges are unsupported')
    paths, loops = branch_paths(graph)
    if not paths:
        arrays = dict(data, branch=np.zeros(len(graph), dtype=np.uint32),
                      label=np.zeros(len(graph), dtype=np.uint32),
                      segmentation_label=np.zeros(len(graph), dtype=np.uint32),
                      ellipsoid_seed=np.zeros(len(graph), dtype=bool),
                      dist_from_largest=np.zeros(len(graph), dtype=np.uint32))
        return arrays, [], dict(vertices=len(graph), edges=graph.number_of_edges(),
            components=nx.number_connected_components(graph), branches=0,
            isolated_loops=loops, isolated_vertices=list(nx.isolates(graph)), total_length_nm=0)
    medians = np.array([np.median(data['radius'][np.unique(p)]) for p in paths])
    k = min(settings['radius_clusters'], len(np.unique(medians)))
    fit = KMeans(n_clusters=k, random_state=settings['random_seed'], n_init=10).fit(medians[:, None])
    order = np.argsort(fit.cluster_centers_[:, 0])
    ranks = np.empty(k, dtype=np.uint32)
    ranks[order] = np.arange(1, k + 1)
    labels = ranks[fit.labels_]
    incidence = [[] for _ in graph]
    for i, path in enumerate(paths):
        for node in set(path):
            incidence[node].append(i)
    connectivity = nx.Graph()
    connectivity.add_nodes_from(range(len(paths)))
    for members in incidence:
        connectivity.add_edges_from(combinations(members, 2))
    # One-based graph hops: 1 is a largest-caliber branch; 0 means unreachable.
    distances = dict(nx.multi_source_dijkstra_path_length(connectivity,
                     [i for i, label in enumerate(labels) if label == k], weight=None))
    hops = np.array([distances[i] + 1 if i in distances else 0 for i in range(len(paths))], dtype=np.uint32)
    owners = np.array([max(members, key=lambda i: (medians[i], -i)) if members else -1 for members in incidence])
    arrays = dict(data, edges=np.array(sorted(graph.edges), dtype=np.int64).reshape(-1, 2),
                  branch=(owners + 1).astype(np.uint32), label=np.where(owners >= 0, labels[owners], 0),
                  dist_from_largest=np.where(owners >= 0, hops[owners], 0))
    junctions = [n for n in graph if graph.degree(n) > 2]
    # Zero denotes absent branch ownership, not a background seed.
    arrays['branch'][junctions] = 0
    arrays['label'][junctions] = 0
    arrays['dist_from_largest'][junctions] = 0
    # Segmentation classes: sandwich first, then branch-end and junction labels on the result.
    if settings.get('sandwich', True):
        segmentation_classes, sandwiched = sandwich_labels(graph, paths, labels)
    else:
        segmentation_classes, sandwiched = labels.copy(), []
    projected = np.where(owners >= 0, segmentation_classes[owners], 0).astype(np.uint32)
    projected[junctions] = 0
    arrays['segmentation_label'], corrections = caliber_seed_labels(
        graph, paths, segmentation_classes, data, projected, settings.get('correct_branch_ends', True))
    arrays['ellipsoid_seed'] = ellipsoid_seed_nodes(graph, paths, segmentation_classes, data,
                                                    settings.get('ellipsoid_reach', 1.5))
    branches = []
    for i, path in enumerate(paths):
        length = np.linalg.norm(np.diff(data['vertices'][path], axis=0), axis=1).sum()
        branches.append(dict(id=i + 1, nodes=path, length_nm=float(length),
                             median_radius_nm=float(medians[i]), caliber_class=int(labels[i]),
                             segmentation_class=int(segmentation_classes[i]),
                             hops_from_largest=int(hops[i])))
    angles = []
    for node in graph:
        if graph.degree(node) > 2:
            for a, b in combinations(sorted(graph[node]), 2):
                u, v = data['vertices'][[a, b]] - data['vertices'][node]
                norm = np.linalg.norm(u) * np.linalg.norm(v)
                if norm:
                    angles.append(dict(node=node, neighbors=[a, b],
                                       degrees=float(np.degrees(np.arccos(np.clip(u @ v / norm, -1, 1))))))
    metrics = dict(vertices=len(graph), edges=graph.number_of_edges(),
                   components=nx.number_connected_components(graph),
                   endpoints=sum(d == 1 for _, d in graph.degree),
                   junctions=sum(d > 2 for _, d in graph.degree), branches=len(paths),
                   total_length_nm=sum(b['length_nm'] for b in branches),
                   radius_clusters=k, requested_radius_clusters=settings['radius_clusters'],
                   caliber_centers_nm=fit.cluster_centers_[order, 0].tolist(),
                   connectivity_edges=[[a + 1, b + 1] for a, b in sorted(connectivity.edges)],
                   branching_angles=angles, isolated_loops=loops, isolated_vertices=list(nx.isolates(graph)),
                   sandwiched_branches=sandwiched, branch_end_corrections=corrections,
                   corrected_node_count=len(corrections), ellipsoid_seed_nodes=int(arrays['ellipsoid_seed'].sum()),
                   junction_caliber_labels=[dict(node=n, segmentation_class=int(arrays['segmentation_label'][n])) for n in junctions])
    return arrays, branches, metrics
