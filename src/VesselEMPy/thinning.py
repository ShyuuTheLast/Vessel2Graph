"""Topological-thinning skeletons. Coordinates and radii are in nanometers, ZYX.

Thinning peels the mask to a one-voxel centreline without changing its topology,
so vessel loops survive and no caliber-dependent tracing parameter exists. It has
no notion of significance, so two radius-relative clean-ups follow: loops too small
to encircle a gap in their own vessel are opened, and dead-end stubs shorter than
the local cross-section are pruned.
"""
import time
import cc3d
import edt
import fill_voids
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize
from .graph import branch_paths
from .volume_estimation import close_physical


def voxel_graph(zyx, spacing):
    """26-adjacency over skeleton voxels, minus diagonals that only shortcut a corner."""
    n = len(zyx)
    index = -np.ones(zyx.max(0) + 3, np.int64)
    index[tuple((zyx + 1).T)] = np.arange(n)
    graph = nx.Graph()
    graph.add_nodes_from((i, dict(nm=zyx[i] * spacing)) for i in range(n))
    for offset in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]:
        other = index[zyx[:, 0] + 1 + offset[0], zyx[:, 1] + 1 + offset[1], zyx[:, 2] + 1 + offset[2]]
        found = other >= 0
        # order: 1 face, 2 edge, 3 corner neighbor.
        graph.add_edges_from((int(a), int(b), dict(order=int(np.count_nonzero(offset)),
            length=float(np.linalg.norm(np.array(offset) * spacing)))) for a, b in zip(np.flatnonzero(found), other[found]))
    for order in (3, 2):
        for a, b in [(a, b) for a, b, d in graph.edges(data=True) if d['order'] == order]:
            if any(graph[a][c]['order'] < order and graph[b][c]['order'] < order for c in set(graph[a]) & set(graph[b])):
                graph.remove_edge(a, b)
    return graph


def anchored_thinning(mask, on_volume_face):
    """Thin the mask with every vessel that leaves the volume pinned at its exit.

    Lee thinning deletes from free ends until a one-voxel tip forms, so an even-width
    vessel recedes from a cut face and a straight one can vanish. A one-voxel whisker
    grown outward from the centre of each cross-section on a volume face is such a tip:
    it survives thinning and holds the centreline at the face. Whiskers lie in padding
    that is cropped off afterwards.
    """
    pad = 2
    padded, centres = np.pad(mask, pad), []
    for axis in range(3):
        for side in (0, 1):
            if not on_volume_face[axis][side]:
                continue
            face = np.ascontiguousarray(mask.take(-1 if side else 0, axis=axis))
            labels, count = cc3d.connected_components(face, connectivity=8, return_N=True)
            if not count:
                continue
            depth = edt.edt(face, black_border=True)
            for label in range(1, count + 1):
                centre = np.unravel_index(np.argmax(np.where(labels == label, depth, -1)), face.shape)
                where = [slice(None)] * 3
                where[axis] = slice(-pad, None) if side else slice(0, pad)
                for other, value in zip([a for a in range(3) if a != axis], centre):
                    where[other] = value + pad
                padded[tuple(where)] = 1
                where[axis] = -pad - 1 if side else pad
                centres.append(tuple(where))
    line = skeletonize(padded) > 0
    # Face voxels left hanging on a whisker are adjacent to its base, so keeping the
    # base in the skeleton preserves their connection once the padding is cropped.
    for centre in centres:
        line[centre] = True
    return line[pad:-pad, pad:-pad, pad:-pad]


def path_length(graph, path):
    return sum(graph[a][b]['length'] for a, b in zip(path[:-1], path[1:]))


def open_small_loops(graph, radius, factor, min_perimeter):
    """Open every loop that is too small to be a vessel loop.

    Two limits apply and the larger one decides. Geometric: a loop of vessel around a
    real gap lies at least one radius from it, so loops below factor * 2*pi * the largest
    node radius are pores or lattice artefacts. Anatomical: min_perimeter, in nm, is the
    smallest loop the capillary mesh can form; narrow slits between touching strands
    exceed the geometric limit but not this one. The loop is opened by deleting its
    narrowest stretch between two junctions, which keeps every junction on it connected
    through the wider side.
    """
    opened, limit = 0, max(factor * 2 * np.pi * float(radius.max()), min_perimeter)
    while True:
        # Every loop contains an edge outside a spanning tree, so only those need a search.
        tree = {frozenset(e) for e in nx.minimum_spanning_edges(graph, weight='length', data=False)}
        found = False
        for a, b in [e for e in graph.edges if frozenset(e) not in tree]:
            if not graph.has_edge(a, b):
                continue
            data = dict(graph[a][b])
            graph.remove_edge(a, b)
            try:
                detour, cycle = nx.single_source_dijkstra(graph, a, b, cutoff=limit, weight='length')
            except nx.NetworkXNoPath:
                cycle = None
            graph.add_edge(a, b, **data)
            if cycle is None or detour + data['length'] >= max(factor * 2 * np.pi * radius[cycle].max(), min_perimeter):
                continue
            junctions = [i for i, node in enumerate(cycle) if graph.degree(node) > 2]
            if not junctions:
                continue  # A free-standing ring has no stretch that can be judged against another.
            arcs = []
            for start, stop in zip(junctions, junctions[1:] + [junctions[0] + len(cycle)]):
                arc = [cycle[i % len(cycle)] for i in range(start, stop + 1)]
                inner = arc[1:-1]
                arcs.append((float(radius[inner].mean()) if inner else float(radius[arc].min()),
                             -path_length(graph, arc), arc, inner))
            _, _, arc, inner = min(arcs, key=lambda item: item[:2])
            if inner:
                graph.remove_nodes_from(inner)
            else:
                graph.remove_edge(arc[0], arc[1])
            opened += 1
            found = True
        if not found:
            return opened


def cross_section_half_widths(mask, spacing, centre, axis, cap, spokes=32):
    """Short and long half-width of the lumen in the plane perpendicular to axis.

    Each of `spokes` diameters is the sum of two opposite exit distances, so a
    centreline that sits off-centre does not bias the widths.
    """
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    angles = np.arange(spokes) * np.pi / spokes
    half = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * w
    directions = np.concatenate([half, -half])
    steps = np.arange(1, int(cap / (0.5 * spacing.min())) + 2) * 0.5 * spacing.min()
    index = np.rint((centre + steps[None, :, None] * directions[:, None, :]) / spacing).astype(int)
    inside = ((index >= 0) & (index < mask.shape)).all(-1)
    index = np.clip(index, 0, np.array(mask.shape) - 1)
    inside &= mask[index[..., 0], index[..., 1], index[..., 2]] != 0
    # A ray that never leaves is held at the cap.
    leave = np.where(inside.all(1), steps[-1], steps[np.argmin(inside, axis=1)])
    widths = leave[:spokes] + leave[spokes:]
    return widths.min() / 2, widths.max() / 2


def junction_yardstick(graph, mask, spacing, radius, junction, stub, cache):
    """Long half-width of the vessel a stub hangs off, measured beside the junction.

    The inscribed radius is the short half-width, but a stub grows along the long one.
    Each other branch is followed for 1.5 junction radii; the next 1.5 radii of centreline
    give the axis. The smallest result is used, never less than the inscribed radius.
    """
    others = tuple(sorted(n for n in graph[junction] if n != stub[-1]))
    if (junction, others) in cache:
        return cache[junction, others]
    r, longs = radius[junction], []
    for first in others:
        previous, node, arc, window = junction, first, graph[junction][first]['length'], []
        while True:
            if arc >= 1.5 * r:
                window.append(node)
            onward = [n for n in graph[node] if n != previous]
            if arc >= 3.0 * r or graph.degree(node) != 2 or not onward:
                break
            arc += graph[node][onward[0]]['length']
            previous, node = node, onward[0]
        if len(window) < 3:
            continue
        points = np.array([graph.nodes[n]['nm'] for n in window], float)
        centre = window[len(window) // 2]
        longs.append(cross_section_half_widths(mask, spacing, graph.nodes[centre]['nm'],
            np.linalg.svd(points - points.mean(0))[2][0], 6 * radius[centre] + float(np.linalg.norm(spacing)))[1])
    cache[junction, others] = max(r, min(longs)) if longs else r
    return cache[junction, others]


def prune_stubs(graph, radius, factor, mask, spacing, protected):
    """Repeatedly delete dead-end branches shorter than factor * junction yardstick.

    Protected tips, vessels leaving the volume, are real ends and are never removed.
    One voxel diagonal is allowed for lattice rounding.
    """
    removed, slack = 0, float(np.linalg.norm(spacing))
    while True:
        # The yardstick reads centreline beyond the junction, which pruning changes: no reuse across rounds.
        cache, doomed = {}, []
        for path in branch_paths(graph)[0]:
            for stub in (path, path[::-1]):
                if graph.degree(stub[0]) == 1 and graph.degree(stub[-1]) > 2 and stub[0] not in protected:
                    size = junction_yardstick(graph, mask, spacing, radius, stub[-1], stub[:-1], cache)
                    if path_length(graph, stub) < factor * size + slack:
                        doomed.append((stub[:-1], stub[-1]))
        if not doomed:
            return removed
        # Shortest first, and a junction keeps at least two branches within one round.
        for stub, junction in sorted(doomed, key=lambda item: len(item[0])):
            if graph.degree(junction) > 2:
                graph.remove_nodes_from(stub)
                removed += 1


def thin_skeleton(mask, spacing, options, offset=(0, 0, 0), full_shape=None, parallel=1):
    """Skeleton arrays in the layout of io.skeleton_arrays, plus a statistics report.

    mask may be a crop: offset places it in the full volume, whose faces decide which
    tips are protected from pruning.
    """
    spacing, offset = np.asarray(spacing, float), np.asarray(offset)
    full_shape = np.asarray(mask.shape if full_shape is None else full_shape)
    clock, report = time.perf_counter(), dict(original_foreground=int(np.count_nonzero(mask)))
    if options['closing_radius_nm']:
        mask, report['closing'] = close_physical(mask, spacing, options['closing_radius_nm'])
    if options['fill_holes']:
        mask, report['filled_voxels'] = fill_voids.fill(mask, in_place=False, return_fill_count=True)
        report['filled_voxels'] = int(report['filled_voxels'])
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    report['prepare_seconds'] = time.perf_counter() - clock

    clock = time.perf_counter()
    line = anchored_thinning(mask, [(offset[a] == 0, offset[a] + mask.shape[a] == full_shape[a]) for a in range(3)])
    zyx = np.argwhere(line)
    if not len(zyx):
        raise ValueError('Thinning produced no skeleton voxels')
    # Thinning can still erase a small symmetric object that touches no volume face.
    labels, count = cc3d.connected_components(mask, connectivity=26, return_N=True)
    report.update(mask_components=int(count), components_without_skeleton=int(count - len(np.unique(labels[line]))))
    del labels
    report['thinning_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    radius = edt.edt(mask, anisotropy=tuple(spacing), black_border=False, parallel=parallel)[line].astype(np.float64)
    report['distance_seconds'] = time.perf_counter() - clock

    clock = time.perf_counter()
    graph = voxel_graph(zyx, spacing)
    def loops():
        return graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
    report.update(skeleton_voxels=len(zyx), loops_before=loops())
    report['loops_opened'] = (open_small_loops(graph, radius, options['loop_factor'], options['min_loop_perimeter_nm'])
                              if options['loop_factor'] or options['min_loop_perimeter_nm'] else 0)
    # A tip within two voxels of a face of the full volume is a vessel leaving it. Protecting such
    # tips is off by default: a vessel cut at an angle thins into a fan of tips on the face, all of
    # which protection keeps, while without it the vessel still ends within about a radius of the face.
    position = zyx + offset
    tips = {n for n in graph if graph.degree(n) == 1 and ((position[n] < 2) | (position[n] > full_shape - 3)).any()}
    protected = tips if options.get('protect_face_tips', False) else set()
    report.update(endpoints_before=sum(1 for n in graph if graph.degree(n) == 1), face_tips=len(tips), protected_tips=len(protected))
    report['stubs_pruned'] = prune_stubs(graph, radius, options['prune_factor'], mask, spacing, protected) if options['prune_factor'] else 0
    report.update(loops_after=loops(), endpoints_after=sum(1 for n in graph if graph.degree(n) == 1),
                  graph_seconds=time.perf_counter() - clock)

    keep = np.array(sorted(graph.nodes))
    renumber = -np.ones(len(zyx), np.int64)
    renumber[keep] = np.arange(len(keep))
    edges = np.array([(renumber[a], renumber[b]) for a, b in graph.edges], np.int64).reshape(-1, 2)
    return dict(vertices=position[keep] * spacing, edges=edges, radius=radius[keep],
                object_id=np.ones(len(keep), np.int64)), report
