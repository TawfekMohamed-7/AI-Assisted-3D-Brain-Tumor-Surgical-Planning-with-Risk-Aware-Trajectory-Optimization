import heapq
import numpy as np
from scipy.ndimage import binary_erosion

MOVES = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _get_surgical_target(et_mask, whole_tumor):
    if et_mask.sum() > 0:
        coords = np.array(np.where(et_mask)).T
    else:
        coords = np.array(np.where(whole_tumor)).T
    return tuple(coords.mean(axis=0).astype(int))


def _get_brain_surface(t1ce_data):
    brain_mask     = t1ce_data > t1ce_data.mean() * 0.15
    brain_interior = binary_erosion(brain_mask, iterations=3)
    return np.array(np.where(brain_mask & ~brain_interior)).T


def _heuristic(a, b):
    return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float))


def _astar(start, goal, risk_map, max_steps=5000):
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score   = {start: 0.0}
    shape     = risk_map.shape
    steps     = 0

    while open_set and steps < max_steps:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        cx, cy, cz = current
        for dx, dy, dz in MOVES:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue
            nb = (nx, ny, nz)
            g  = g_score[current] + 1.0 + risk_map[nx, ny, nz]
            if nb not in g_score or g < g_score[nb]:
                g_score[nb] = g
                heapq.heappush(open_set, (g + _heuristic(nb, goal), nb))
                came_from[nb] = current

        steps += 1

    return None


def _score_path(path, risk_map, functional_masks):
    motor    = functional_masks.get('motor')
    language = functional_masks.get('language')
    visual   = functional_masks.get('visual')

    risk_score = float(np.mean([risk_map[v] for v in path]))

    eloquent = sum(
        1 for v in path
        if (motor    is not None and motor[v])    or
           (language is not None and language[v]) or
           (visual   is not None and visual[v])
    ) / len(path)

    arr        = np.array(path, dtype=float)
    smoothness = float(np.std(np.linalg.norm(np.diff(arr, axis=0), axis=1))) if len(path) > 1 else 0.0

    return 0.2 * len(path) + 0.5 * risk_score + 0.3 * eloquent + 0.1 * smoothness


def plan_paths(risk_map, t1ce_data, et_mask, whole_tumor,
               functional_masks, num_starts=150, progress_cb=None):
    """
    Run A* from `num_starts` random brain-surface candidates to the tumor target.
    Returns (scored_paths, target_vox) where scored_paths is sorted best-first.
    """
    if progress_cb:
        progress_cb("Finding surgical target...")

    target = _get_surgical_target(et_mask, whole_tumor)

    if progress_cb:
        progress_cb("Computing brain surface candidates...")

    surface = _get_brain_surface(t1ce_data)

    if len(surface) > num_starts:
        idx    = np.random.choice(len(surface), num_starts, replace=False)
        starts = [tuple(surface[i]) for i in idx]
    else:
        starts = [tuple(v) for v in surface]

    if progress_cb:
        progress_cb(f"Running A* path planning ({len(starts)} entry candidates)...")

    all_paths = []
    for start in starts:
        path = _astar(start, target, risk_map)
        if path is not None:
            all_paths.append(path)

    if not all_paths:
        return [], target

    if progress_cb:
        progress_cb("Scoring and ranking surgical paths...")

    scored = [((_score_path(p, risk_map, functional_masks)), p) for p in all_paths]
    scored.sort(key=lambda x: x[0])
    return scored, target
