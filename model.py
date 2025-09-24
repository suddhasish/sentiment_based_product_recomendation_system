# model.py (Option C)
import os, joblib, numpy as np, pandas as pd
from difflib import get_close_matches

ART_PATH = os.path.join("models", "user_compact_map.pkl")  # update path if needed

# Load artifact
_art = joblib.load(ART_PATH)
user_neighbors_idx = np.array(_art.get("user_neighbors_idx"))
user_neighbors_sim = np.array(_art.get("user_neighbors_sim"))
user_labels = list(_art.get("user_labels"))
item_labels = list(_art.get("item_labels"))
user_item_map = _art.get("user_item_map")   # dict: user -> (idxs ndarray, values ndarray)
norm_map = _art.get("norm_map", {str(u).strip().lower(): u for u in user_labels})
k = _art.get("k", user_neighbors_idx.shape[1] if user_neighbors_idx is not None else 0)

# Build quick popularity fallback from user_item_map
from collections import Counter
_counter = Counter()
for u, (idxs, vals) in user_item_map.items():
    for i in idxs.tolist():
        _counter[i] += 1
_popular_item_idxs = [i for i,_ in _counter.most_common(100)]
_popular_items = [item_labels[i] for i in _popular_item_idxs]

def _norm(u):
    return str(u).strip().lower() if u is not None else None

def _resolve_username(raw):
    if raw is None: return None
    if raw in user_labels: return raw
    n = _norm(raw)
    if n in norm_map: return norm_map[n]
    close = get_close_matches(raw, user_labels, n=1, cutoff=0.6)
    if close:
        return close[0]
    close2 = get_close_matches(n, list(norm_map.keys()), n=1, cutoff=0.8)
    if close2:
        return norm_map[close2[0]]
    return None

def recommend_for_user(raw_username, top_n=5, debug=False):
    user = _resolve_username(raw_username)
    if debug: print("Resolved:", raw_username, "->", user)
    if user is None:
        # return popular fallback
        return [{"item": it, "score": None} for it in _popular_items[:top_n]]

    # get neighbors
    try:
        uidx = user_labels.index(user)
    except ValueError:
        return [{"item": it, "score": None} for it in _popular_items[:top_n]]

    if user_neighbors_idx is None:
        return [{"item": it, "score": None} for it in _popular_items[:top_n]]

    neigh_ids = user_neighbors_idx[uidx].astype(int)
    neigh_sims = user_neighbors_sim[uidx].astype(float)
    valid = neigh_ids >= 0
    neigh_ids = neigh_ids[valid]
    neigh_sims = neigh_sims[valid]
    if len(neigh_ids) == 0:
        return [{"item": it, "score": None} for it in _popular_items[:top_n]]

    # Accumulate weighted ratings across neighbor users using their compressed rating lists
    # We'll compute numerator and denominator per item index using dictionary for speed
    numer = {}
    denom = {}
    for nid, sim in zip(neigh_ids.tolist(), neigh_sims.tolist()):
        neighbor_user = user_labels[nid]
        idxs, vals = user_item_map.get(neighbor_user, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
        # for each rated item by neighbor
        for ii, val in zip(idxs.tolist(), vals.tolist()):
            numer[ii] = numer.get(ii, 0.0) + sim * float(val)
            denom[ii] = denom.get(ii, 0.0) + abs(float(sim))
    # convert to scores
    scores = []
    for ii, num in numer.items():
        den = denom.get(ii, 1e-8)
        scores.append((ii, num / den))
    if not scores:
        return [{"item": it, "score": None} for it in _popular_items[:top_n]]

    # Build Series or sort
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    # mask items that user already rated
    user_idxs, user_vals = user_item_map.get(user, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
    seen_set = set(user_idxs.tolist())
    recs = []
    for ii, s in scores_sorted:
        if ii in seen_set: 
            continue
        recs.append((item_labels[ii], float(s)))
        if len(recs) >= top_n:
            break

    # fallback to popular if not enough
    if len(recs) < top_n:
        for ii in _popular_item_idxs:
            if item_labels[ii] in [r[0] for r in recs]: continue
            if ii in seen_set: continue
            recs.append((item_labels[ii], None))
            if len(recs) >= top_n: break

    return [{"item": it, "score": sc} for it, sc in recs]
