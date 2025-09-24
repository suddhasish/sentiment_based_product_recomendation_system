# check_users.py
import os
import joblib
import pickle
from difflib import get_close_matches
import numpy as np
import pandas as pd

# Try compact path first, then legacy names
CANDIDATES = [
    "models/user_user_compact.pkl",
]

def _norm(u):
    return str(u).strip().lower() if u is not None else None

def robust_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # try joblib then pickle
    try:
        obj = joblib.load(path)
        print(f"Loaded artifact via joblib: {path} (type={type(obj)})")
        return obj
    except Exception as e:
        # print("joblib.load failed:", e)
        pass
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"Loaded artifact via pickle: {path} (type={type(obj)})")
        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to load artifact {path}: {e}")

def find_artifact():
    for p in CANDIDATES:
        if os.path.exists(p):
            try:
                return p, robust_load(p)
            except Exception as e:
                print("Found file but failed to load:", p, "->", e)
    raise FileNotFoundError("No artifact found at expected paths. Check models/ folder.")

# --- helpers to interpret artifact ---
def normalize_artifact(loaded):
    """
    Convert a loaded artifact into a normalized dict with possible keys:
      - dummy_train (DataFrame users x items)
      - user_neighbors_idx (ndarray n_users x K)
      - user_neighbors_sim (ndarray n_users x K)
      - user_sim (ndarray n_users x n_users)  # optional
      - item_sim (ndarray n_items x n_items)  # optional
      - user_labels, item_labels, norm_map, k
    """
    art = {}
    # If the loaded object *is* a DataFrame assume it's dummy_train
    if isinstance(loaded, pd.DataFrame):
        art["dummy_train"] = loaded.copy()
    elif isinstance(loaded, dict):
        # copy relevant keys (defensive)
        art.update(loaded)
    else:
        # unknown wrapper: try to introspect attributes
        # if it has .dummy_train attribute
        if hasattr(loaded, "dummy_train"):
            art.update({"dummy_train": getattr(loaded, "dummy_train")})
        else:
            raise RuntimeError("Loaded artifact must be dict or DataFrame or wrapper with dummy_train")

    # normalize types
    if "dummy_train" in art:
        if not isinstance(art["dummy_train"], pd.DataFrame):
            # maybe it's stored as something else
            try:
                art["dummy_train"] = pd.DataFrame(art["dummy_train"])
            except Exception:
                raise RuntimeError("dummy_train present but cannot coerce to DataFrame")

    # ensure user_labels / item_labels
    if "user_labels" not in art and "dummy_train" in art:
        art["user_labels"] = list(art["dummy_train"].index)
    if "item_labels" not in art and "dummy_train" in art:
        art["item_labels"] = list(art["dummy_train"].columns)

    # build norm_map if missing
    if "norm_map" not in art and "user_labels" in art:
        art["norm_map"] = { _norm(lbl): lbl for lbl in art["user_labels"] }

    return art

# --- capability checks ---
def has_compact_neighbors(art):
    return ("user_neighbors_idx" in art) and ("user_neighbors_sim" in art)

def has_user_sim(art):
    return ("user_sim" in art) and getattr(art["user_sim"], "shape", None) is not None

def has_item_sim(art):
    return ("item_sim" in art) and getattr(art["item_sim"], "shape", None) is not None

# --- recommendation engines ---
def recommend_user_based_compact(art, user_label, topk=5):
    """
    Use compact neighbor lists to compute weighted user-based recommendations.
    """
    if "dummy_train" not in art:
        raise RuntimeError("artifact lacks dummy_train")
    if not has_compact_neighbors(art):
        raise RuntimeError("artifact lacks user_neighbors_idx/sim")

    pivot = art["dummy_train"]
    users = art["user_labels"]
    items = art["item_labels"]
    norm_map = art.get("norm_map", { _norm(u): u for u in users })

    # resolve user label
    if user_label in pivot.index:
        resolved = user_label
    elif _norm(user_label) in norm_map:
        resolved = norm_map[_norm(user_label)]
    else:
        idxs = [str(x) for x in pivot.index]
        close = get_close_matches(user_label, idxs, n=1, cutoff=0.6)
        resolved = close[0] if close else None

    if resolved is None:
        raise KeyError(f"user '{user_label}' not found (compact).")

    uidx = users.index(resolved)
    neigh_idx = np.array(art["user_neighbors_idx"])[uidx]
    neigh_sim = np.array(art["user_neighbors_sim"])[uidx]

    valid = neigh_idx >= 0
    neigh_idx = neigh_idx[valid].astype(int)
    neigh_sim = neigh_sim[valid].astype(float)
    if len(neigh_idx) == 0:
        return []

    neighbor_labels = [users[i] for i in neigh_idx]
    # ensure neighbors exist in pivot
    neighbor_labels = [lbl for lbl in neighbor_labels if lbl in pivot.index]
    if len(neighbor_labels) == 0:
        return []

    neigh_ratings = pivot.loc[neighbor_labels].values.astype(np.float32)  # (n_neigh, n_items)
    mask = ~np.isnan(neigh_ratings)
    sims_arr = neigh_sim[:len(neighbor_labels)].astype(np.float32)
    weighted = sims_arr[:, None] * np.nan_to_num(neigh_ratings, nan=0.0)
    numer = weighted.sum(axis=0)
    denom = (np.abs(sims_arr)[:, None] * mask).sum(axis=0) + 1e-8
    scores = numer / denom
    s = pd.Series(scores, index=items)
    # mask seen
    try:
        seen = pivot.loc[resolved].dropna().index.tolist()
        s.loc[seen] = -np.inf
    except Exception:
        pass
    top = s.nlargest(topk)
    return list(zip(top.index.tolist(), top.values.tolist()))

def recommend_user_based_full_user_sim(art, user_label, topk=5):
    """
    Use full user_sim (dense) matrix if present.
    """
    if "dummy_train" not in art:
        raise RuntimeError("artifact lacks dummy_train")
    if not has_user_sim(art):
        raise RuntimeError("artifact lacks user_sim")

    pivot = art["dummy_train"]
    users = art["user_labels"]
    items = art["item_labels"]
    norm_map = art.get("norm_map", { _norm(u): u for u in users })

    if user_label in pivot.index:
        resolved = user_label
    elif _norm(user_label) in norm_map:
        resolved = norm_map[_norm(user_label)]
    else:
        idxs = [str(x) for x in pivot.index]
        close = get_close_matches(user_label, idxs, n=1, cutoff=0.6)
        resolved = close[0] if close else None

    if resolved is None:
        raise KeyError(f"user '{user_label}' not found (full user_sim).")

    uidx = users.index(resolved)
    user_sim = np.array(art["user_sim"], dtype=np.float32)
    sims = user_sim[uidx]
    sims[uidx] = 0.0
    # neighbors = where sims != 0
    neighbor_idxs = np.where(sims != 0)[0]
    if len(neighbor_idxs) == 0:
        return []

    neighbor_labels = [users[i] for i in neighbor_idxs]
    neigh_sims = sims[neighbor_idxs]

    neigh_ratings = pivot.loc[neighbor_labels].values.astype(np.float32)
    mask = ~np.isnan(neigh_ratings)
    weighted = neigh_sims[:, None] * np.nan_to_num(neigh_ratings, nan=0.0)
    numer = weighted.sum(axis=0)
    denom = (np.abs(neigh_sims)[:, None] * mask).sum(axis=0) + 1e-8
    scores = numer / denom
    s = pd.Series(scores, index=items)
    try:
        seen = pivot.loc[resolved].dropna().index.tolist()
        s.loc[seen] = -np.inf
    except Exception:
        pass
    top = s.nlargest(topk)
    return list(zip(top.index.tolist(), top.values.tolist()))

def recommend_item_based(art, user_label, topk=5):
    """
    Recommend using item_sim (item-item).
    """
    if "dummy_train" not in art:
        raise RuntimeError("artifact lacks dummy_train")
    if not has_item_sim(art):
        raise RuntimeError("artifact lacks item_sim")

    df = art["dummy_train"]
    norm_map = art.get("norm_map", { _norm(lbl): lbl for lbl in df.index })
    if user_label in df.index:
        label = user_label
    elif _norm(user_label) in norm_map:
        label = norm_map[_norm(user_label)]
    else:
        idxs = [str(x) for x in df.index]
        close = get_close_matches(user_label, idxs, n=1, cutoff=0.6)
        label = close[0] if close else None
    if label is None:
        raise KeyError(f"user '{user_label}' not found (item-based).")

    item_sim = np.asarray(art["item_sim"])
    item_labels = art.get("item_labels", list(df.columns))
    item_index = {it: idx for idx, it in enumerate(item_labels)}

    user_ratings = df.loc[label].dropna()
    rated_items = [it for it in user_ratings.index if it in item_index]
    if len(rated_items) == 0:
        raise ValueError(f"user '{label}' has 0 rated items present in item_labels")

    rated_idxs = [item_index[it] for it in rated_items]
    ratings_vec = user_ratings.loc[rated_items].values.astype(np.float32)

    numer = item_sim[:, rated_idxs].dot(ratings_vec)
    denom = np.abs(item_sim[:, rated_idxs]).sum(axis=1) + 1e-8
    scores = numer / denom

    scores_series = pd.Series(scores, index=item_labels)
    scores_series.loc[rated_items] = -np.inf
    top = scores_series.nlargest(topk)
    return list(zip(top.index.tolist(), top.values.tolist()))

# --- wrapper utilities ---
def can_recommend_user(art, user_label):
    """
    Check if user is present and whether we can generate user-based or item-based recs.
    Returns dict with booleans and reasons.
    """
    out = {"present": False, "user_based": False, "item_based": False, "reason": None}
    if "dummy_train" not in art:
        out["reason"] = "no dummy_train"
        return out
    df = art["dummy_train"]
    norm_map = art.get("norm_map", { _norm(lbl): lbl for lbl in df.index })

    # resolve
    resolved = None
    if user_label in df.index:
        resolved = user_label
    elif _norm(user_label) in norm_map:
        resolved = norm_map[_norm(user_label)]
    else:
        idxs = [str(x) for x in df.index]
        close = get_close_matches(user_label, idxs, n=1, cutoff=0.6)
        if close:
            resolved = close[0]

    if resolved is None:
        out["reason"] = "user not found"
        return out

    out["present"] = True
    rated_count = int(df.loc[resolved].notna().sum())
    out["rated_count"] = rated_count

    if has_compact_neighbors(art) or has_user_sim(art):
        out["user_based"] = True
    if has_item_sim(art):
        out["item_based"] = True

    out["reason"] = "present"
    return out

def list_eligible_users(art, min_ratings=1):
    if "dummy_train" not in art:
        return []
    df = art["dummy_train"]
    counts = df.notna().sum(axis=1)
    return counts[counts >= min_ratings].index.tolist()

# --- MAIN diagnostics run ---
if __name__ == "__main__":
    try:
        path, loaded = find_artifact()
    except Exception as e:
        print("Artifact load error:", e)
        raise SystemExit(1)

    art = normalize_artifact(loaded)
    print("\n--- Artifact summary ---")
    print("Source path:", path)
    print("Keys:", list(art.keys()))
    if "dummy_train" in art:
        print("dummy_train shape:", art["dummy_train"].shape)
        print("Sample usernames:", list(art["dummy_train"].index[:20]))
    if has_compact_neighbors(art):
        print("compact neighbors shape:", np.array(art["user_neighbors_idx"]).shape)
    if has_user_sim(art):
        print("user_sim shape:", np.array(art['user_sim']).shape)
    if has_item_sim(art):
        print("item_sim shape:", np.array(art['item_sim']).shape)

    # Test some users
    test_users = ["george", "Joshua", " joshua ", "00dog3"]
    for u in test_users:
        info = can_recommend_user(art, u)
        print(f"\nUser check: '{u}' ->", info)
        if info["present"]:
            # try user-based preferentially (compact -> full)
            try:
                if has_compact_neighbors(art):
                    recs = recommend_user_based_compact(art, u, topk=5)
                    print("User-based (compact) recs:", recs)
                elif has_user_sim(art):
                    recs = recommend_user_based_full_user_sim(art, u, topk=5)
                    print("User-based (full user_sim) recs:", recs)
                elif has_item_sim(art):
                    recs = recommend_item_based(art, u, topk=5)
                    print("Item-based recs:", recs)
                else:
                    print("No recommender available in artifact.")
            except Exception as e:
                print("Error generating recs for", u, "->", repr(e))

    # show counts
    eligible = list_eligible_users(art, min_ratings=1)
    print("\nNumber of users with >=1 rating:", len(eligible))
    print("First 30 eligible users:", eligible[:30])
