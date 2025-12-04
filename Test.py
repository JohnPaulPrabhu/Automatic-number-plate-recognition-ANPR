import os
import json
import hashlib

# -------------------------------------------------------
# Compute MD5 of a file
# -------------------------------------------------------
def compute_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

# -------------------------------------------------------
# Load cached profiler or run new profiler
# -------------------------------------------------------
def load_or_profile(
    dataset_dir,
    cache_path,
    run_profiler_fn,
):
    """
    dataset_dir: folder containing H5 files
    cache_path: file to store profiler output + md5 list
    run_profiler_fn: your profiler callback that returns a dict
    """

    # find all h5 files
    h5_files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith(".h5")
    ])

    # prepare current hash map
    current_hash = {f: compute_md5(f) for f in h5_files}

    # ---------------------------------------------------
    # CASE 1 — cache exists → validate cache
    # ---------------------------------------------------
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)

            cached_hash = cache.get("file_md5", {})

            # Compare file list
            if set(cached_hash.keys()) != set(current_hash.keys()):
                print("[cache] H5 file list changed → re-run profiler")
            else:
                # Compare md5 checksums
                md5_same = all(
                    cached_hash[f] == current_hash[f]
                    for f in cached_hash
                )
                if md5_same:
                    print("[cache] Using cached profiler results")
                    return cache["profiler_result"]
                else:
                    print("[cache] H5 file content changed → re-run profiler")
        except:
            print("[cache] Cache corrupted → rebuild")

    # ---------------------------------------------------
    # CASE 2 — cache missing or invalid → run profiler
    # ---------------------------------------------------
    print("[cache] Running profiler...")
    profiler_result = run_profiler_fn(dataset_dir)

    # save cache
    with open(cache_path, "w") as f:
        json.dump({
            "file_md5": current_hash,
            "profiler_result": profiler_result
        }, f, indent=2)

    print("[cache] Saved profiler cache")
    return profiler_result
