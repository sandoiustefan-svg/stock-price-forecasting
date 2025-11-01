import numpy as np
import tensorflow as tf

def make_cross_split_val_ds(w, train_df, val_df):
    """
    Build a validation tf.data.Dataset where each element is:
      X: last w.input_width rows from TRAIN per ID (features = w.feature_columns)
      y: first w.label_width rows from VAL per ID (labels = w.label_columns)

    Assumes `w` is your WindowGenerator with:
      - w.id_col, w.time_col
      - w.input_width (e.g. 36), w.label_width (e.g. 18)
      - w.feature_columns (numeric-only features, excludes id/time)
      - w.label_columns (subset of feature_columns)
      - w.batch_size
    """
    id_col   = w.id_col
    time_col = w.time_col
    in_w     = w.input_width
    out_w    = w.label_width
    feats    = w.feature_columns
    labels   = w.label_columns

    if id_col is None or time_col is None:
        raise ValueError("w.id_col and w.time_col must be set.")

    # Iterate IDs in deterministic order (train order)
    ids_train = train_df[id_col].drop_duplicates().tolist()
    ids_val   = set(val_df[id_col].unique())

    X_list, Y_list, used_ids, skipped = [], [], [], {}

    for idv in ids_train:
        if idv not in ids_val:
            skipped[idv] = "missing in val"
            continue

        g_train = train_df[train_df[id_col] == idv].sort_values(time_col)
        g_val   = val_df[val_df[id_col]   == idv].sort_values(time_col)

        if len(g_train) < in_w:
            skipped[idv] = f"need ≥{in_w} train rows, got {len(g_train)}"
            continue
        if len(g_val) < out_w:
            skipped[idv] = f"need ≥{out_w} val rows, got {len(g_val)}"
            continue

        X = g_train[feats].tail(in_w).to_numpy(dtype=np.float32)      # (in_w, F)
        y = g_val[labels].head(out_w).to_numpy(dtype=np.float32)      # (out_w, L)

        X_list.append(X)
        Y_list.append(y)
        used_ids.append(idv)

    if not X_list:
        raise ValueError("No IDs had enough data to form cross-split validation windows.")

    X_arr = np.stack(X_list, axis=0)  # (M, in_w, F)
    Y_arr = np.stack(Y_list, axis=0)  # (M, out_w, L)

    ds = tf.data.Dataset.from_tensor_slices((X_arr, Y_arr)).batch(w.batch_size, drop_remainder=False)
    opts = tf.data.Options(); opts.deterministic = True
    ds = ds.with_options(opts)

    return ds, used_ids, skipped
