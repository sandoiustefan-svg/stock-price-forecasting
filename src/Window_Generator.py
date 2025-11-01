import numpy as np
import tensorflow as tf

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None, id_col=None, time_col=None, batch_size=32, shuffle=False):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # ID + features
        self.id_col = id_col
        self.time_col = time_col
        self.batch_size = batch_size
        self.shuffle = shuffle

        # -------- pick feature columns: drop id/time, keep numeric-only --------
        drop_cols = [c for c in [id_col, time_col] if c is not None]
        feat_df = train_df.drop(columns=drop_cols, errors="ignore")
        feat_df = feat_df.select_dtypes(include=[np.number])  # only numeric
        self.feature_columns = feat_df.columns.tolist()
        if not self.feature_columns:
            raise ValueError("No numeric feature columns found after dropping id/time.")

        # Label column indices (w.r.t. feature columns)
        self.label_columns = label_columns
        self.column_indices = {name: i for i, name in enumerate(self.feature_columns)}
        if label_columns is not None:
            self.label_columns_indices = {name: self.column_indices[name]
                                          for name in label_columns}

        # Window params
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
            f'ID column: {self.id_col}',
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def _make_one(self, arr):
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=arr,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,               # keep order inside each ID
            batch_size=self.batch_size,
        )
        return ds.map(self.split_window)

    def make_dataset(self, data):
        # If no id_col, just build one dataset in order
        if self.id_col is None:
            arr = np.asarray(data[self.feature_columns], dtype=np.float32)
            ds = self._make_one(arr)
        else:
            # Build one dataset per ID (no cross-ID windows), then concatenate in ID order
            ds = None
            for _, g in data.groupby(self.id_col, sort=False):
                arr = np.asarray(g[self.feature_columns], dtype=np.float32)
                dsg = self._make_one(arr)
                ds = dsg if ds is None else ds.concatenate(dsg)

        # Optional global shuffle after concatenation (keeps per-epoch order if False)
        if self.shuffle:
            ds = ds.shuffle(1024, reshuffle_each_iteration=True)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
