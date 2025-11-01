import pandas as pd
from data_formating import DataFormating
from decomp_per_id import Decompose
from normalize_per_id import NormalizeMean
from holdout_decomposer_per_id import HoldoutDecomposer
from plot_decomposition import DecompPlotter

def main():
    decomp = int(input("Fill in the decomposition method (1=STL, 2=Stats Additive, 3=Stats Multiplicative): ").strip()) 

    raw_data = pd.read_csv(
        "/Users/bogdansandoiu/Documents/Neural Networks/Stocks Forecasting/Stocks-forecasting/data/raw/Industry_Month.csv",
        sep=';', decimal=',', encoding='utf-8-sig'
    ).drop(columns=["Category"]) 

    raw_data["Series"] = raw_data["Series"].astype(str)

    meta_columns = ["Series", "N", "NF", "Starting Year", "Starting Month"]
    value_columns = [c for c in raw_data.columns if c not in meta_columns]

    formating = DataFormating()
    long_data = raw_data.apply(formating.long_format, axis=1, args=(value_columns,))
    long_data = pd.concat(long_data.tolist(), ignore_index=True)

    print(f"Here is the long format data shape: {long_data.shape}\n")

    train, val, test = formating.split_the_data(long_data, number_val=18, number_test=18)

    print(f"The dataset has the following columns: {val.columns.to_list()}")
    print(train.head())
    print(f"Here is the shape for a specific sequence (id=N1877) in val:  {val[val['Series'] == 'N1877'].shape}")
    print(f"Here is the shape for a specific sequence (id=N1877) in test: {test[test['Series'] == 'N1877'].shape}")

    if decomp == 1:
        decomposer = Decompose(data=train, decomp_type="STL")
        #checks if there are <0 or ==0 values before applying log
        _, hash_lower_zero = decomposer.check_value_bigger_0_per_id()
        #here we apply log_transform
        decomposer.log_transform()
        decomp_train = decomposer.stl_decompose_per_id()
        print(f"These are the new columns from the decomp dataset with the {decomposer.decomp_type} type: {decomp_train.columns.to_list()}")
    elif decomp == 2:
        decomposer = Decompose(data=train, decomp_type="stats_decompose_additive")
        #checks if there are <0 or ==0 values before applying log
        _, hash_lower_zero = decomposer.check_value_bigger_0_per_id()
        #here we apply log_transform
        decomposer.log_transform()
        decomp_train = decomposer.normal_decompose_per_id(model="additive", require_positive=False)
        print(f"These are the new columns from the decomp dataset with the {decomposer.decomp_type} type: {decomp_train.columns.to_list()}")
    elif decomp == 3:
        decomposer = Decompose(data=train, decomp_type="stats_decompose_multiplicative")
        #checks if there are <0 or ==0 values before multiplicative decomp
        _, hash_lower_zero = decomposer.check_value_bigger_0_per_id()
        #here we apply log_transform
        decomp_train = decomposer.normal_decompose_per_id(model="multiplicative", require_positive=True)
        print(f"These are the new columns from the decomp dataset with the {decomposer.decomp_type} type: {decomp_train.columns.to_list()}")

    train_residuals = decomp_train[["Series", "date", "residuals"]]

    train_seasonal = decomp_train[["Series", "date", "seasonal"]]
    train_seasonal.to_csv(f"data/preprocessed/{decomposer.decomp_type}/train/seasonal.csv", index=False)

    train_trend = decomp_train[["Series", "date", "trend"]]
    train_trend.to_csv(f"data/preprocessed/{decomposer.decomp_type}/train/trend.csv", index=False)

    print(f" The train residuals before normalization: {train_residuals.head()}")

    normalization = NormalizeMean(train_residuals)
    mean_var_per_id = normalization.compute_mean_variance_per_id()
    # here we are going to save the mean, var of the residuals per id
    mean_var_per_id.to_csv(f"data/preprocessed/{decomposer.decomp_type}/mean_var_per_id.csv", index=False)

    train_residuals = normalization.normalize_each_value_per_id(mean_var_per_id)
    train_residuals.to_csv(f"data/preprocessed/{decomposer.decomp_type}/train/residuals.csv", index=False)

    print(f" The train residuals after normalization: {train_residuals.head()}")


    hd_decomposer = HoldoutDecomposer(
        train_decomp=decomp_train,           
        method=decomposer.decomp_type,                 
        mean_var_per_id=mean_var_per_id,     
        period=12,
    )

    val_full, val_trend, val_seasonal, val_residuals = hd_decomposer.transform(val, harmonics=3, offset_by_id=None)
    val_trend.to_csv(f"data/preprocessed/{decomposer.decomp_type}/val/trend.csv", index=False)
    val_seasonal.to_csv(f"data/preprocessed/{decomposer.decomp_type}/val/seasonal.csv", index=False)
    val_residuals.to_csv(f"data/preprocessed/{decomposer.decomp_type}/val/residuals.csv", index=False)

    val_len_by_id = val.groupby("Series").size().to_dict()
    test_full, test_trend, test_seasonal, test_residuals = hd_decomposer.transform(test, harmonics=3,  offset_by_id=val_len_by_id)
    test_trend.to_csv(f"data/preprocessed/{decomposer.decomp_type}/test/trend.csv", index=False)
    test_seasonal.to_csv(f"data/preprocessed/{decomposer.decomp_type}/test/seasonal.csv", index=False)
    test_residuals.to_csv(f"data/preprocessed/{decomposer.decomp_type}/test/residuals.csv", index=False)

    train_plot = (
        train[["Series","date","value"]]  # raw (linear) value
        .merge(
            decomp_train[["Series","date","trend","seasonal"]],
            on=["Series","date"], how="left", validate="one_to_one"
        )
        .merge(
            train_residuals[["Series","date","residuals"]],
            on=["Series","date"], how="left", validate="one_to_one"
        )
        [["Series","date","value","trend","seasonal","residuals"]]
    )

    val_plot = (
            val_full.rename(columns={"trend_fc": "trend"})
            [["Series","date","value","trend","seasonal","residuals"]]
    )
    test_plot = (
            test_full.rename(columns={"trend_fc": "trend"})
            [["Series","date","value","trend","seasonal","residuals"]]
    )

    plotter = DecompPlotter(out_root="src/results", dpi=150, figsize=(12, 8))
    plotter.plot_all(
        train_df=train_plot,
        val_df=val_plot,
        test_df=test_plot,
        method_label=decomposer.decomp_type,  # "STL", "stats_decompose_additive", "stats_decompose_multiplicative"
        series_ids=None,                      # or e.g. ["N1877", "N1880"]
        max_series=3,                      # or an int like 24
        log_domain=(decomposer.decomp_type in {"STL", "stats_decompose_additive"})
    )

    print("\nDiagnostics after per-ID normalization:")
    for name, df in [("TRAIN", train_residuals), ("VAL", val_residuals), ("TEST", test_residuals)]:
        mu = df["residuals"].mean()
        sd = df["residuals"].std()
        per_id_mu = df.groupby("Series")["residuals"].mean().abs()
        per_id_sd = df.groupby("Series")["residuals"].std()
        print(f"[{name}] global mean/std: {mu:.4f} / {sd:.4f} | "
              f"per-ID |mean|>0.10: {(per_id_mu > 0.10).sum()} / {per_id_mu.size} | "
              f"per-ID std outside [0.8,1.2]: {((per_id_sd < 0.8) | (per_id_sd > 1.2)).sum()} / {per_id_sd.size}")



if __name__ == "__main__":
    main()