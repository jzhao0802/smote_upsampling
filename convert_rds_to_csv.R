library(tidyverse)
data_folder = "F:/Projects/Strongbridge/data/modelling/"
results_folder = "F:/Daniel/lookback_matching/data"
freqs = read_rds(file.path(data_folder, "01_train_combined_common_freq_topcoded.rds"))
date_diffs = read_rds(file.path(data_folder, "01_train_combined_date_differences_topcoded.rds"))
write_csv(freqs, file.path(results_folder, "freqs.csv"))
write_csv(date_diffs, file.path(results_folder, "date_diffs.csv"))
