library(tidyverse)
results_folder = "F:/Daniel/lookback_matching/data/"

pos = read_rds("F:/Projects/Strongbridge/data/modelling/Advanced_model_data/04_combined_train_matched_test_capped_freq_datediff.rds")
pos = pos %>% filter(subset=="pos")
pos$subset = NULL
# VERY IMPORTANT - take the top 1000 of positives so SMOTE doesn't see test pos
train_ix = sample(1:1553, 1000)
write_csv(data.frame(train_ix), file.path(results_folder, "pos_train_ix.csv"))
test_ix = setdiff(1:1553, train_ix)
write_csv(data.frame(test_ix), file.path(results_folder, "pos_test_ix.csv"))
pos_train = pos[train_ix,]
pos_test = pos[test_ix,]
write_csv(pos_train, file.path(results_folder, "pos_train.csv"))
write_csv(pos_test, file.path(results_folder, "pos_test.csv"))

# get a random subsample of negatives to have the lookback dist of the scoring
random_neg = read_rds("F:/Projects/Strongbridge/data/modelling/03_random_scoring_freq_topcoded.rds")
random_neg_index_lookback = random_neg[,c('index_date', 'lookback_date')]
random_neg_index_lookback = random_neg_index_lookback %>%  sample_n(100000)
write_csv(random_neg_index_lookback, file.path(results_folder, "date_diffs.csv"))

