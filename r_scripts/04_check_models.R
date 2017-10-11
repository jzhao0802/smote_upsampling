library(tidyverse)
library(mlr)
library(palabmod)

# ------------------------------------------------------------------------------
#
# LOAD DATASETS
#
# ------------------------------------------------------------------------------

results_folder = "F:/Daniel/lookback_matching/results"
data_folder = "F:/Daniel/lookback_matching/data"
pics_folder = "F:/Daniel/lookback_matching/pics"

train_matched = read_rds(file.path(data_folder, "train_matched.rds"))
train_matched$label = as.factor(train_matched$label)

train_unmatched_df = read_rds(file.path(data_folder, "train_unmatched.rds"))
train_unmatched_df$label = as.factor(train_unmatched_df$label)

test_dataset = read_rds(file.path(data_folder, "test_unmatched.rds"))
test_dataset$label = as.factor(test_dataset$label)

test_smote_dataset = read_rds(file.path(data_folder, "test_smote_unmatched.rds"))
test_smote_dataset$label = as.factor(test_smote_dataset$label)

smote_1000_dataset = read_rds(file.path(data_folder, "smote_1000_dataset.rds"))
smote_1000_dataset$label = as.factor(smote_1000_dataset$label)

smote_10000_dataset= read_rds(file.path(data_folder, "smote_10000_dataset.rds"))
smote_10000_dataset$label = as.factor(smote_10000_dataset$label)

smote_rand_dataset= read_rds(file.path(data_folder, "smote_random_dataset.rds"))
smote_rand_dataset$label = as.factor(smote_rand_dataset$label)

# ------------------------------------------------------------------------------
# My last idea why smote is doing so well, especially smote random, is because
# smote will have non-integer values for certain features, which are ALWAYS 
# integers in the real data. For example, date diff vars are always integers, but
# SMOTE might generate a 34.321 for example instead of 34. So I'll round certain
# cols to the nearest integer in all smote datasets. 
# ------------------------------------------------------------------------------

cols_to_round = colnames(train_matched)[345:1026]
test_smote_dataset[cols_to_round] = lapply(test_smote_dataset[cols_to_round], as.integer)
smote_1000_dataset[cols_to_round] = lapply(smote_1000_dataset[cols_to_round], as.integer)
smote_10000_dataset[cols_to_round] = lapply(smote_10000_dataset[cols_to_round], as.integer)
smote_rand_dataset[cols_to_round] = lapply(smote_rand_dataset[cols_to_round], as.integer)

cols_to_round = colnames(train_matched)[2:344]
train_matched[cols_to_round] = lapply(train_matched[cols_to_round], function(x) round(x, 1))
train_unmatched_df[cols_to_round] = lapply(train_unmatched_df[cols_to_round], function(x) round(x, 1))
test_dataset[cols_to_round] = lapply(test_dataset[cols_to_round], function(x) round(x, 1))
test_smote_dataset[cols_to_round] = lapply(test_smote_dataset[cols_to_round], function(x) round(x, 1))
smote_1000_dataset[cols_to_round] = lapply(smote_1000_dataset[cols_to_round], function(x) round(x, 1))
smote_10000_dataset[cols_to_round] = lapply(smote_10000_dataset[cols_to_round], function(x) round(x, 1))
smote_rand_dataset[cols_to_round] = lapply(smote_rand_dataset[cols_to_round], function(x) round(x, 1))

# ------------------------------------------------------------------------------
#
# CHECK CERTAIN VARIABLES
#
# ------------------------------------------------------------------------------

plot_col <- function(col, xlim1, xlim2){
  col_s = smote_1000_dataset[1:1000, col]
  col_orig = train_unmatched_df[1:1000, col]
  df = data.frame(c(col_s, col_orig), c(rep("smote", 1000), rep("original", 1000)))
  colnames(df) = c("col", "label")
  if (xlim2 == "max"){
    xlim2 = max(df$col)
  }
  ggplot(df, aes(x=col, ..density.., fill=label)) + geom_density(alpha=.3) + xlim(xlim1,xlim2)
  ggsave(file.path(pics_folder, paste(col, "_orig_vs_smote.jpg", sep="")))
  label = "label"
  ggplot(train_unmatched_df, aes_string(x=col, fill=as.name("label"))) + 
    geom_density(alpha=.3) + 
    xlim(xlim1, xlim2)
  ggsave(file.path(pics_folder, paste(col, "_orig.jpg", sep="")))
  ggplot(smote_1000_dataset, aes_string(x=col, fill=as.name("label"))) + 
    geom_density(alpha=.3) + 
    xlim(xlim1, xlim2)
  ggsave(file.path(pics_folder, paste(col, "_smote.jpg", sep="")))
}

cols = c("D_2768_AVG_CLAIM_CNT", "G_797000_AVG_CLAIM_CNT")
sapply(cols, plot_col, 0, .5)
cols = c("D_2768_LAST_EXP_DT", "G_797000_FIRST_EXP_DT", "S_S37_LAST_EXP_", "G_371000_FIRST_EXP_DT")
sapply(cols, plot_col, 0, "max")

