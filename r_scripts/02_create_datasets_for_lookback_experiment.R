library(tidyverse)
library(mlr)
data_folder = "F:/Projects/Strongbridge/data/modelling/Advanced_model_data/"

# ------------------------------------------------------------------------------
#
# CREATE DATASETS FOR LOOKBACK MATCHING EXPERIMENT
#
# SB has 1553 positives. We train on 1000 pos, and 50 matched neg for benchmark.
# Then we train on 1000 SMOTE pos which have been lookback matched and 50 random
# negs.
# Finally we also match on all the SMOTE samples that were upsampled from the
# 1000 original positives.
#
# As per John's suggestion the positives in the test set are also SMOTE-d. 
# The 553 is upsampled and their lookback is matched too. Then we combine these
# with 1000 random negatives. This is called smote test dataset.
#
# As a baseline, we also have a test set which contains the original positives 
# with skewed lookback dist. This was used in the first experiment and is called
# basic test dataset.
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# prepare matched train dataset: 1000 pos and 50 matched neg for each
# ------------------------------------------------------------------------------

train_matched = read_rds(file.path(data_folder, "04_combined_train_matched_test_capped_freq_datediff.rds"))
pos_matched = train_matched %>% filter(subset=="pos")
pos_matched$subset = NULL
pos_train_ix = read_csv("F:/Daniel/lookback_matching/data/pos_train_ix.csv")$train_ix
pos_test_ix = read_csv("F:/Daniel/lookback_matching/data/pos_test_ix.csv")$test_ix
pos_matched_train = pos_matched[pos_train_ix,]
pos_test = pos_matched[pos_test_ix,]
neg_matched_train = train_matched %>% filter(test_patient_id %in% pos_matched_train$PATIENT_ID & subset == "train_neg")
neg_matched_train$subset = NULL
train_matched = bind_rows(pos_matched_train, neg_matched_train)
train_matched$test_patient_id[1:1000] = train_matched$PATIENT_ID[1:1000]
matching = as.factor(train_matched$test_patient_id)
train_matched$PATIENT_ID = NULL
train_matched$index_date = NULL
train_matched$lookback_date = NULL
train_matched$test_patient_id = NULL
write_rds(train_matched, "F:/Daniel/lookback_matching/data/train_matched.rds")

# ------------------------------------------------------------------------------
# create basic test dataset: 553 pos and 1000 unmatched negs for each
# ------------------------------------------------------------------------------

all_unmatched = read_rds(file.path(data_folder, "05_combined_train_unmatched_test_capped_freq_datediff.rds"))
all_negs = all_unmatched %>% filter(subset=="test_neg")
all_negs$subset = NULL
# some cols are character because they are all zeros
num_neg_cols = as.data.frame(lapply(all_negs[,colnames(all_negs)[5:1030]], as.numeric))
neg_test_unmatched = num_neg_cols[50001:603000,]
test_dataset = bind_rows(pos_test, neg_test_unmatched)
test_dataset$PATIENT_ID = NULL
test_dataset$index_date = NULL
test_dataset$lookback_date = NULL
test_dataset$test_patient_id = NULL
write_rds(test_dataset, "F:/Daniel/lookback_matching/data/test_unmatched.rds")

# ------------------------------------------------------------------------------
# create SMOTE test dataset: 553 pos and 1000 unmatched negs for each
# ------------------------------------------------------------------------------

smote_553 = read_csv("F:/Daniel/lookback_matching/data/pos_test_smote_500.csv")
smote_553$LOOKBACK = NULL
smote_553$label = rep(1, dim(smote_553)[1])
smote_553 = smote_553[,colnames(train_matched)]
smote_553 = as.data.frame(lapply(smote_553, as.numeric))
test_smote_dataset = bind_rows(smote_553, neg_test_unmatched)
write_rds(test_smote_dataset, "F:/Daniel/lookback_matching/data/test_smote_unmatched.rds")

# ------------------------------------------------------------------------------
# prepare SMOTE 1000: 1000 lookback adjusted smote pos and 50 matched neg for each
# ------------------------------------------------------------------------------

smote_1000 = read_csv("F:/Daniel/lookback_matching/data/pos_train_smote_1000.csv")
smote_1000$LOOKBACK = NULL
smote_1000$label = rep(1, 1000)
smote_1000 = smote_1000[,colnames(train_matched)]
neg_train_unmatched = num_neg_cols[1:50000,]
# some cols are character because they are all zeros
smote_1000 = as.data.frame(lapply(smote_1000, as.numeric))
smote_1000_dataset = bind_rows(smote_1000, neg_train_unmatched)
write_rds(smote_1000_dataset, "F:/Daniel/lookback_matching/data/smote_1000_dataset.rds")

# ------------------------------------------------------------------------------
# prepare SMOTE: 10000 lookback adjusted smote pos and 50 matched neg for each
# ------------------------------------------------------------------------------

smote_10000 = read_csv("F:/Daniel/lookback_matching/data/pos_train_smote_all.csv")
smote_10000$LOOKBACK = NULL
smote_10000$label = rep(1, dim(smote_10000)[1])
smote_10000 = smote_10000[,colnames(train_matched)]
# some cols are character because they are all zeros
smote_10000 = as.data.frame(lapply(smote_10000, as.numeric))
smote_10000_dataset = bind_rows(smote_10000, neg_train_unmatched)
write_rds(smote_10000_dataset, "F:/Daniel/lookback_matching/data/smote_10000_dataset.rds")

# ------------------------------------------------------------------------------
# prepare random SMOTE: this is without lookback matching
# ------------------------------------------------------------------------------

smote_rand = read_csv("F:/Daniel/lookback_matching/data/pos_train_random_smote.csv")
smote_rand$label = rep(1, dim(smote_rand)[1])
smote_rand = smote_rand[,colnames(train_matched)]
# some cols are character because they are all zeros
smote_rand = as.data.frame(lapply(smote_rand, as.numeric))
smote_rand_dataset = bind_rows(smote_rand, neg_train_unmatched)
write_rds(smote_rand_dataset, "F:/Daniel/lookback_matching/data/smote_random_dataset.rds")

# ------------------------------------------------------------------------------
# prepare unmatched train dataset: 1000 pos and 50 unmatched neg for each
# ------------------------------------------------------------------------------

pos_matched_train$PATIENT_ID = NULL
pos_matched_train$index_date = NULL
pos_matched_train$lookback_date = NULL
pos_matched_train$test_patient_id = NULL
train_unmatched = bind_rows(pos_matched_train, neg_train_unmatched)
write_rds(train_unmatched, "F:/Daniel/lookback_matching/data/train_unmatched.rds")
