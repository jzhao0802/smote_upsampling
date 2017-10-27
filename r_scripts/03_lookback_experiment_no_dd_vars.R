library(tidyverse)
library(mlr)
library(xgboost)
library(palabmod)

# ------------------------------------------------------------------------------
#
# LOAD DATASETS
#
# ------------------------------------------------------------------------------

results_folder = "F:/Daniel/lookback_matching/results_no_dd_vars"
data_folder = "F:/Daniel/lookback_matching/data"

train_matched = read_rds(file.path(data_folder, "train_matched.rds"))
train_matched$label = as.factor(train_matched$label)

train_unmatched_df = read_rds(file.path(data_folder, "train_unmatched.rds"))
train_unmatched_df$label = as.factor(train_unmatched_df$label)

test_dataset = read_rds(file.path(data_folder, "test_unmatched.rds"))
test_dataset$label = as.factor(test_dataset$label)

test_matched_dataset = read_rds(file.path(data_folder, "test_matched.rds"))
test_matched_dataset$label = as.factor(test_matched_dataset$label)

test_smote_dataset = read_rds(file.path(data_folder, "test_smote_unmatched.rds"))
test_smote_dataset$label = as.factor(test_smote_dataset$label)

duplicated_test_ix = read_csv(file.path(data_folder, "duplicated_test_ix.csv"))
test_weighted_dataset = test_dataset
test_weighted_dataset[1:553,] = test_weighted_dataset[duplicated_test_ix[['0']],]

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

# ------------------------------------------------------------------------------
# Similarly and following the discussion with the team, all freq vars are rounded
# to 1 decimal place.
# ------------------------------------------------------------------------------

cols_to_round = colnames(train_matched)[2:344]
train_matched[cols_to_round] = lapply(train_matched[cols_to_round], function(x) round(x, 1))
train_unmatched_df[cols_to_round] = lapply(train_unmatched_df[cols_to_round], function(x) round(x, 1))
test_dataset[cols_to_round] = lapply(test_dataset[cols_to_round], function(x) round(x, 1))
test_matched_dataset[cols_to_round] = lapply(test_matched_dataset[cols_to_round], function(x) round(x, 1))
test_smote_dataset[cols_to_round] = lapply(test_smote_dataset[cols_to_round], function(x) round(x, 1))
test_weighted_dataset[cols_to_round] = lapply(test_weighted_dataset[cols_to_round], function(x) round(x, 1))
smote_1000_dataset[cols_to_round] = lapply(smote_1000_dataset[cols_to_round], function(x) round(x, 1))
smote_10000_dataset[cols_to_round] = lapply(smote_10000_dataset[cols_to_round], function(x) round(x, 1))
smote_rand_dataset[cols_to_round] = lapply(smote_rand_dataset[cols_to_round], function(x) round(x, 1))

train_matched = train_matched[,1:344]
train_unmatched_df = train_unmatched_df[,1:344]
test_dataset = test_dataset[,1:344]
test_matched_dataset = test_matched_dataset[,1:344]
test_smote_dataset = test_smote_dataset[,1:344]
test_weighted_dataset = test_weighted_dataset[,1:344]
smote_1000_dataset = smote_1000_dataset[,1:344]
smote_10000_dataset = smote_10000_dataset[,1:344]
smote_rand_dataset = smote_rand_dataset[,1:344]

# ------------------------------------------------------------------------------
#
# TRAIN XGBOOST MODELS AND EVALUATE PERFORMANCE
#
# ------------------------------------------------------------------------------

# define learner
lrn <- makeLearner("classif.xgboost", predict.type="prob", predict.threshold=0.5)
lrn$par.vals = list(
  nrounds = 100,
  verbose = F,
  objective = "binary:logistic"
)

# define mlr datasets - since we're not doing any CV, blocking does not need to
# be set for the matched train dataset
train_matched <- makeClassifTask(id="train_matched", data=train_matched, 
                                 target="label", positive=1)
train_unmatched <- makeClassifTask(id="train_unmatched", data=train_unmatched_df, 
                                   target="label", positive=1)

# train with weights - IMPORTANT, weights cannot be too small otherwise XGB 
# will bot train, hhence the *10
cdf_weights = read_csv(file.path(data_folder, "cdf_weights.csv"))
cdf_weights = cdf_weights$w * 10
# make the negatives the same weight as the positives - this is basically the 
# same as having inverse class freq as weights
# cdf_weights[1001:51000] = cdf_weights[1001:51000]/50
train_unmatched_weighted <- makeClassifTask(id="train_unmatched_wighted", 
                                            data=train_unmatched_df, target="label",
                                            positive=1, weights=cdf_weights)

test_dataset = makeClassifTask(id="test_dataset", 
                               data=test_dataset, target="label", positive=1)
test_matched_dataset = makeClassifTask(id="test_matched_dataset", 
                               data=test_matched_dataset, target="label", 
                               positive=1)
test_smote_dataset = makeClassifTask(id="test_smote_dataset", 
                                     data=test_smote_dataset, target="label", 
                                     positive=1)
test_weighted_dataset = makeClassifTask(id="test_weighted_dataset", 
                                     data=test_weighted_dataset, target="label", 
                                     positive=1)

smote_1000_dataset = makeClassifTask(id="smote_1000_dataset", 
                                     data=smote_1000_dataset, target="label", 
                                     positive=1)
smote_10000_dataset = makeClassifTask(id="smote_10000_dataset", 
                                      data=smote_10000_dataset, target="label", 
                                      positive=1)
smote_rand_dataset = makeClassifTask(id="smote_rand_dataset", 
                                     data=smote_rand_dataset, target="label", 
                                     positive=1)

# setup lists for iterating through experiments
train_datasets_names = c("matched", "unmatched", "weighted", "smote1000", 
                         "smote10000", "smote_rand")
train_datasets = list(train_matched, train_unmatched, train_unmatched_weighted, 
                      smote_1000_dataset, smote_10000_dataset, 
                      smote_rand_dataset)
test_datasets_names = c("normal", "smote", "weighted", "matched")
test_datasets = list(test_dataset, test_smote_dataset, test_weighted_dataset, 
                     test_matched_dataset)

# load variable name lookup 
vlookup = read_csv(file.path(data_folder, "var_lookup.csv"))

# iterate through all combinations
for (i in 1:length(train_datasets)){
  print(i)
  # train model
  
  xgb = train(lrn, train_datasets[[i]])
  # save detailed VI table, with vlookup var names - DOESN'T WORK FOR SOME REASON
  # vi = results_xgb_splits(xgb$learner.model, train_datasets[[i]])
  # vi = plyr::join(vi, vlookup, by="Feature")
  # filename = paste("vi_", train_datasets_names[[i]], ".csv", sep="")
  # write_csv(vi, file.path(results_folder, filename))
  
  # save model
  filename = paste("model_", train_datasets_names[[i]], ".rds", sep="")
  write_rds(xgb, file.path(results_folder, filename))
  
  # predict on test datasets
  for (p in 1:length(test_datasets)){
    pred = predict(xgb, test_datasets[[p]])
    pred = perf_binned_perf_curve(pred)
    filename = paste("train_", train_datasets_names[[i]], "_test_", 
                     test_datasets_names[[p]], ".csv", sep="")
    write_csv(pred$curve, file.path(results_folder, filename))
  }
}
