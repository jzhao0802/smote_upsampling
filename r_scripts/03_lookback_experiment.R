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
#
# TRAIN XGBOOST MODELS AND EVALUATE PERFORMANCE
#
# ------------------------------------------------------------------------------

# define learner
lrn <- makeLearner("classif.xgboost", predict.type="prob", predict.threshold=0.5)
lrn$par.vals = list(
  nrounds = 100,
  verbose = T,
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
test_smote_dataset = makeClassifTask(id="test_smote_dataset", 
                                     data=test_smote_dataset, target="label", 
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
test_datasets_names = c("normal", "smote")
test_datasets = list(test_dataset, test_smote_dataset)

# iterate through all combinations
for (i in 1:length(train_datasets)){
  # train model
  xgb = train(lrn, train_datasets[[i]])
  
  # save VI
  vi = xgboost::xgb.importance(getTaskFeatureNames(train_datasets[[i]]), 
                               xgb$learner.model)
  filename = paste("vi_", train_datasets_names[[i]], ".csv", sep="")
  write_csv(vi, file.path(results_folder, filename))
  
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
