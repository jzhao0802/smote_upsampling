library(tidyverse)
library(mlr)
library(palabmod)

# ------------------------------------------------------------------------------
#
# LOAD DATASETS
#
# ------------------------------------------------------------------------------

train_matched = read_rds("F:/Daniel/lookback_matching/data/train_matched.rds")
train_matched$label = as.factor(train_matched$label)

test_dataset = read_rds("F:/Daniel/lookback_matching/data/test_unmatched.rds")
test_dataset$label = as.factor(test_dataset$label)

test_smote_dataset = read_rds("F:/Daniel/lookback_matching/data/test_smote_unmatched.rds")
test_smote_dataset$label = as.factor(test_smote_dataset$label)

smote_1000_dataset = read_rds("F:/Daniel/lookback_matching/data/smote_1000_dataset.rds")
smote_1000_dataset$label = as.factor(smote_1000_dataset$label)

smote_10000_dataset= read_rds("F:/Daniel/lookback_matching/data/smote_10000_dataset.rds")
smote_10000_dataset$label = as.factor(smote_10000_dataset$label)

smote_rand_dataset= read_rds("F:/Daniel/lookback_matching/data/smote_random_dataset.rds")
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
  verbose = F,
  objective = "binary:logistic"
)

# define mlr datasets - since we're not doing any CV, blocking does not need to
# be set for the matched train dataset
train_matched <- makeClassifTask(id="train_matched", data=train_matched, target="label", positive=1)

test_dataset = makeClassifTask(id="test_dataset", data=test_dataset, target="label", positive=1)

test_smote_dataset = makeClassifTask(id="test_smote_dataset", data=test_smote_dataset, target="label", positive=1)

smote_1000_dataset = makeClassifTask(id="smote_1000_dataset", data=smote_1000_dataset, target="label", positive=1)

smote_10000_dataset = makeClassifTask(id="smote_10000_dataset", data=smote_10000_dataset, target="label", positive=1)

smote_rand_dataset = makeClassifTask(id="smote_rand_dataset", data=smote_rand_dataset, target="label", positive=1)

# train models
xgb_train_matched = train(lrn, train_matched)
xgb_smote_1000 = train(lrn, smote_1000_dataset)
xgb_smote_10000 = train(lrn, smote_10000_dataset)
xgb_smote_rand = train(lrn, smote_rand_dataset)

# predict with models (both normal and smote-d test)
pred_train_matched = predict(xgb_train_matched, test_dataset)
pred_xgb_smote_1000 = predict(xgb_smote_1000, test_dataset)
pred_xgb_smote_10000 = predict(xgb_smote_10000, test_dataset)
pred_xgb_smote_rand = predict(xgb_smote_rand, test_dataset)

pred_smote_train_matched = predict(xgb_train_matched, test_smote_dataset)
pred_smote_xgb_smote_1000 = predict(xgb_smote_1000, test_smote_dataset)
pred_smote_xgb_smote_10000 = predict(xgb_smote_10000, test_smote_dataset)
pred_smote_xgb_smote_rand = predict(xgb_smote_rand, test_smote_dataset)

# evaluate performance
perf_train_matched = perf_binned_perf_curve(pred_train_matched)
perf_xgb_smote_1000 = perf_binned_perf_curve(pred_xgb_smote_1000)
perf_xgb_smote_10000 = perf_binned_perf_curve(pred_xgb_smote_10000)
perf_xgb_smote_rand = perf_binned_perf_curve(pred_xgb_smote_rand)

perf_smote_train_matched = perf_binned_perf_curve(pred_smote_train_matched)
perf_smote_xgb_smote_1000 = perf_binned_perf_curve(pred_smote_xgb_smote_1000)
perf_smote_xgb_smote_10000 = perf_binned_perf_curve(pred_smote_xgb_smote_10000)
perf_smote_xgb_smote_rand = perf_binned_perf_curve(pred_smote_xgb_smote_rand)

# save pr curves
write_csv(perf_train_matched$curve, "F:/Daniel/lookback_matching/results/normal_test_matched_pr.csv")
write_csv(perf_xgb_smote_1000$curve, "F:/Daniel/lookback_matching/results/normal_test_smote_1000_pr.csv")
write_csv(perf_xgb_smote_10000$curve, "F:/Daniel/lookback_matching/results/normal_test_smote_10000_pr.csv")
write_csv(perf_xgb_smote_rand$curve, "F:/Daniel/lookback_matching/results/normal_test_smote_rand_pr.csv")

write_csv(perf_smote_train_matched$curve, "F:/Daniel/lookback_matching/results/smote_test_matched_pr.csv")
write_csv(perf_smote_xgb_smote_1000$curve, "F:/Daniel/lookback_matching/results/smote_test_smote_1000_pr.csv")
write_csv(perf_smote_xgb_smote_10000$curve, "F:/Daniel/lookback_matching/results/smote_test_smote_10000_pr.csv")
write_csv(perf_smote_xgb_smote_rand$curve, "F:/Daniel/lookback_matching/results/smote_test_smote_rand_pr.csv")

# save VIs of both models
vi_matched = xgboost::xgb.importance(getTaskFeatureNames(test_dataset), xgb_train_matched$learner.model)
vi_smote_1000 = xgboost::xgb.importance(getTaskFeatureNames(test_dataset), xgb_smote_1000$learner.model)
vi_smote_10000 = xgboost::xgb.importance(getTaskFeatureNames(test_dataset), xgb_smote_10000$learner.model)
vi_smote_rand = xgboost::xgb.importance(getTaskFeatureNames(test_dataset), xgb_smote_rand$learner.model)

write_csv(vi_matched, "F:/Daniel/lookback_matching/results/vi_matched.csv")
write_csv(vi_smote_1000, "F:/Daniel/lookback_matching/results/vi_smote_1000.csv")
write_csv(vi_smote_10000, "F:/Daniel/lookback_matching/results/vi_smote_10000.csv")
write_csv(vi_smote_rand, "F:/Daniel/lookback_matching/results/vi_smote_rand.csv")

# save models
write_rds(xgb_train_matched, "F:/Daniel/lookback_matching/results/xgb_matched.rds")
write_rds(xgb_smote_1000, "F:/Daniel/lookback_matching/results/xgb_smote_1000.rds")
write_rds(xgb_smote_10000, "F:/Daniel/lookback_matching/results/xgb_smote_10000.rds")
write_rds(xgb_smote_rand, "F:/Daniel/lookback_matching/results/xgb_smote_rand.rds")
