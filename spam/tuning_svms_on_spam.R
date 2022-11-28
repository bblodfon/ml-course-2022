library(mlr3verse)
library(tidyverse)
library(mlr3mbo)
library(progressr)
# should take 5-10min on an 8 core laptop

# reproducibility
set.seed(42)

# Progress bars
options(progressr.enable = TRUE)
handlers(global = TRUE)
handlers('progress')

# Logging (less)
lgr::get_logger('bbotk')$set_threshold('warn')
lgr::get_logger('mlr3')$set_threshold('warn')

split = readRDS(file = 'spam/spam_split.rds')
train_indx = split$train
test_indx  = split$test

base_svm = mlr3tuningspaces::lts(lrn('classif.svm'))
base_svm$param_set$values$type = 'C-classification' # needs to be explicitly set
base_svm$param_set$values$degree = NULL
base_svm$param_set$values$kernel = 'radial'

base_svm$param_set$values

svm_at = AutoTuner$new(
  learner = base_svm,
  resampling = rsmp('cv', folds = 5),
  measure = msr('classif.ce'),
  terminator = trm('evals', n_evals = 100),
  tuner = tnr('mbo')
)

future::plan('multisession')
svm_at$train(task, row_ids = train_indx)

autoplot(svm_at$tuning_instance, type = 'performance')
autoplot(svm_at$tuning_instance, type = 'parameter', trafo = TRUE)
autoplot(svm_at$tuning_instance, type = 'surface')

best_hpc = svm_at$archive$best()$x_domain[[1L]]
best_hpc
svm = base_svm$clone(deep = TRUE)
svm$param_set$values = mlr3misc::insert_named(svm$param_set$values, best_hpc)
svm
svm$train(task, train_indx)

svm_pred = svm$predict(task, row_ids = test_indx)
svm_pred$confusion
svm_pred$score() # ~6%
