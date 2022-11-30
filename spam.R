library(mlr3verse)
library(tidyverse)
library(precrec)
library(scales)
set.seed(42) # for reproducibility

# Data ----
## Info ----
task = tsk('spam')
task$help()
which(task$missings() > 0) # no missing values, phew!

## Table visualization ----
DT::datatable(
  # pick some rows to show
  data = task$data(rows = sample(x = task$nrow, size = 42)),
  caption = 'Sample of spam dataset from UCI ML repo (4601 x 58)',
  options = list(searching = FALSE)
) %>%
  DT::formatStyle(columns = 'type', backgroundColor =
    DT::styleEqual(c('spam', 'nonspam'), c('#f18384', '#9cd49a')))

## Check balance between target classes ----
autoplot(task, type = 'target') + ylim(c(0,3000))

task$data(cols = task$target_names) %>%
  group_by(type) %>%
  tally() %>%
  mutate(freq.percent = scales::percent(n/sum(n), accuracy = 0.1))

# Stratified dataset partition (train:test ratio => 3:1, 75%:25%)
split = partition(task, ratio = 0.75, stratify = TRUE)

# check train data split
length(split$train) # how many rows (emails)?
task$data(rows = split$train, cols = task$target_names) %>%
  group_by(type) %>%
  tally() %>%
  mutate(freq.percent =  scales::percent(n/sum(n), accuracy = 0.1))

# check test data split
length(split$test) # how many rows (emails)?
task$data(rows = split$test, cols = task$target_names) %>%
  group_by(type) %>%
  tally() %>%
  mutate(freq.percent = scales::percent(n/sum(n), accuracy = 0.1))

# save split
if ((!file.exists('spam/spam_split.rds'))) {
  saveRDS(split, file = 'spam/spam_split.rds')
  readr::write_lines(x = split$train, file = 'spam/train_index.txt')
}
split = readRDS(file = 'spam/spam_split.rds')
train_indx = split$train
test_indx  = split$test

# Tree complexity tuning ----
if (!file.exists('spam/tree_res.rds')) {
  # `cp` hyperparameter controls the bias-variance trade-off
  cps = c(0, 0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01,
    0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1)

  data_list = list()
  index = 1
  for (cp in cps) {
    learner = lrn('classif.rpart', cp = cp)
    learner$train(task, row_ids = train_indx)

    train_error = learner$predict(task, row_ids = train_indx)$score()
    test_error  = learner$predict(task, row_ids = test_indx )$score()

    data_list[[index]] = list(cp = cp, train_error = train_error,
      test_error = test_error)
    index = index + 1
  }
  tree_res = dplyr::bind_rows(data_list)
  saveRDS(tree_res, file = 'spam/tree_res.rds')
}

tree_res = readRDS(file = 'spam/tree_res.rds')

tree_res %>%
  tidyr::pivot_longer(cols = c('train_error', 'test_error'), names_to = 'type',
     values_to = 'error', names_pattern = '(.*)_') %>%
  ggplot(aes(x = cp, y = error, color = type)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(
    trans = c('log10', 'reverse'),
    #breaks = scales::trans_breaks('log10', function(x) 10^x),
    labels = scales::trans_format('log10', scales::math_format(10^.x))
  ) +
  labs(y = 'Missclassification Error') +
  theme_bw(base_size = 14) +
  guides(color = guide_legend('Data set'))

# Tree performance ----
tree = lrn('classif.rpart', keep_model = TRUE, cp = 0.001)
tree$predict_type = 'prob'
tree

# train
tree$train(task, row_ids = train_indx)
tree$model
#autoplot(tree) # how does the tree look like?

## Feature importance ----
vimp = tibble::enframe(tree$importance(), name = 'Variable', value = 'Importance')
vimp %>%
  mutate(Variable = forcats::fct_reorder(Variable, Importance, .desc = TRUE)) %>%
  dplyr::slice(1:15) %>% # keep only the 15 most important
  ggplot(aes(x = Variable, y = Importance, fill = Variable)) +
  scale_y_continuous(expand = c(0,0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  ggpubr::theme_classic2(base_size = 14) +
  labs(y = 'Importance', x = 'Features') +
  coord_flip()

# Predictions
tree_pred = tree$predict(task, row_ids = test_indx)
tree_pred

tree_pred$confusion # confusion matrix

tree_pred$score(msr('classif.auc')) # roc-AUC
autoplot(tree_pred, type = 'roc')

# Various classification performance metrics
mlr_measures$keys(pattern = '^classif')
tree_pred$score() # classification error (default)
tree_pred$score(msr('classif.acc'))
tree_pred$score(msr('classif.bacc'))
tree_pred$score(msr('classif.sensitivity'))
tree_pred$score(msr('classif.specificity'))
tree_pred$score(msr('classif.precision'))

# Lasso - Linear Regularized model ----
set.seed(42)
lasso = lrn('classif.cv_glmnet')
lasso$param_set$default$nfolds
lasso$param_set$default$alpha # lasso
lasso$param_set$default$nlambda # how many regularization parameters to try?
lasso$param_set$default$s # which lambda to use for prediction?
lasso$param_set$default$standardize # as it should!

lasso$train(task, row_ids = train_indx)

lasso$model # lots of features!

plot(lasso$model)
log(lasso$model$lambda.min)
log(lasso$model$lambda.1se)

## prediction performance
lasso_pred = lasso$predict(task, row_ids = test_indx)
lasso_pred
lasso_pred$confusion
lasso_pred$score()

## change lambda tuning to use misclassification error
lasso2 = lasso$clone(deep = TRUE)$reset() # remove trained model
lasso2$param_set$values = list(type.measure = 'class')

lasso2$train(task, row_ids = train_indx)
lasso2$model

lasso2_pred = lasso2$predict(task, row_ids = test_indx)
lasso2_pred$confusion
lasso2_pred$score()

# Bagging and Random Forests ----
nfeats = length(task$feature_names)
base_rf = lrn('classif.ranger', verbose = FALSE, num.threads = 16)
#' Now random forest is setup to predict responses and not probabilities of
#' spam/nonspam, so we will use the misclassification error to measure performance

if (!file.exists('spam/forest_res.rds')) {
  ntrees = c(1, seq(from = 10, to = 500, by = 10))
  mtrys = c(nfeats, ceiling(nfeats/2), ceiling(sqrt(nfeats)), 1)

  data_list = list()
  index = 1
  for (num.trees in ntrees) {
    for (mtry in mtrys) {
      message('#Trees: ', num.trees, ', mtry: ', mtry)

      base_rf$reset()
      base_rf$param_set$values$num.trees = num.trees
      base_rf$param_set$values$mtry = mtry

      # train model, get train set, test set and OOB errors
      base_rf$train(task, train_indx)
      train_error = base_rf$predict(task, train_indx)$score()
      test_error  = base_rf$predict(task, test_indx )$score()
      oob_error   = base_rf$oob_error()
      train_time  = base_rf$timings['train'] # in secs

      data_list[[index]] = tibble::tibble(ntrees = num.trees, mtry = mtry,
        train_time = train_time, train_error = train_error,
        test_error = test_error, oob_error = oob_error)
      index = index + 1
    }
  }

  forest_res = dplyr::bind_rows(data_list)

  saveRDS(forest_res, file = 'spam/forest_res.rds')
}
forest_res = readRDS(file = 'spam/forest_res.rds')

## Performance comparison for different #trees and mtry ----

# tuning mtry is important!
forest_res %>%
  mutate(mtry = as.factor(mtry)) %>%
  ggplot(aes(x = ntrees, y = test_error, color = mtry)) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = 200, color = 'red', linetype = 'dashed') +
  labs(y = 'Test Error', x = 'Number of Trees',
    title = 'Bagging vs Random Forests (mtry)') +
  theme_bw(base_size = 14)

## Time to train vs #trees and mtry ----

# parallelization is important!
forest_res %>%
  mutate(mtry = as.factor(mtry)) %>%
  ggplot(aes(x = ntrees, y = train_time, color = mtry)) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = 200, color = 'red', linetype = 'dashed') +
  labs(y = 'Time to train Forest (secs)', x = 'Number of Trees',
    title = 'Parallelized training using 16 cores') +
  theme_bw(base_size = 14)

## OOB and test error ----
# OOB error is a good approximation of the generalization error
# (train error definitely is not!)
forest_res %>%
  filter(mtry == 8) %>%
  tidyr::pivot_longer(cols = ends_with('error'), names_pattern = '(.*)_error',
    names_to = 'type', values_to = 'error') %>%
  ggplot(aes(x = ntrees, y = error, color = type)) +
  geom_line() +
  labs(x = 'Number of Trees', y = 'Misclassification Error',
    title = 'Out-Of-Bag error for generalization',
    subtitle = 'mtry = 8 features') +
  theme_bw(base_size = 14)

## Bagged Trees and Random Forest performance ----
set.seed(42)
bagged_trees = base_rf$clone(deep = TRUE)$reset()
random_forest = base_rf$clone(deep = TRUE)$reset()

bagged_trees$param_set$values$num.trees = 200
bagged_trees$param_set$values$mtry = nfeats
random_forest$param_set$values$num.trees = 200
random_forest$param_set$values$mtry = 8
random_forest$param_set$values$importance = 'permutation'

bagged_trees$train(task, train_indx)
bt_pred = bagged_trees$predict(task, test_indx)

random_forest$train(task, train_indx)
rf_pred = random_forest$predict(task, test_indx)

bt_pred$confusion
rf_pred$confusion

bt_pred$score()
rf_pred$score()

## Feature Importance (FI) ----
# a bit different from single tree FI
rf_vimp = tibble::enframe(random_forest$importance(), name = 'Variable',
  value = 'Importance')
rf_vimp %>%
  mutate(Variable = forcats::fct_reorder(Variable, Importance, .desc = TRUE)) %>%
  dplyr::slice(1:15) %>% # keep only the 15 most important
  ggplot(aes(x = Variable, y = Importance, fill = Variable)) +
  scale_y_continuous(expand = c(0,0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  ggpubr::theme_classic2(base_size = 14) +
  labs(y = 'Permutation Importance', x = 'Features') +
  coord_flip()

# Gradient-boosted Trees ----
base_xgboost = lrn('classif.xgboost', nthread = 8,
  nrounds = 150, max_depth = 5, eta = 0.3)

if (!file.exists('spam/xgboost_res.rds')) {
  max_depths = c(1,3,5,7) # how large is each tree?
  etas = c(0.001, 0.01, 0.1, 0.3) # learning rate
  nrounds = c(1, seq(from = 25, to = 1500, by = 25)) # number of trees

  param_grid = data.table::CJ(nrounds = nrounds, max_depth = max_depths,
    eta = etas, sorted = FALSE)

  data_list = list()
  index = 1
  for (row_id in 1:nrow(param_grid)) {
    max_depth = param_grid[row_id]$max_depth
    nrounds = param_grid[row_id]$nrounds
    eta = param_grid[row_id]$eta

    message('#Trees: ', nrounds, ', max_depth: ', max_depth, ', eta: ', eta)

    base_xgboost$reset()
    base_xgboost$param_set$values$nrounds = nrounds
    base_xgboost$param_set$values$max_depth = max_depth
    base_xgboost$param_set$values$eta = eta

    # train model, get train set, test set and OOB errors
    base_xgboost$train(task, train_indx)
    train_error = base_xgboost$predict(task, train_indx)$score()
    test_error  = base_xgboost$predict(task, test_indx )$score()
    train_time  = base_xgboost$timings['train'] # in secs

    data_list[[row_id]] = tibble::tibble(nrounds = nrounds,
      max_depth = max_depth, eta = eta, train_time = train_time,
      train_error = train_error, test_error = test_error)
  }

  xgboost_res = dplyr::bind_rows(data_list)

  saveRDS(xgboost_res, file = 'spam/xgboost_res.rds')
}
xgboost_res = readRDS(file = 'spam/xgboost_res.rds')

## Effect of max_depth vs nrounds ----

# max_depth = 5 seems to be the best
xgboost_res %>%
  filter(eta == 0.01) %>%
  mutate(max_depth = as.factor(max_depth)) %>%
  ggplot(aes(x = nrounds, y = test_error, color = max_depth)) +
  geom_point() +
  geom_line() +
  labs(y = 'Test Error', x = 'Number of Trees (Boosting Iterations)',
    title = 'Effect of max_depth on test error (eta = 0.01)') +
  theme_bw(base_size = 14)

## Effect of eta vs nrounds ----
# eta small => slow training
# eta large => overfitting occurs!
xgboost_res %>%
  filter(max_depth == 5) %>%
  mutate(eta = as.factor(eta)) %>%
  ggplot(aes(x = nrounds, y = test_error, color = eta)) +
  geom_point() +
  geom_line() +
  labs(y = 'Test Error', x = 'Number of Trees (Boosting Iterations)',
    title = 'Effect of eta (learning rate) on test error (max_depth = 5)') +
  theme_bw(base_size = 14)

## Training time ----
# growing larger trees takes more time
# growing more trees takes more time
xgboost_res %>%
  filter(eta == 0.01) %>%
  mutate(max_depth = as.factor(max_depth)) %>%
  ggplot(aes(x = nrounds, y = train_time, color = max_depth)) +
  geom_point() +
  geom_line() +
  labs(y = 'Time to train (secs)', x = 'Number of Trees (Boosting Iterations)',
    title = 'Parallelized training using 8 cores (eta = 0.01)') +
  theme_bw(base_size = 14)

xgboost_res %>%
  filter(max_depth == 5) %>%
  mutate(eta = as.factor(eta)) %>%
  ggplot(aes(x = nrounds, y = train_time, color = eta)) +
  geom_point() +
  geom_line() +
  labs(y = 'Time to train (secs)', x = 'Number of Trees (Boosting Iterations)',
    title = 'Parallelized training using 8 cores (max_depth = 5)') +
  theme_bw(base_size = 14)

## Boosting Trees performance ----
## choosing a good tuning config
set.seed(42)
xgb = base_xgboost$clone(deep = TRUE)$reset()
xgb$param_set$values$nrounds = 1000
xgb$param_set$values$max_depth = 5
xgb$param_set$values$eta = 0.01

xgb$train(task, train_indx)
xgb_pred = xgb$predict(task, test_indx)
xgb_pred$confusion
xgb_pred$score()

## Feature importance ----
# using `xgboost::xgb.importance`
xgb_vimp = tibble::enframe(xgb$importance(), name = 'Variable',
  value = 'Importance')
xgb_vimp %>%
  mutate(Variable = forcats::fct_reorder(Variable, Importance, .desc = TRUE)) %>%
  dplyr::slice(1:15) %>% # keep only the 15 most important
  ggplot(aes(x = Variable, y = Importance, fill = Variable)) +
  scale_y_continuous(expand = c(0,0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  ggpubr::theme_classic2(base_size = 14) +
  labs(y = 'XGBoost Importance', x = 'Features') +
  coord_flip()

# SVMs ----
base_svm = lrn('classif.svm')
base_svm$param_set$values$type = 'C-classification'

## Let's try a polynomial Kernel
poly_svm = base_svm$clone(deep = TRUE)
poly_svm$param_set$values$kernel = 'polynomial'
poly_svm$param_set$values$degree = 3
poly_svm$param_set$values$cost = 1000

poly_svm$train(task, row_ids = train_indx)
svm_pred = poly_svm$predict(task, row_ids = test_indx)
svm_pred$confusion
svm_pred$score() # ~6.6%

## Let's try a Radial Basis kernel
radial_svm = base_svm$clone(deep = TRUE)
radial_svm$param_set$values$kernel = 'radial'
radial_svm$param_set$values$gamma = 0.01
radial_svm$param_set$values$cost = 3

radial_svm$train(task, row_ids = train_indx)
svm_pred = radial_svm$predict(task, row_ids = test_indx)
svm_pred$confusion
svm_pred$score() # ~5.2% (~RF performance)

# can we do better? well, difficult to say, I couldn't
# see `tuning_svms_on_spam.R` => ~6%

# NN with a single hidden layer
set.seed(42)
nnet = lrn('classif.nnet', size = 20,
  MaxNWts = 10000, maxit = 500)
nnet$train(task, train_indx)
nnet$predict(task, test_indx)$score() # ~5.4%

# MLP NN ----
#=> see http://tiny.cc/spam-nn

# Stacked model ----
#?mlr_graphs_stacking
set.seed(42)
base_learners = list(
  lrn('classif.cv_glmnet', type.measure = 'class'), # Lasso
  lrn('classif.ranger', verbose = FALSE, num.threads = 16,
    num.trees = 200, mtry = 8), # RFs
  lrn('classif.xgboost', nthread = 8, nrounds = 1000,
    max_depth = 5, eta = 0.01), # XGBoost
  lrn('classif.svm', type = 'C-classification', kernel = 'radial',
    gamma = 0.01, cost = 3) # SVM
)
super_learner = lrn('classif.rpart', keep_model = TRUE, cp = 0.001) # tree

graph_stack = mlr3pipelines::pipeline_stacking(
  base_learners, super_learner, folds = 5, use_features = TRUE
)
graph_stack$plot()
stacked_lrn = as_learner(graph_stack)
stacked_lrn

stacked_lrn$train(task, row_ids = train_indx)

png(filename = 'stacked_tree.png', units = 'in', width = 6, height = 6, res = 300)
rpart.plot::rpart.plot(stacked_lrn$model$classif.rpart$model,
  digits = 2, extra = 103) # display misclassification error in every node
dev.off()

stacked_lrn$predict(task, row_ids = train_indx)$score()
stack_pred = stacked_lrn$predict(task, row_ids = test_indx)
stack_pred$score()

# Benchmark ----
# save all trained models
saveRDS(object = list(tree = tree, lasso = lasso, lasso2 = lasso2,
  bagged_trees = bagged_trees, random_forest = random_forest,
  xgb = xgb, poly_svm = poly_svm, radial_svm = radial_svm,
  nnet = nnet, stacked_lrn = stacked_lrn), file = 'spam/models.rds')
