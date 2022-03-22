set.seed(200)
data<- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
View(data)
colnames(data)
is.data.frame(data)
dim(data)
install.packages("skimr")
set.seed(200)
library("skimr")
skim(data)
summary(data)
install.packages("tidyverse")
library("tidyverse")
DataExplorer::plot_bar(data, ncol = 3)
DataExplorer::plot_histogram(data, ncol = 3)
DataExplorer::plot_boxplot(data, by = "fatal_mi", ncol = 3)
set.seed(200)
q1 <- quantile(data$age, 0.01)         
q99 <- quantile(data$age, 0.99)       
data[data$age < q1,]$age <- q1  
data[data$age > q99,]$age < -q99
q1 <- quantile(data$creatinine_phosphokinase, 0.01)         
q99 <- quantile(data$creatinine_phosphokinase, 0.99)       
data[data$creatinine_phosphokinase < q1,]$creatinine_phosphokinase <- q1  
data[data$creatinine_phosphokinase > q99,]$creatinine_phosphokinase < -q99
q1 <- quantile(data$ejection_fraction, 0.01)         
q99 <- quantile(data$ejection_fraction, 0.99)       
data[data$ejection_fraction < q1,]$ejection_fraction <- q1  
data[data$ejection_fraction > q99,]$ejection_fraction < -q99
q5 <- quantile(data$platelets, 0.05)         
q95 <- quantile(data$platelets, 0.95)       
data[data$platelets < q5,]$platelets <- q5  
data[data$platelets > q95,]$platelets < -q95
q1 <- quantile(data$serum_creatinine, 0.01)         
q99 <- quantile(data$serum_creatinine, 0.99)       
data[data$serum_creatinine < q1,]$serum_creatinine <- q1  
data[data$serum_creatinine > q99,]$serum_creatinine < -q99
q5 <- quantile(data$serum_sodium, 0.05)         
q95 <- quantile(data$serum_sodium, 0.95)       
data[data$serum_sodium < q5,]$serum_sodium <- q5  
data[data$serum_sodium > q95,]$serum_sodium < -q95
q1 <- quantile(data$time, 0.01)         
q99 <- quantile(data$time, 0.99)       
data[data$time < q1,]$time <- q1  
data[data$time > q99,]$time < -q99
summary(data)
install.packages("ggforce")

set.seed(200)
data1<-data
data1.par <- data1 %>%
  select(fatal_mi,smoking, high_blood_pressure, diabetes,anaemia) %>%
  group_by(fatal_mi, smoking, high_blood_pressure, diabetes,anaemia) %>%
  summarize(value = n())
library("ggforce")

data1.par$high_blood_pressure<-as.character(data1.par$high_blood_pressure)
data1.par$smoking<-as.character(data1.par$smoking)

data1.par$diabetes<-as.character(data1.par$diabetes)
data1.par$anaemia<-as.character(data1.par$anaemia)
data1.par$fatal_mi<-as.character(data1.par$fatal_mi)
ggplot(data1.par  %>% gather_set_data(x = c(1,c(2:5))),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor( fatal_mi)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()
install.packages("data.table")
install.packages("mlr3verse")
library("data.table")
library("mlr3verse")
set.seed(200)
data$fatal_mi<-as.factor(data$fatal_mi)
Heart_task <- TaskClassif$new(id = "Heart",
                              backend =data, 
                              target = "fatal_mi",
                              positive ="0")
cv5 <- rsmp("cv",folds = 5)
cv5$instantiate(Heart_task)
bootstrap <- rsmp("bootstrap",ratio=0.8)
bootstrap$instantiate(Heart_task)
install.packages("kknn")
library("kknn")
set.seed(200)

lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
lrn_svm <- lrn("classif.svm", predict_type = "prob")
lrn_naive_bayes <- lrn("classif.naive_bayes", predict_type = "prob")

set.seed(200)
res <- benchmark(data.table(
  task       = list(Heart_task),
  learner    = list(lrn_rpart,lrn_rpart,lrn_ranger,lrn_ranger,lrn_svm ,lrn_svm ,lrn_naive_bayes,lrn_naive_bayes),
  resampling = list(bootstrap,cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.tpr")))
print(res)

set.seed(200)
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
lrn_log_reg $param_set


install.packages(randomForest)
library(randomForest)
set.seed(200)
index <- sample(2,nrow(data),replace = TRUE,prob=c(0.7,0.3))
traindata <- data[index==1,]
testdata <- data[index==2,]
set.seed(1234)
x<-list()
y<-list()
y_acc<-list()
y_tpr<-list()
y_fpr<-list()
for (i in 1:12){
  mtry_fit<- randomForest(as.factor(fatal_mi)~., data=traindata, mtry=i)
  pred.lr <- predict(mtry_fit, testdata , type = "response")
  print(pred.lr)
  conf.mat <- table(`true fatal_mi` = testdata$fatal_mi, `predict fatal_mi` = pred.lr=='1')
  print(conf.mat)
  x[i]<-i
  y_acc[i]<-(conf.mat[1,1]+conf.mat[2,2])/(conf.mat[1,1]+conf.mat[2,2]+conf.mat[2,1]+conf.mat[1,2])
  y_tpr[i]<-(conf.mat[1,1])/(conf.mat[1,1]+conf.mat[1,2])
  y_fpr[i]<-(conf.mat[2,1])/(conf.mat[2,1]+conf.mat[2,2])
}
plot(x,y_acc,type = "b")
plot(x,y_tpr,type = "b")
plot(x,y_fpr,type = "b")


library(randomForest)
set.seed(200)
index <- sample(2,nrow(data),replace = TRUE,prob=c(0.7,0.3))
traindata <- data[index==1,]
testdata <- data[index==2,]
set.seed(1234)
x<-vector()
y_acc<-vector()
y_tpr<-vector()
y_fpr<-vector()
rf_ntree<- randomForest(fatal_mi ~ ., data=traindata,  
                        ntree=800,important=TRUE,proximity=TRUE)
plot(rf_ntree)
print(rf_ntree)


library(randomForest)
set.seed(200)
index <- sample(2,nrow(data),replace = TRUE,prob=c(0.7,0.3))
traindata <- data[index==1,]
testdata <- data[index==2,]
set.seed(1234)
x<-vector()
y_acc<-vector()
y_tpr<-vector()
y_fpr<-vector()
for (i in 1:10){
  mtry_fit<- randomForest(as.factor(fatal_mi)~., data=traindata, max.depth =i) 
  pred.lr <- predict(mtry_fit, testdata , type = "response")
  pred.lr<-as.numeric(pred.lr)
  
  conf.mat <- table(`true fatal_mi` = testdata$fatal_mi, `predict fatal_mi` = pred.lr>1)
  print(conf.mat)
  x[i]<-i
  y_acc[i]<-(conf.mat[1,1]+conf.mat[2,2])/(conf.mat[1,1]+conf.mat[2,2]+conf.mat[2,1]+conf.mat[1,2])
  y_tpr[i]<-(conf.mat[1,1])/(conf.mat[1,1]+conf.mat[1,2])
  y_fpr[i]<-(conf.mat[2,1])/(conf.mat[2,1]+conf.mat[2,2])
}
plot(x,y_acc,type = "b")
plot(x,y_tpr,type = "b")
plot(x,y_fpr,type = "b")




library(randomForest)
set.seed(200)
data4<-data
index <- sample(2,nrow(data4),replace = TRUE,prob=c(0.7,0.3))
traindata <- data4[index==1,]
testdata <- data4[index==2,]
set.seed(1234)
x<-vector()
y_acc<-vector()
y_tpr<-vector()
y_fpr<-vector()
for (i in 1:10){
  mtry_fit<- randomForest(as.factor(fatal_mi)~., data=traindata, min.node.size =i) 
  pred.lr <- predict(mtry_fit, testdata , type = "response")
  conf.mat <- table(`true fatal_mi` = testdata$fatal_mi, `predict fatal_mi` = pred.lr=='1')
  
  x[i]<-i
  y_acc[i]<-mean(conf.mat[1,1]+conf.mat[2,2])/(conf.mat[1,1]+conf.mat[2,2]+conf.mat[2,1]+conf.mat[1,2])
  y_tpr[i]<-mean(conf.mat[1,1])/(conf.mat[1,1]+conf.mat[1,2])
  y_fpr[i]<-mean(conf.mat[2,1])/(conf.mat[2,1]+conf.mat[2,2])
}
plot(x,y_acc,type = "b")
plot(x,y_tpr,type = "b")
plot(x,y_fpr,type = "b")



set.seed(200)
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")


search_space = ps(
  num.trees = p_int(lower = 1, upper = 500),
  mtry = p_int(lower = 1, upper = 12),
  max.depth=p_int(lower = 1, upper = 5),
  min.node.size=p_int(lower = 1, upper = 10)
)

hout<-rsmp("cv",folds = 5)
hout$instantiate(Heart_task)
measures = msrs(c("classif.acc", "classif.fpr","classif.tpr","classif.auc"))
evals20 = trm("evals", n_evals = 20)
instance2 = TuningInstanceMultiCrit$new(
  task =Heart_task,
  learner = lrn_ranger,
  resampling = hout,
  measure = measures,
  search_space = search_space,
  terminator = evals20
)

tuner = tnr("grid_search", resolution = 20)
tuner$optimize(instance2)


instance2$result_learner_param_vals
instance2$result_y


install.packages("ggplot2")
library(ggplot2)


library("mlr3verse")
set.seed(123)

data8<-data
task = as_task_classif(data8, target = "fatal_mi")
task


set.seed(200)
split = partition(task,ratio =0.7)
task$set_row_roles(split$test, "holdout")

library(ggplot2)
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
lrn_ranger$param_set$values = instance2$result_learner_param_vals[[1]]
lrn_ranger$train(task, row_ids = split$train)
prediction1 = lrn_ranger$predict(task, row_ids = split$test)
autoplot(prediction1,type ="roc")
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
lrn_ranger$param_set$values = instance2$result_learner_param_vals[[2]]
lrn_ranger$train(task, row_ids = split$train)
prediction1 = lrn_ranger$predict(task, row_ids = split$test)
autoplot(prediction1,type ="roc")
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
lrn_ranger$param_set$values = instance2$result_learner_param_vals[[3]]
lrn_ranger$train(task, row_ids = split$train)
prediction1 = lrn_ranger$predict(task, row_ids = split$test)
autoplot(prediction1,type ="roc")
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
lrn_ranger$param_set$values = instance2$result_learner_param_vals[[4]]
lrn_ranger$train(task, row_ids = split$train)
prediction1 = lrn_ranger$predict(task, row_ids = split$test)
autoplot(prediction1,type ="roc")
lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
lrn_ranger$param_set$values = instance2$result_learner_param_vals[[5]]
lrn_ranger$train(task, row_ids = split$train)
prediction1 = lrn_ranger$predict(task, row_ids = split$test)
autoplot(prediction1,type ="roc")


x<-vector()
y_acc<-vector()
y_tpr<-vector()
y_fpr<-vector()
y_auc<-vector()
for(i in 1:length(instance2$result_learner_param_vals)){
  lrn_ranger <- lrn("classif.ranger", predict_type = "prob")
  lrn_ranger$param_set$values = instance2$result_learner_param_vals[[i]]
  lrn_ranger$train(task, row_ids = split$train)
  prediction1 = lrn_ranger$predict(task, row_ids = split$test)
  library(ggplot2)
  autoplot(prediction1,type ="roc")
  x[i]<-i
  y_acc[i]<-mean(prediction1$confusion[1,1]+prediction1$confusion[2,2])/(prediction1$confusion[1,1]+prediction1$confusion[2,2]+prediction1$confusion[2,1]+prediction1$confusion[1,2])
  y_tpr[i]<-mean(prediction1$confusion[1,1])/(prediction1$confusion[1,1]+prediction1$confusion[1,2])
  y_fpr[i]<-mean(prediction1$confusion[2,1])/(prediction1$confusion[2,1]+prediction1$confusion[2,2])
  y_auc[i]<-prediction1$score(msr("classif.auc"))
}
print(y_acc)
print(y_tpr)
print(y_fpr)
print(y_auc)
plot(x,y_acc,type = "b")
plot(x,y_tpr,type = "b")
plot(x,y_fpr,type = "b")
plot(x,y_auc,type = "b")


set.seed(200)
lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10),
  maxdepth= p_int(lower = 1, upper = 5)
)

hout<-rsmp("cv",folds = 5)
hout$instantiate(Heart_task)
measures = msrs(c("classif.acc", "classif.fpr","classif.tpr","classif.auc"))
evals20 = trm("evals", n_evals = 20)
instance3 = TuningInstanceMultiCrit$new(
  task =Heart_task,
  learner = lrn_rpart,
  resampling = hout,
  measure = measures,
  search_space = search_space,
  terminator = evals20
)



tuner = tnr("grid_search", resolution = 10)

tuner$optimize(instance3)



set.seed(200)
data8<-data
data8$fatal_mi<- as.factor(data8$fatal_mi)
task1 = as_task_classif(data8, target = "fatal_mi")
split = partition(task1, ratio = 0.8) 
task$set_row_roles(split$test, "holdout")
x<-vector()
y_acc<-vector()
y_tpr<-vector()
y_fpr<-vector()
y_auc<-vector()
for( i in 1:length(instance2$result_learner_param_vals)){
  lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
  lrn_rpart$param_set$values = instance3$result_learner_param_vals[[i]]
  print(lrn_rpart$param_set$values)
  lrn_rpart$train(task1, row_ids = split$train)
  prediction1 = lrn_rpart$predict(task1, row_ids = split$test)
  print(prediction1$confusion)
  x[i]<-i
  y_acc[i]<-mean(prediction1$confusion[1,1]+prediction1$confusion[2,2])/(prediction1$confusion[1,1]+prediction1$confusion[2,2]+prediction1$confusion[2,1]+prediction1$confusion[1,2])
  y_tpr[i]<-mean(prediction1$confusion[1,1])/(prediction1$confusion[1,1]+prediction1$confusion[1,2])
  y_fpr[i]<-mean(prediction1$confusion[2,1])/(prediction1$confusion[2,1]+prediction1$confusion[2,2])
  y_auc[i]<-prediction1$score(msr("classif.auc"))
}
print(y_acc)
print(y_tpr)
print(y_fpr)
print(y_auc)
plot(x,y_acc,type = "b")
plot(x,y_tpr,type = "b")
plot(x,y_fpr,type = "b")
plot(x,y_auc,type = "b")

set.seed(100)
lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10),
  maxdepth= p_int(lower = 1, upper = 5)
)

hout<-rsmp("cv",folds = 5)
hout$instantiate(Heart_task)
measures = msrs(c("classif.acc", "classif.fpr","classif.tpr","classif.auc"))
evals20 = trm("evals", n_evals = 20)
instance3 = TuningInstanceMultiCrit$new(
  task =Heart_task,
  learner = lrn_rpart,
  resampling = hout,
  measure = measures,
  search_space = search_space,
  terminator = evals20
)

instance3

tuner = tnr("grid_search", resolution = 10)

tuner$optimize(instance3)

set.seed(200)
data8<-data
data8$fatal_mi<- as.factor(data8$fatal_mi)
task1 = as_task_classif(data8, target = "fatal_mi")
split = partition(task1, ratio = 0.8) 
task$set_row_roles(split$test, "holdout")
x<-vector()
y_acc<-vector()
y_tpr<-vector()
y_fpr<-vector()
y_auc<-vector()
for( i in 1:length(instance2$result_learner_param_vals)){
  lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
  lrn_rpart$param_set$values = instance3$result_learner_param_vals[[i]]
  print(lrn_rpart$param_set$values)
  lrn_rpart$train(task1, row_ids = split$train)
  prediction1 = lrn_rpart$predict(task1, row_ids = split$test)
  print(prediction1$confusion)
  x[i]<-i
  y_acc[i]<-mean(prediction1$confusion[1,1]+prediction1$confusion[2,2])/(prediction1$confusion[1,1]+prediction1$confusion[2,2]+prediction1$confusion[2,1]+prediction1$confusion[1,2])
  y_tpr[i]<-mean(prediction1$confusion[1,1])/(prediction1$confusion[1,1]+prediction1$confusion[1,2])
  y_fpr[i]<-mean(prediction1$confusion[2,1])/(prediction1$confusion[2,1]+prediction1$confusion[2,2])
  y_auc[i]<-prediction1$score(msr("classif.auc"))
}
print(y_acc)
print(y_tpr)
print(y_fpr)
print(y_auc)
plot(x,y_acc,type = "b")
plot(x,y_tpr,type = "b")
plot(x,y_fpr,type = "b")
plot(x,y_auc,type = "b")

