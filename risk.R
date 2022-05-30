library(performance)
library(Boruta)
library(randomForest)
library(Rcpp)
library(pROC)
library(ROCR)
library(caret)
library(dplyr)
library(irr)
selectdata_train_all<- read.csv("selectdata_train_all.csv")
selectdata_test_all<- read.csv("selectdata_test_all.csv")

risk_train <- selectdata_train_all[,-2]
risk_test <- selectdata_test_all[,-2]

######
colnames(risk_train)[1] <- "TYPE"
colnames(risk_test)[1] <- "TYPE"
train1=risk_train
test1=risk_test
train1$TYPE=as.factor(train1$TYPE)
test1$TYPE=as.factor(test1$TYPE) 
train1$TYPE=factor(train1$TYPE,levels = c(1,0),labels=c("Yes","No"))
test1$TYPE=factor(test1$TYPE,levels = c(1,0),labels=c("Yes","No"))

######
trControl = trainControl(method = "repeatedcv",number = 10,repeats = 3,search = "grid",
                         classProbs = TRUE,summaryFunction = twoClassSummary )
tuneGrid=expand.grid(mtry=c(1:30))

set.seed(123)
mtry_fit=train(TYPE~.,data=train1, method = "rf", metric="ROC",tuneGrid=tuneGrid,trControl=trControl)

tuneGrid=expand.grid(mtry=c(1))
store_maxtrees=list()
for (ntree in c(50,100,150,200,250,300,350,400,450,500,550,
                600,650,700,750,800)){set.seed(123)
  rf_maxtrees=train(TYPE~.,data=train1, method = "rf", metric="ROC",tuneGrid=tuneGrid,trControl=trControl,ntree=ntree)
  key=toString(ntree)
  store_maxtrees[[key]]=rf_maxtrees}

results_tree=resamples(store_maxtrees)
summary(results_tree)

set.seed(22)
modelfit1=train(TYPE~.,data=train1, method = "rf", metric="ROC",tuneGrid=expand.grid(mtry=c(1)),ntree=500,
                trControl = trainControl(method = "repeatedcv",number = 10,repeats = 3,
                                         classProbs = TRUE,summaryFunction = twoClassSummary ))
modelfit1$finalModel
predictions1_train<- predict(modelfit1$finalModel)
confusionMatrix(predictions1_train, train1$TYPE,mode = "everything")
predictions1_prob_train=predict(modelfit1$finalModel,type = "prob")

######
predictions1_prob=predict(modelfit1,newdata = test1,type = "prob")
predictions1=predict(modelfit1,newdata = test1)

confusionMatrix(predictions1,test1$TYPE,mode = "everything")
roc_train1= roc(risk_train[,1],predictions1_prob_train[,1])
roc_test1= roc(risk_test[,1],predictions1_prob[,1])

######
#install.packages('modEvA')
library(modEvA)
aupr=AUC(obs=risk_train[,1],pred=predictions1_prob_train[,1],curve = "PR", simplif=TRUE, main = "TRAIN PR curve",plot.values=F)
aupr=AUC(obs=risk_test[,1],pred=predictions1_prob[,1],curve = "PR", simplif=TRUE, main = "TEST PR curve",plot.values=F)

######

pred_train1=prediction(predictions1_prob_train[,1],risk_train[,1])
pred_test1=prediction(predictions1_prob[,1],risk_test[,1])

cutoff.train1=coords(roc_train1,"best",ret="threshold",transpose=TRUE)

pred_train1@predictions[[1]][pred_train1@predictions[[1]]>=cutoff.train1]=1
pred_train1@predictions[[1]][pred_train1@predictions[[1]]<cutoff.train1]=0

pred_test1@predictions[[1]][pred_test1@predictions[[1]]>=cutoff.train1]=1
pred_test1@predictions[[1]][pred_test1@predictions[[1]]<cutoff.train1]=0

pred_train1@predictions[[1]]=as.factor(pred_train1@predictions[[1]])
pred_train1@predictions[[1]]=factor(pred_train1@predictions[[1]],levels = c(1,0),labels=c("Yes","No"))

pred_test1@predictions[[1]]=as.factor(pred_test1@predictions[[1]])
pred_test1@predictions[[1]]=factor(pred_test1@predictions[[1]],levels = c(1,0),labels=c("Yes","No"))

confusionMatrix(pred_train1@predictions[[1]],train1$TYPE,mode = "everything")
confusionMatrix(pred_test1@predictions[[1]],test1$TYPE,mode = "everything")

######
rets <- c("threshold", "specificity", "sensitivity", "accuracy","recall","precision")
ci.coords(roc_train1, x=cutoff.train1, input = "threshold", ret=rets)
ci.auc(roc_train1)
######
rets <- c("threshold", "specificity", "sensitivity", "accuracy","recall","precision")
ci.coords(roc_test1, x=cutoff.train1, input = "threshold", ret=rets)
ci.auc(roc_test1)

######
library(MLmetrics)
library(boot)
set.seed(123)
fci<-function(data,indices,x,y){
  d<-as.data.frame(data[indices,])
  r<-F1_Score(d[,1],d[,2],positive=1)
  r
}
test<-cbind(selectdata_train_all[,1],pred_train1@predictions[[1]])
bootout<-boot(data=test,
              R=100,
              statistic=fci
)
boot.ci(bootout,type="basic")

library(MLmetrics)
library(boot)
set.seed(123)
fci<-function(data,indices,x,y){
  d<-as.data.frame(data[indices,])
  r<-F1_Score(d[,1],d[,2],positive=1)
  r
}
test<-cbind(selectdata_test_all[,1],pred_test1@predictions[[1]])
bootout<-boot(data=test,
              R=100,
              statistic=fci
)
boot.ci(bootout,type="basic")


##############      Deleong analysis     ######################
roc.test(roc_train1, roc_test1, method = "delong")

plot.roc(roc_train1,percent = TRUE, lty = 1, lwd = 3, col = "red",
         cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5, font.lab = 2, col.main = "Black")
legend("bottomright",legend = c("TYPE_A   0.886"),
       text.font = 2, lty = 1, lwd = 3, cex = 0.8,
       col = c("red"))

plot.roc(roc_test1,percent = TRUE, lty = 1, lwd = 3, col = "red",
         cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.5, font.lab = 2, col.main = "Black")
legend("bottomright",legend = c("TYPE_A   0.868"),
       text.font = 2, lty = 1, lwd = 3, cex = 0.8,
       col = c("red"))


