library("rjson")
library("pROC")


resultAll <- as.data.frame(t(as.data.frame(fromJSON(file = "cough_eval_0.json"))))
resultAll<- data.frame(sample=resultAll[seq(1,nrow(resultAll),2),],label=resultAll[seq(2,nrow(resultAll),2),])
for (i in 1:4){
  result0 <- as.data.frame(t(as.data.frame(fromJSON(file = paste("cough_eval_",i,".json",sep = "")))))
  result0 <- data.frame(sample=result0[seq(1,nrow(result0),2),],label=result0[seq(2,nrow(result0),2),])
  resultAll<- rbind(resultAll,result0)
}

resultAll$sample<- gsub("Users/zmoad/Documents/Course/06.AST_Audio/cough-heavy/","",resultAll$sample)
resultAll$sample<- gsub(".cough-heavy.wav","",resultAll$sample)
resultAll$label<- as.numeric(gsub("/m/07rwj","",resultAll$label))
dim(resultAll)


predictions<- read.table("predictions.txt",header = F,sep = ",")
dim(predictions)
colnames(predictions)<- c("p_normal","p_infect")
predictions$prediction_infected<- predictions$p_infect > predictions$p_normal
#predictions$prediction_infected[predictions$prediction_infected==TRUE]<- 0
comd<- cbind(resultAll,predictions)


err_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision =(TP)/(TP+FP)
  recall_score =(FP)/(FP+TN)
  
  f1_score=2*((precision*recall_score)/(precision+recall_score))
  accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
  False_positive_rate =(FP)/(FP+TN)
  False_negative_rate =(FN)/(FN+TP)
  
  print(paste("Precision value of the model: ",round(precision,4)))
  print(paste("Accuracy of the model: ",round(accuracy_model,4)))
  print(paste("Recall value of the model: ",round(recall_score,4)))
  print(paste("False Positive rate of the model: ",round(False_positive_rate,4)))
  
  print(paste("False Negative rate of the model: ",round(False_negative_rate,4)))
  
  print(paste("f1 score of the model: ",round(f1_score,4)))
}


head(comd)
comd$prediction_normal<- comd$p_normal > 0.5
CM= table(comd[,2], comd[,5])
# CM= as.data.frame(CM)[,-c(1,2),drop=F]
# CM$`1`<- 0
print(CM)


err_metric(CM)

roc_score=roc(comd[,5],comd[,2]) #AUC score
cat("AUC is",roc_score$auc,"\n")
plot(roc_score ,main ="ROC curve")




