#Loading required packages
install.packages('tidyverse')
library(tidyverse)
install.packages('ggplot2')
library(ggplot2)
install.packages('caret')
library(caret)
install.packages('caretEnsemble')
library(caretEnsemble)
install.packages('psych')
library(psych)
install.packages('Amelia')
library(Amelia)
install.packages('mice')
library(mice)
install.packages('GGally')
library(GGally)
install.packages('rpart')
library(rpart)
install.packages('randomForest')
library(randomForest)

library(readr)
dataset <- read_csv("C:/Users/Omer KOCAK/Desktop/Deu/Machine Learning/Term Project/Term_Project/train.csv")
View(dataset)

library(Boruta)

# Decide if a variable is important or not using Boruta
boruta_output <- Boruta(Response ~ ., data=na.omit(dataset), doTrace=2)  # perform Boruta search
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  # collect Confirmed and Tentative variables
print(boruta_signif)

data <- dataset[,c(boruta_signif,"Response")]

library(Amelia)
png(file="C:/Users/Omer KOCAK/Desktop/Deu/Machine Learning/Term Project/Term_Project/plots/missingnessMap.jpg",width = 600,height = 350)
missmap(data)
dev.off()

ageLabels <- c("20-29","30-39","40-49","50-59","60-69","70-79","80-89")
data$Age <- cut(data$Age,breaks = seq(from=min(data$Age),to=max(data$Age+10),by=10),right = FALSE,labels = ageLabels)


data$Gender<-as.factor(data$Gender)
data$Age<-as.factor(data$Age)
data$Driving_License<-as.factor(data$Driving_License)
data$Region_Code<-as.factor(data$Region_Code)
data$Vehicle_Age<-as.factor(data$Vehicle_Age)
data$Annual_Premium<-as.factor(data$Annual_Premium)
data$Policy_Sales_Channel<-as.factor(data$Policy_Sales_Channel)
data$Response<-as.factor(data$Response)

install.packages("caret")
install.packages("rlang")
install.packages("klaR")
library(caret)
library(e1071)
library(klaR)
#Building a model
#split data into training and test data sets
indxTrain <- createDataPartition(y = data$Response,p = 0.75,list = FALSE)
training <- data[indxTrain,]
testing <- data[-indxTrain,]


#naive bayes
nb <- naiveBayes(as.factor(training$Response)~., data = training[,-8],na.action = na.pass,metric="Accuracy")
prediction <- predict(nb,testing,type="class")
table (prediction,testing$Response,dnn = c("Prediction","Actual"))
Metrics::accuracy(predicted = prediction,actual = testing$Response) #for calculating accuracy

png(file="C:/Users/Omer KOCAK/Desktop/Deu/Machine Learning/Term Project/Term_Project/plots/ConfusionMatrix.jpg",width = 600,height = 350)
plot(table (prediction,testing$Response,dnn = c("Prediction","Actual")),main="Confusion Matrix")
dev.off()

#with laplace value
nbLaplace <- naiveBayes(as.factor(training$Response)~., data = training[,-8],na.action = na.pass,laplace = 1,metric="Accuracy")
predictionLaplace <- predict(nbLaplace,testing,type="class")
table (predictionLaplace,testing$Response,dnn = c("Prediction","Actual"))
Metrics::accuracy(predicted = predictionLaplace,actual = testing$Response)


png(file="C:/Users/Omer KOCAK/Desktop/Deu/Machine Learning/Term Project/Term_Project/plots/ConfusionMatrixWithLaplace.jpg",width = 600,height = 350)
plot(table (predictionLaplace,testing$Response,dnn = c("Prediction","Actual")),main="Confusion Matrix With Laplace")
dev.off()

#cross validation
install.packages("fastmap")
library(fastmap)
install.packages("klaR")
library(klaR)
install.packages("caret")
library(caret)
library(e1071)

#5-k fold cross validation
trainControl <- trainControl(method = "cv",number = 5)
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
model <- train(training$Response~.,training,trControl=trainControl, method="nb",tuneGrid=grid)
print(model)

#5-k fold cross validation with repeated 3 times
trainControl <- trainControl(method = "repeatedcv",number = 5,repeats=3)
model <- train(data$Response~.,data=data,trControl = trainControl,method="nb")
print(model)
