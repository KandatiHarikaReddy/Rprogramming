

library(ggplot2)
library(readr) 
system("ls ../input")
system("ls ../input", intern=TRUE)
rm(list=ls())
library(readr)
library(caTools)
library(e1071)
library(rpart)
library(rpart.plot)
library(wordcloud)
library(tm)
library(SnowballC)
library(ROCR)
library(pROC)
library(RColorBrewer)
library(stringr)

setwd("C:/R/Kaggle/Ham or Spam")
sms= read.csv("C:/R/Kaggle/Ham or Spam/spam.csv",header=TRUE)
sms<- sms[,1:2]

names(sms) <- c("label","message")
levels(as.factor(sms$label))
sms$message<- as.character(sms$message)
str(sms)
which(!complete.cases(sms))
table(sms$label)
prop.table(table(sms$label))
sms$text_length<- nchar(sms$message)
hist(sms$text_length)
ggplot(sms,aes(text_length,fill=label))+geom_histogram(binwidth = 6)+
  facet_wrap(~label) 
histogram(~text_length|label,data=sms)
bag <- Corpus(VectorSource(sms$message))
print(bag)
inspect(bag[1:3])
bag <- tm_map(bag, tolower)
bag <- tm_map(bag, removeNumbers)
bag <- tm_map(bag, stemDocument)
bag <- tm_map(bag, removePunctuation)
bag <- tm_map(bag, removeWords, c(stopwords("english")))
bag <- tm_map(bag, stripWhitespace)

print(bag)
inspect(bag[1:3])

graphics.off()
wordcloud(bag, max.words=200,scale=c(3,1),colors=brewer.pal(6,"Dark2"))
frequencies <- DocumentTermMatrix(bag)
frequencies
findFreqTerms(frequencies, lowfreq = 200)
sparseWords <- removeSparseTerms(frequencies, 0.995)
sparseWords
freq <- colSums(as.matrix(sparseWords))
length(freq)
ord <- order(freq)
ord                
findAssocs(sparseWords, c('call','get'), corlimit=0.10)
library(wordcloud)
set.seed(142)
wf <- data.frame(word = names(freq), freq = freq)
head(wf)
wordcloud(names(freq), freq, max.words = 5000, scale = c(6, .1), colors = brewer.pal(6, 'Dark2'))
ham_cloud<- which(sms$label=="spam")
spam_cloud<- which(sms$label=="ham")

wordcloud(bag[ham_cloud],min.freq=40)
wordcloud(bag[spam_cloud],min.freq=40)
library(ggplot2)
chart <- ggplot(subset(wf, freq >100), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart
sparseWords <- as.data.frame(as.matrix(sparseWords))
colnames(sparseWords) <- make.names(colnames(sparseWords))
str(sparseWords)
sparseWords$label <- sms$label
colnames(sparseWords)
set.seed(987)
split <- sample.split(sparseWords$label, SplitRatio = 0.75)
train <- subset(sparseWords, split == T)
test <- subset(sparseWords, split == F)

table(test$label)
print(paste("Predicting all messages as non-spam gives an accuracy of: ",
            100*round(table(test$label)[1]/nrow(test), 4), "%"))


sms_classifier<- naiveBayes(label~.,train,laplace = 1)
sms_test_pred<- predict(sms_classifier,newdata = test)
table(test$label, sms_test_pred)




glm.model <- glm(label ~ ., data = train, family = "binomial")
glm.predict <- predict(glm.model, test, type = "response")

glm.ROCR <- prediction(glm.predict, test$label)
print(glm.AUC <- as.numeric(performance(glm.ROCR,"auc")@y.values))

glm.prediction <- prediction(abs(glm.predict), test$label)
glm.performance <- performance(glm.prediction,"tpr","fpr")
plot(glm.performance)
table(test$label, glm.predict > 0.75)


glm.accuracy.table <- as.data.frame(table(test$label, glm.predict > 0.75))
print(paste("logistic model accuracy:",
            100*round(((glm.accuracy.table$Freq[1]+glm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))
svm.model <- svm(label ~ ., data = train, kernel = "linear", cost = 0.1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
table(test$label, svm.predict)


svm.accuracy.table <- as.data.frame(table(test$label, svm.predict))
print(paste("SVM accuracy:",
            100*round(((svm.accuracy.table$Freq[1]+svm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))

tree.model <- rpart(label ~ ., data = train, method = "class", minbucket = 35)

prp(tree.model) 
tree.predict <- predict(tree.model, test, type = "class")
table(test$label, tree.predict)
rpart.accuracy.table <- as.data.frame(table(test$label, tree.predict))
print(paste("rpart (decision tree) accuracy:",
            100*round(((rpart.accuracy.table$Freq[1]+rpart.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))

library(randomForest)
treeRF_spam <- randomForest( label~ ., data = train, ntree=25, proximity = T,importance=TRUE)
treeRF_spam
treeRF_spam_predict<- predict(treeRF_spam, test)
table(test$label, treeRF_spam_predict)
dataimp_spam <- varImpPlot(treeRF_spam, main = "Importance of each variable")

