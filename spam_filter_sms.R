library(tm)
library(wordcloud)
library(e1071)
library(gmodels)
library(caret)
sms_raw<- read.csv('sms.csv',sep='\t',stringsAsFactor=F)
sms_raw$class<- sms_raw[,1]
sms_raw$text <- sms_raw[,2]

sms_raw$class<- as.factor(sms_raw$class)
sms_corpus <- Corpus(VectorSource(sms_raw$text))

#cleaning the sms 
corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean,removeNumbers)
corpus_clean <- tm_map(corpus_clean,removeWords,stopwords())
corpus_clean <- tm_map(corpus_clean,removePunctuation)
corpus_clean <- tm_map(corpus_clean,stripWhitespace)
#Document Term Matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)

# train and test data
sms_raw_train <- sms_raw[1:3000,]
sms_raw_test <- sms_raw[3001:3183,]
sms_dtm_train <- sms_dtm[1:3000,]
sms_dtm_test <- sms_dtm[3000,3183,]
#Corpus Training and Test Data sets
sms_corpus_train <- corpus_clean[1:3000]
sms_corpus_test <- corpus_clean[3001:3183]
prop.table(table(sms_raw_train$class))
prop.table(table(sms_raw_test$class))
#Visualization using world clouds
png('wordcloud_spam_1.png')
wordcloud(sms_corpus_train,min.freq=40,random.order=F)
dev.off()
spam <- subset(sms_raw_train,class='spam')
ham <- subset(sms_raw_train,class='ham')
png('wordcloud_with_spam.png')
wordcloud(spam$text,max.words = 40,scale = c(3,0.5))
dev.off()
png('wordcloud_with_ham.png')
wordcloud(ham$text,max.words=40,scale=c(3,0.5))
dev.off()
# indicating features for frequent words
findFreqTerms(sms_dtm_train,5)
sms_dict <- (findFreqTerms(sms_dtm_train,5))
sms_train <- DocumentTermMatrix(sms_corpus_train,list(dictionary=sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test,list(dictionary = sms_dict))
convert_counts <- function(x)
{
  x <- ifelse(x>0,1,0)
  x <- factor(x,levels=c(0,1),labels = c('No','Yes'))
  x
}
sms_train <- apply(sms_train,MARGIN=2,convert_counts)
sms_test <- apply(sms_test,MARGIN=2,convert_counts)
#training
sms_classifier <- naiveBayes(sms_train,sms_raw_train$class)
sms_test_pred <- predict(sms_classifier,sms_test)
png('CrossTable_first.png')
CrossTable(sms_test_pred,sms_raw_test$class,prop.chsiq=F,prop.t=F,dnn=c('predicted','actual'))
dev.off()
sensitivity(sms_test_pred,sms_raw_test$class,positive='spam')
specificity(sms_test_pred,sms_raw_test$class,negative='ham')
#Improving Model Performance
#We need to assign value to the Laplace Estimator
sms_classifier2 <- naiveBayes(sms_train,sms_raw_train$class,laplace = 0.01)
sms_test_pred2 <- predict(sms_classifier2,sms_test)
png('CrossTable_improved.png')
CrossTable(sms_test_pred2,sms_raw_test$class,prop.chisq=F,prop.t=F,dnn=c('predicted','actual'))
dev.off()
sensitivity(sms_test_pred2,sms_raw_test$class,positive='spam')
specificity(sms_test_pred2,sms_raw_test$class,negative='ham')




