#TODO work on person history-journey


rm(list=ls())
require(h2o)

#########################################################################

cat("reading data \n")
t <- Sys.time()

people <- read.csv("~/GitHub/predicting-red-hat-business-value/data/people.csv")
activity.train <- read.csv("~/GitHub/predicting-red-hat-business-value/data/act_train.csv")
activity.test <- read.csv("~/GitHub/predicting-red-hat-business-value/data/act_test.csv")

names(people)[2:41] <- paste0("p", names(people[2:41]))
train <- merge(people, activity.train, by=c("people_id"))
test <- merge(people, activity.test, by=c("people_id"))

train$outcome <- as.factor(train$outcome)

rm(people, activity.train, activity.test)


####################################################

#Generate new features

train$timelag <- as.numeric(as.Date(train$date) - as.Date(train$pdate)) #days activity takes place after user starts
test$timelag <- as.numeric(as.Date(test$date) - as.Date(test$pdate))    #days activity takes place after user starts

train$pmonth <- substr(train$pdate, 6, 7)
train$month <- substr(train$date, 6, 7)
train$pyear <- substr(train$pdate, 1, 4)
train$year <- substr(train$date, 1, 4)

test$pmonth <- substr(test$pdate, 6, 7)
test$month <- substr(test$date, 6, 7)
test$pyear <- substr(test$pdate, 1, 4)
test$year <- substr(test$date, 1, 4)

####################################################

cat("engage h2o cluster \n")

h2o.init(nthreads=2, max_mem_size="6G")
h2o.removeAll()

#Naive/frequentist forecast
train.13 <- subset(train, !(train$pchar_2=="type 2" | train$pgroup_1 == "group 27940"))
train.13$pchar_2 <- factor(train.13$pchar_2, levels = c("type 1", "type 3"))

data.hex <- as.h2o(train.13, destination_frame = "data.hex")

####################################################

# GBM

cat("running gbm \n")
Sys.time()

y <- "outcome"

disregard <- c(y, "people_id", "pdate", "date", "activity_category")
x <- setdiff(names(data.hex), disregard)

fit.gbm <- h2o.gbm(y = y, x = x, distribution = "bernoulli",
                   training_frame = data.hex, nfolds = 10,
                   ntrees = 100, max_depth = 5, learn_rate = 0.1)


# Predict
cat("generating predictions \n")

test.13 <- subset(test, !(test$pchar_2=="type 2" | test$pgroup_1 == "group 27940"))
test.2 <- subset(test, (test$pchar_2=="type 2" | test$pgroup_1 == "group 27940"))

test.hex <- as.h2o(test.13, destination_frame = "test.hex")

predict <- h2o.predict(fit.gbm, test.hex)[,1]
predict <- cbind(as.data.frame(test.hex$activity_id), as.data.frame(predict))
names(predict) <- c("activity_id", "outcome")

predict.2 <- cbind(as.data.frame(test.2$activity_id), 0)
names(predict.2) <- c("activity_id", "outcome")

predict <- rbind(predict, predict.2)

write.csv(predict, "submission.gbm7.csv", row.names = FALSE, quote = FALSE)

cat("finished \n")
Sys.time()-t