rm(list=ls())
require(h2o)


#########################################################################

cat("reading data \n")
Sys.time()

people <- read.csv("~/GitHub/predicting-red-hat-business-value/data/people.csv")
activity.train <- read.csv("~/GitHub/predicting-red-hat-business-value/data/act_train.csv")
activity.test <- read.csv("~/GitHub/predicting-red-hat-business-value/data/act_test.csv")

names(people)[2:41] <- paste0("p", names(people[2:41]))
train <- merge(people, activity.train, by=c("people_id"))
test <- merge(people, activity.test, by=c("people_id"))

train$outcome <- as.factor(train$outcome)

#saveRDS(train, "~/GitHub/predicting-red-hat-business-value/data/train.Rdata")
#saveRDS(test, "~/GitHub/predicting-red-hat-business-value/data/test.Rdata")

rm(people, activity.train, activity.test)

#train <- readRDS("~/GitHub/predicting-red-hat-business-value/data/train.Rdata")
#test <- readRDS("~/GitHub/predicting-red-hat-business-value/data/test.Rdata")

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
Sys.time()

h2o.init(nthreads=2, max_mem_size="6G")
h2o.removeAll()

data.hex <- as.h2o(train, destination_frame = "data.hex")

train1 <- data.hex[data.hex$activity_category == "type 1", ]
train2 <- data.hex[data.hex$activity_category != "type 1", ]

####################################################

# GBM

cat("running gbm \n")
Sys.time()

y <- "outcome"

#TODO generate new date variables
disregard1 <- c(y, "people_id", "pdate", "date", "activity_category", "char_10")
x1 <- setdiff(names(train1), disregard1)

fit.gbm1 <- h2o.gbm(y = y, x = x1, distribution = "bernoulli",
                    training_frame = train1, nfolds = 10,
                    ntrees = 100, max_depth = 5, learn_rate = 0.1)

disregard2 <- c(y, "people_id", "pdate", "date",
                   "char_1", "char_2", "char_3", "char_4", "char_5", 
                   "char_6", "char_7", "char_8", "char_9")
x2 <- setdiff(names(train2), disregard2)

fit.gbm2 <- h2o.gbm(y = y, x = x2, distribution = "bernoulli",
                    training_frame = train2, nfolds = 10,
                    ntrees = 100, max_depth = 5, learn_rate = 0.1)


# Predict
cat("generating predictions \n")
Sys.time()

test.hex <- as.h2o(test, destination_frame = "test.hex")
test1 <- test.hex[test.hex$activity_category == "type 1", ]
test2 <- test.hex[test.hex$activity_category != "type 1", ]

predict1 <- h2o.predict(fit.gbm1, test1)[,1]
predict1 <- cbind(as.data.frame(test1$activity_id), as.data.frame(predict1))
predict2 <- h2o.predict(fit.gbm2, test2)[,1]
predict2 <- cbind(as.data.frame(test2$activity_id), as.data.frame(predict2))

prediction <- rbind(predict1, predict2)
names(prediction) <- c("activity_id", "outcome")

write.csv(prediction, "submission.gbm3.csv", row.names = FALSE, quote = FALSE)

cat("finished \n")
Sys.time()