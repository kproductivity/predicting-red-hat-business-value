rm(list=ls())
require(h2o)


#########################################################################


people <- read.csv("~/GitHub/predicting-red-hat-business-value/data/people.csv")
activity.train <- read.csv("~/GitHub/predicting-red-hat-business-value/data/act_train.csv")
activity.test <- read.csv("~/GitHub/predicting-red-hat-business-value/data/act_test.csv")

names(people)[2:41] <- paste0("p", names(people[2:41]))
train <- merge(people, activity.train, by=c("people_id"))
test <- merge(people, activity.test, by=c("people_id"))

train$outcome <- as.factor(train$outcome)

saveRDS(train, "~/GitHub/predicting-red-hat-business-value/data/train.Rdata")
saveRDS(test, "~/GitHub/predicting-red-hat-business-value/data/test.Rdata")

rm(people, activity.train, activity.test)

#train <- readRDS("~/GitHub/predicting-red-hat-business-value/data/train.Rdata")
#test <- readRDS("~/GitHub/predicting-red-hat-business-value/data/test.Rdata")

####################################################

h2o.init(nthreads=-1, max_mem_size="6G")
h2o.removeAll()

data.hex <- as.h2o(train, destination_frame = "data.hex")

#TODO Improve split considering type1 vs other types
data.split <- h2o.splitFrame(data=data.hex, 
                             ratios = 0.75, 
                             seed = 123)

train.split <- data.split[[1]]
train1 <- train.split[train.split$activity_category == "type 1", ]
train2 <- train.split[train.split$activity_category != "type 1", ]

test.split <- data.split[[2]]
test1 <- test.split[test.split$activity_category == "type 1", ]
test2 <- test.split[test.split$activity_category != "type 1", ]

rm(data.hex, data.split)

####################################################

# GBM

y <- "outcome"

#TODO generate new date variables
disregard1 <- c(y, "people_id", "pdate", "date", "activity_category", "char10")
x1 <- setdiff(names(train1), disregard1)

fit.gbm1 <- h2o.gbm(y = y, x = x1,
                    distribution="bernoulli",
                    training_frame = train1,
                    validation_frame = test1,
                    ntrees=100, max_depth=4, learn_rate=0.1)

disregard2 <- c("char_1", "char_2", "char_3", "char_4", "char_5", "char_6", "char_7", "char_8", "char_9")
x2 <- setdiff(x1, disregard2)

fit.gbm2 <- h2o.gbm(y = y, x = x1,
                    distribution="bernoulli",
                    training_frame = train2,
                    validation_frame = test2,
                    ntrees=100, max_depth=4, learn_rate=0.1)

# Predict
val.hex <- as.h2o(test, destination_frame = "val.hex")
val1 <- val.hex[val.hex$activity_category == "type 1", ]
val2 <- val.hex[val.hex$activity_category != "type 1", ]

predict1 <- h2o.predict(fit.gbm1, val1)[,1]
predict1 <- cbind(as.data.frame(val1$activity_id), as.data.frame(predict1))
predict2 <- h2o.predict(fit.gbm2, val2)[,1]
predict2 <- cbind(as.data.frame(val2$activity_id), as.data.frame(predict2))

prediction <- rbind(predict1, predict2)

write.csv(prediction, "submission.gbm1.csv", row.names = FALSE)
