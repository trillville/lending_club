path <- "/Users/tillmanelser/Documents/R/Earnest"
setwd(path)

source("functions_libs.R")

# Load and pre-process data ---------------------------------------------------------------

approved.list <- list.files(path = paste(path, "/approved", sep = ""), full.names = TRUE)
approved.loans <- rbind.fill(lapply(approved.list, read_csv, skip = 1, 
                                    col_types = list(
                                      issue_d = col_date(format = "%b-%Y"),
                                      funded_amnt_inv = col_double(),
                                      earliest_cr_line = col_date(format = "%b-%Y"),
                                      last_pymnt_d = col_date(format = "%b-%Y"),
                                      last_credit_pull_d = col_date(format = "%b-%Y"),
                                      annual_inc_joint = col_double()
                                    )))

# this column would probably better represented by integers
approved.loans$emp_length_int <- ifelse(approved.loans$emp_length == "10+ years", "15 years",
                                        ifelse(approved.loans$emp_length == "< 1 year", "0.5 years",
                                               ifelse(approved.loans$emp_length == "1 year", "1 years", 
                                                      ifelse(approved.loans$emp_length == "n/a", 0, approved.loans$emp_length))))

# clean up some strings
approved.loans <- approved.loans %>%
  filter(!is.na(id)) %>%
  mutate(int_rate = as.numeric(gsub("%", "", int_rate)),
         revol_util = as.numeric(gsub("%", "", revol_util)),
         term = as.numeric(gsub(" months", "", term)),
         emp_length_int = as.numeric(gsub(" years", "", emp_length_int)) * 12) # convert to months

approved.loans[, c(55, 59:ncol(approved.loans))] <- apply(approved.loans[, c(55, 59:ncol(approved.loans))], 2, as.numeric)

# Visualizations ----------------------------------------------------------

plot1.data <- approved.loans %>% 
  group_by(issue_d, grade) %>% 
  summarise(Funded_Amount = sum(funded_amnt))

# Gross loan amount over time - group by grade
plot1 <- ggplot(data = plot1.data) +
  geom_area(aes(x = issue_d, y = Funded_Amount/1000000, fill = grade)) +
  xlab("Issue Date") +
  ylab("Funded Amount ($mln)") +
  scale_x_date(limits = c(as.Date("2008-01-01"), NA)) +
  ggtitle("Funded Loans: 2008-2016")

plot2.data <- approved.loans %>%
  group_by(issue_d, purpose) %>%
  summarise(Funded_Amount = sum(funded_amnt))

# Distribution of loan purpose over time
plot2 <- ggplot(data = plot2.data) +
  geom_area(aes(x = issue_d, y = Funded_Amount, fill = purpose), position = "fill") +
  xlab("Issue Date") +
  ylab("Proportion of Funded Loans") +
  scale_x_date(limits = c(as.Date("2008-01-01"), NA)) +
  ggtitle("Distribution of Loan Purpose")


plot3 <- ggplot(data = approved.loans) +
  geom_smooth(aes(x = issue_d, y = int_rate, col = grade)) +
  xlab("Issue Date") +
  ylab("Average interest Rate") +
  scale_x_date(limits = c(as.Date("2008-01-01"), NA)) +
  facet_wrap(~purpose)

plot3b <- ggplot(data = approved.loans[approved.loans$grade %in% c("A", "B", "C"), ]) +
  geom_smooth(aes(x = issue_d, y = int_rate, col = purpose), se = FALSE) +
  xlab("Issue Date") +
  ylab("Average interest Rate") +
  scale_x_date(limits = c(as.Date("2008-01-01"), NA)) +
  facet_wrap(~purpose)

#distributions of grades/purposes
plot3c <- ggplot(data = approved.loans) +
  geom_density(aes(x = int_rate, fill = purpose), alpha = 0.5) +
  facet_wrap(~purpose)

plot3d <- ggplot(data = approved.loans) +
  geom_bar(aes(x = grade, fill = purpose), stat = "count", position = "fill") +
  ylab("Proportion of Loan Grade") 

# show the correlation over time between different loan purposes
plot3e.data <- approved.loans %>%
  group_by(purpose, issue_d) %>%
  summarize(mean_rate = mean(int_rate, na.rm = TRUE)) %>%
  spread(purpose, mean_rate) %>%
  select(-educational, -wedding)

plot3e.data <- na.omit(plot3e.data)
plot3e <- corrplot.mixed(cor(plot3e.data[, -1]))

# densities of interest rates by grade
plot4 <- ggplot(data = approved.loans[approved.loans$issue_d == as.Date("2016-01-01"), ], aes(x = grade, y = int_rate, fill = grade)) +
  geom_violin(alpha=0.5) +
  coord_flip()

key <- data.frame(state.name, state.abb)

# map of average interest rate by state
plot5.data <- approved.loans %>% 
  filter(year(issue_d) >= 2014) %>%
  group_by(addr_state) %>%
  summarize(rate = mean(int_rate, na.rm = TRUE), count = n()) %>%
  left_join(key, by = c("addr_state" = "state.abb")) %>%
  rename(region = state.name, value = rate) %>%
  mutate(region = tolower(region))

plot5.data$region[plot5.data$addr_state == "DC"] = "district of columbia"

plot5 <- state_choropleth(plot5.data, title = "Average Interest Rate by State: 2014-2016")

# Title Text Analysis -----------------------------------------------

# using a subset of the data to speed things up from here on
set.seed(100)
eda.subset <- createDataPartition(approved.loans$loan_status, p = .2, 
                                  list = FALSE, 
                                  times = 1)
approved.small <- approved.loans[eda.subset, ]

approved.small$title <- gsub("[[:cntrl:]]", " ", approved.small$title)  # replace control characters with space

title.corpus = VCorpus(DataframeSource(approved.small[!is.na(approved.small$title), ]), 
                       readerControl = list(reader = 
                                              readTabular(mapping = list(content = "title", id = "id"))))

title.dtm <- DocumentTermMatrix(title.corpus, control = list(stopwords = TRUE,
                                                             removePunctuation = TRUE,
                                                             removeNumbers = TRUE,
                                                             sparse = TRUE,
                                                             tolower = TRUE))
title.dtm <- title.dtm[row_sums(title.dtm)>0 ,]

lda.title <- LDA(title.dtm, k = 5, method = "Gibbs")
title.topics <- data.frame(topics(lda.title))
title.terms <- data.frame(terms(lda.title, 6))

approved.small <- approved.small %>%
  left_join(title.topics %>%
              mutate(id = as.integer(rownames(title.topics))))


# Modeling Fun ------------------------------------------------------------

# quick check for columns that we may want to drop
colSums(approved.loans != 0, na.rm = TRUE)/nrow(approved.loans)

# Looks like many of the columns aren't populated until December 2015 - let's just use the post December 2015 subset

model1.data <- approved.loans %>%
  select(-id, -member_id, -funded_amnt, -funded_amnt_inv, -url, -desc, -title, -zip_code, -total_acc, -out_prncp, -out_prncp_inv, 
         -total_pymnt, -total_pymnt_inv, -total_rec_prncp, -total_rec_int, -total_rec_late_fee, -recoveries, -collection_recovery_fee,
         -last_pymnt_d, -last_pymnt_amnt, -next_pymnt_d, -last_credit_pull_d, -last_fico_range_high, -last_fico_range_low, -annual_inc_joint,
         -dti_joint, -verification_status_joint, -tot_cur_bal, -installment, -loan_status, -emp_title, -grade, -sub_grade) %>%
  filter(issue_d >= as.Date("2015-01-01"))

# should missing values be imputed, set to zero, or dropped all together?
colSums(!is.na(model1.data))/nrow(model1.data)  
na.to.zero <- c("delinq_2yrs", "inq_last_6mths", "inq_last_12m", "pub_rec", "collections_12_mths_ex_med", "acc_now_delinq",
                "tot_coll_amt", "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m", "total_bal_il", "il_util", 
                "open_rv_12m", "open_rv_24m", "max_bal_bc", "inq_fi", "total_cu_tl", "all_util", "num_tl_120dpd_2m") # zero seems plausible for missing vals

na.drops <- c("inq_last_6mths", "num_rev_accts", "bc_open_to_buy", "mths_since_recent_bc", "percent_bc_gt_75",
              "revol_util", "bc_util") # very few missing observations (less than 0.1%) so just drop these

# These are all months since last event. Seems likely that in this case NA refers to the event never happening.
# Going to set all NA values to the max observed. Could potentially improve performance by instead switching
# to buckets (i.e. recent delinq; medium recent delinq; distant delinq; no delinq)
switch.to.buckets <- c("mths_since_last_record", "mths_since_last_delinq", "mths_since_last_major_derog", "mths_since_rcnt_il",
                       "mo_sin_old_il_acct", "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq") 

one.val.factors <- c("pymnt_plan", "policy_code") # factor variables with 1 unique value

model1.data[, na.to.zero][is.na(model1.data[, na.to.zero])] <- 0

model1.data <- model1.data[complete.cases(model1.data[, na.drops]), ]

model1.data[, switch.to.buckets] <- apply(model1.data[, switch.to.buckets], 2, 
                                          function(x) ifelse(is.na(x), max(x, na.rm = TRUE), x))

model1.data <- model1.data[, -which(names(model1.data) %in% one.val.factors)]

colSums(!is.na(model1.data))/nrow(model1.data)  

# Impute remaining missing values (<5% of the obs missing for a few remaining columns)
pp <- preProcess(model1.data, method = "bagImpute")
model1.data <- predict(pp, model1.data)

ohe <- dummyVars(~., data = model1.data)
model1.mat <- predict(ohe, model1.data)
colnames(model1.mat) <- make.names(colnames(model1.mat))

library(Boruta)
# use 5% of the data to speed it up
model1.subset <- createDataPartition(model1.data$int_rate, p = .015, list = FALSE, times = 1)
model1.small <- model1.mat[model1.subset, ]

feature.analysis <- Boruta(model1.x, model1.y, doTrace = 2)
results <- feature.analysis$finalDecision
good.features <- names(results[as.character(results) %in% c("Confirmed", "Tentative")])
bad.features <- names(results[as.character(results) %in% c("Rejected")])

model1.x <- model1.mat[, good.features]
model1.y <- model1.mat[, "int_rate"]

small.x <- model1.small[, good.features]
small.y <- model1.small[, "int_rate"]

library(randomForest)
rf.imp <- randomForest(x = small.x, y = small.y, ntree = 500, importance = TRUE)
varImpPlot(rf.imp)
best.features <- row.names(rf.imp$importance)[1:10]

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = small.x[, best.features], 
            y = small.y, 
            plot = "scatter",
            type = c("p", "smooth"),
            span = .5)

ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 5, search = "random",
                     adaptive = list(min = 5, alpha = 0.05, 
                                     method = "gls", complete = TRUE))

library(doMC)
registerDoMC(cores = 6)

set.seed(100)
lm1 <- train(x = small.x, y = small.y,
             method = "lm",
             trControl = ctrl)

set.seed(100)
rf1 <- train(x = small.x, y = small.y,
             method = "rf",
             tuneLength = 5,
             trControl = ctrl) 

set.seed(100)
a <- Sys.time()
xgb1 <- train(x = small.x, y = small.y,
              method = "xgbTree",
              tuneLength = 50,
              trControl = ctrl)
Sys.time() - a

set.seed(100)
a <- Sys.time()
rpart1 <- train(x = small.x, y = small.y,
                method = "rpart",
                trControl = ctrl,
                tuneLength = 10) 

Sys.time() - a

set.seed(100)
lasso1 <- train(x = small.x, y = small.y,
                method = "glmnet",
                tuneLength = 25,
                trControl = ctrl,
                preProcess = c("center", "scale"))

# set.seed(100)
# gam1 <- train(x = small.x, y = small.y,
#               method = "gam",
#               trControl = ctrl) 

set.seed(100)
pls1 <- train(x = small.x, y = small.y,
              method = "pls",
              tuneLength = 10,
              trControl = ctrl,
              preProcess = c("center", "scale") )

set.seed(100)
knn1 <- train(x = small.x, y = small.y,
              method = "kknn",
              tuneLength = 10,
              trControl = ctrl,
              preProcess = c("center", "scale") )

all.resamples <- resamples(list("Linear Regression" = lm1,
                                "Partial Least Squares" = pls1,
                                "Lasso" = lasso1,
                                "Regression Tree" = rpart1,
                                "KNN" = knn1,
                                "Random Forest" = rf1,
                                "XGBoost (tree)" = xgb1))

parallel.plot1 <- parallelplot(all.resamples, metric = "Rsquared")
parallel.plot2 <- parallelplot(all.resamples)

# Description Text Analysis -----------------------------------------------

# approved.small$desc <- gsub("Borrower added on ", "", approved.small$desc) # delete leading comment
# approved.small$desc <- gsub("[[:cntrl:]]", " ", approved.small$desc)  # replace control characters with space
# 
# corpus = VCorpus(DataframeSource(approved.small[!is.na(approved.small$desc), ]), 
#                  readerControl = list(reader = 
#                                         readTabular(mapping = list(content = "desc", id = "id"))))
# 
# dtm <- DocumentTermMatrix(corpus, control = list(stopwords = TRUE,
#                                                  removePunctuation = TRUE,
#                                                  removeNumbers = TRUE,
#                                                  sparse = TRUE,
#                                                  tolower = TRUE))
# dtm <- dtm[row_sums(dtm)>0 ,]
# 
# lda.results <- LDA(dtm, k = 10, method = "Gibbs")
# topics <- data.frame(topics(lda.results))
# terms <- data.frame(terms(lda.results, 5))
# 
# approved.small <- approved.small %>%
#   left_join(topics %>%
#               mutate(id = as.integer(rownames(topics))))
