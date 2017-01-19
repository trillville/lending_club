library(readr)
library(choroplethr)
library(plyr)
library(dplyr)
library(ggplot2)
library(tm)
#library(stringr)
library(slam)
library(topicmodels)
library(caret)
library(lubridate)
library(wordcloud)
library(tidyr)

# breaks a data frame of tweets into a data frame of words
clusterDescriptions <- function(df, num.topics) {
  df$desc <- gsub("Borrower added on ", "", df$desc) # delete leading comment
  df$desc <- gsub("[[:cntrl:]]", " ", df$desc)  # replace control characters with space
  df <- df[!is.na(df$desc), ]
  
  corpus = VCorpus(DataframeSource(df), 
                   readerControl = list(reader = 
                                          readTabular(mapping = list(content = "desc", id = "id"))))

  dtm <- DocumentTermMatrix(corpus, control = list(stopwords = TRUE,
                                                 removePunctuation = TRUE,
                                                 removeNumbers = TRUE,
                                                 sparse = TRUE,
                                                 tolower = TRUE))
  dtm <- dtm[row_sums(dtm)>0 ,]
  lda.results <- LDA(dtm, k = num.topics, method = "Gibbs")
  topics <- data.frame(topics(lda.results))
  terms <- data.frame(terms(lda.results, 5))
  
  df <- df %>%
    left_join(topics3 %>%
                mutate(id = as.integer(rownames(topics3))))
  return(list(df, terms))
}

