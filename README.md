# Question Generation Summarizer

Major Project by Team 31: Gaurang Tandon, Smriti Agrawal, Salay Jain, Aklesh Mishra

## Introduction

Given a passage of text, we wish to extract sentences out of the passage which
can then be used to generate useful/interesting questions. This can be thought of
as an extractive summarization task on a given passage of text, where the
extracted sentences are likely to generate good questions.

Some examples are:

1. “In this chapter we will study RNNs”: This is NOT a sentence from which we
can generate an acceptable question.
2. “RNNs are useful to solve sequence to sequence problems”: This IS a
sentence from which we can generate an acceptable question.

The task for us is to not generate the questions, but to extract sentences (of the
second type) from a given paragraph from which we can generate the questions.

## Optimization metric

We attempt to optimize the F1 score for our system, here a positive
datatype=sentence from which question can be generated, and negative
datatype=sentence from which question cannot/should not be generated.

## Deliverables

1. Three summarizers/categorizers
    a. Two baseline summarizers
    b. One own summarizer (gold-standard)
2. Annotated dataset of question-generatable sentences (sentence-Boolean
tuples)

## Manually scraped and annotated dataset
### Training dataset

We scraped text from GeeksOfGeeks GATE notes. These notes contain large quantity and decent
quality plain text which is oriented towards interview preparation. We wrote a headless Selenium
script to scrape articles from GFG. The scraped article plain text was cleaned using regex to remove
irrelevant sections. The script then ran NLTK tokenizer to extract sentences from the cleaned article
plain-text.

In total, we extracted 2172 sentences from these articles belonging to several different categories
like ML, DB, CN, Compiles, and OS. These scraped sentences were then annotated manually on the
basis of whether a question generated using them was acceptable or not. On the basis of this
annotation, we obtained the question-worthy percentages.

### Evaluation dataset
We collected data from the NLP and COA evaluations datasets. We chunked it into sentences using
NLTK tokenizer, and manually cleaned its output later. The manual cleaning was necessary because
NLTK sentence tokenizer failed on cases where full stops were present as part of "i.e." or as part of
"Lui et. al." (and other such cases).

## QG System

The QG system we used was based on T5-base (text-to-text transfer transformer) model. We input
to the model a single sentence, and it outputs at least one question-answer pair for that sentence.We did not use the Michael Heilman Java system for this project as it was outputting extremely
strange/unusable questions due to its rule-based nature. The T5 pretrained transformer is a deep-
learning based QG system hence it outputs much better questions, that is, they seem more natural.

## Summarization Baselines

Trained various baseline classifiers like - SVM, Decision Tree, KNN, Random Forest
etc. on evaluation dataset to classify whether a sentence is questionable or not
with fairly good accuracy on question acceptability score. TF-IDF and Word-2-vec
embeddings were used for data representation along with basic data
preprocessing. The datasets include data collected from GFG and official dataset
include some standard book’s limited contents which are manually annotated
based on questionability of the sentence. 

## Summarization Baselines using TF-IDF Embeddings

These embeddings are combined with the TF-IDF scores for each word of each
sentence by multiplying the word embedding with the word’s TF-IDF score for the
sentence, this is done for all the words in the sentence and the result, of multiplying
the word TF-IDF with the word embedding, is accumulated and divided by the
accumulated TF-IDF scores of the words in sentence. The obtained vector is used
as the sentence embedding.

## Gold summarizer – fine-tuned DistilBERT

We fine-tuned DistilBERT on the training dataset. Our reason to do fine-tuning is simple: we want the
model to determine question-acceptability in a very specific setting, which is the technical interview.
Our training data is entirely technical interview related sentences, and hence, we expect the model
performance to improve when using a fine-tuned model as compared to when using a general-purpose
extractive summarizer.The hyperparameters for our model are all standard. Train-val-test split is 60-20-20. Maximum length of
a sentence is 100, and batch size is 128. The model name is distilbert-base-uncased. We run it for 10
epochs with learning rate 5e-5 which auto reduces on loss plateau. Distilbert was chosen because of its
significantly smaller model size and significantly faster training speed as compared to BERT.

## Final comparisons

We compared precision and recall metrics for all four major
techniques:

1. BERT: our gold standard fine-tuned summarizer
2. W2V: best baseline using word2vec embeddings
3. Extractive: general-purpose pre-trained BERT summarizer
4. TF: best baseline using TF-IDF feature vectors as input

The fine-tuned BERT model outperforms the other three
models on both the metrics. These results also serve as our ablation
study. Our fine-tuned model beats the general-purpose BERT
summarizer, which shows that fine-tuning did really help our model to
learn better features.
