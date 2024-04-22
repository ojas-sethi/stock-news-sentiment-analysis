import tensorflow as tf
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime # For Datetime Functions
import pathlib # handling files and paths on your operating system
import io # dealing with various types of I/O
import os 
import re # for Regular Expressions
import string
import time
from numpy import random
import gensim.downloader as api # to download pre-trained model datasets and word embeddings from Gensim's repository
from PIL import Image # manipulating images, resizing, cropping, adding text
import tensorflow.keras as keras
import tensorflow_datasets as tfds # Tf Datasets
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense,Flatten,InputLayer,BatchNormalization,Dropout,Input,LayerNormalization)
from tensorflow.keras.losses import (BinaryCrossentropy,CategoricalCrossentropy,SparseCategoricalCrossentropy)
from tensorflow.keras.metrics import (Accuracy,TopKCategoricalAccuracy,CategoricalAccuracy, SparseCategoricalAccuracy)
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
from transformers import (BertTokenizerFast,TFBertTokenizer,BertTokenizer,RobertaTokenizerFast,
                          DataCollatorWithPadding,TFRobertaForSequenceClassification,TFBertForSequenceClassification,
                          TFBertModel,create_optimizer)

BATCH_SIZE=8

dataset_id='imdb'
dataset = load_dataset(dataset_id) # load_dataset is a function of datasets library of HuggingFace
# Train, test, and validation datasets are dictionaries containing the dataset splits

model_id="bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)

def preprocess_function(examples):
  return tokenizer(examples["text"],padding=True,truncation=True,)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
)

tf_val_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

def swap_positions(dataset):
  return {'input_ids':dataset['input_ids'],
          'token_type_ids':dataset['token_type_ids'],
          'attention_mask':dataset['attention_mask'],},dataset['label']

tf_train_dataset=tf_train_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)
tf_val_dataset=tf_val_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)

model=TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=1)

num_epochs = 3
batches_per_epoch = len(tokenized_dataset["train"]) // BATCH_SIZE
total_train_steps = int(batches_per_epoch * num_epochs)

optimizer, schedule = create_optimizer(init_lr=2e-5,num_warmup_steps=0, num_train_steps=total_train_steps)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy'],)
    #run_eagerly=True)

history=model.fit(
    tf_train_dataset.take(1000),
    validation_data=tf_val_dataset,
    epochs=3)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

inputs = tokenizer(["this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes! ",
"very good start, but movie started becoming uninteresting at some point though initially i thought it would have been much more fun. There was too much background noise, but later on towards the middle of the movie, my favorite character got in and he did a great job, so over "], padding=True,return_tensors="tf")

logits = model(**inputs).logits
print(logits)

probabilities = tf.nn.softmax(logits, axis=-1)
print(probabilities)