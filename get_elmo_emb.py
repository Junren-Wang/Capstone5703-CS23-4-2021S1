import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np


elmo = hub.load('https://tfhub.dev/google/elmo/3')  # load pretrained elmo model from tf hub

# load QA json file
with open('../data/jsons/QA.json', 'r', encoding='utf-8') as f:
    QA = json.load(f)

# read each question, process and save embedding features
for pair in QA:
    ques_id = pair['Question_Id']  # question id is the file name for saving
    ques = pair['Questions']  # get question strings
    embedding = elmo.signatures['default'](tf.constant([ques]))['elmo']  # embed questions, shape = (1, len, 1024)
    embed = tf.squeeze(embedding)  # reduce shape to (len, 1024)
    embed = embed.numpy()  # convert tensorflow tensor to numpy
    np.save('./data/ques_embeddings/elmo/'+str(ques_id)+'.npy', embed)  # save embedding features
    print('file saved:', ques_id)


# validate saved features and calculate max length of all questions
max_len = 0
for pair in QA:
    ques_id = pair['Question_Id']
    emb = np.load('../data/ques_embeddings/elmo/'+str(ques_id)+'.npy')
    length = emb.shape[0]
    if length >= max_len:
        max_len = length
        print('current max length is', max_len)
