# MusicTimePeriodClassification

This paper explores the use of sequence modeling to do time period classification for classical music. We explore the use of a long short-term memory (LSTM) network to model music. We do this in two steps. (1) In the first step, we design an upstream task that takes in input sequences of music notes. (2) Secondly, we use this model to classify music sequences into musical time periods. We derive this set of mappings through finding lists of time periods and the respective pieces that correspond to the period. We show that by first training a language model on a set of unlabeled data and then fine tuning on a smaller set of labeled data, we are able to improve performance. Through this modeling, we hope to learn what factors and features are the most useful in delineating between time periods. Furthermore, we would like to explore the model that we build through probing to learn more about musical time periods.

Please follow this link to read the work: https://www.cs.utexas.edu/~ssingh/Modeling_Music_Using_Transfer_Learning.pdf




