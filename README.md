# Sentiment Analysis using Reccurent Neural Networks and Attention

In this project, there is a comparison of classic RNN model with the task of sentiment analysis, <br/>
and RNN supplemented with Multihead attention.<br/>

In addition, there is an examination of different RNN based architectures and different hyper-parameters<br/>
for different sentiment analysis based tasks, and performance comparison.<br/>
The two datasets used for the task are Stanford Sentiment Treebank for 3 and 5 sentiment classes (SST-5 is 'fine grained' when regarding sentiment labeling).<br/>
The two datasets are considered difficult where SST-5 has State of the art performance of ~50% accuracy, and SST-3 with ~70%.<br/>
This allows a good comparison over perfomance with noticable improvements for the attention based model.<br/>
Another performance comparison was about the confusion of the different models, trying to visualize differences<br/>
between the performances of different models for different classes and with different statistical properties.<br/><br/>

The models results can be viewed in Project.ipynb, and the hyperparameters search results can be viewed in  Hyper_Parameters.ipynb.<br/>
The models code can be found under the project/ folder
