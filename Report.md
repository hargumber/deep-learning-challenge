# Module 21 - Challenge - Report

### Overview
This analysis is being written as part of Module 21 Challenge which is essentially a challenge on deep learning.
The purpose of this anlaysis is to come up with machine learning model using neural network that can help non profile foundation Alphabet Soup. The model has to use the features provided in the dataset to predict whether the applications will be successful if they are funded by Alphabet Soup.

### Results:
- Target variable for the the model  
The target variable for this model is IS_SUCCESSFUL, the variable indicates if the money was used effectively or not.

- Features for the model  
For part 1 of this challenge, I used the provided features as listed below:  
APPLICATION_TYPE, CLASSIFICATION, USE_CASE, AFFILICATION, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.

- Variables removed from the dataset  
For Part 1, per the ask, the following variables were removed from the dataset i.e. EIN and NAME.


### Compiling, Training and Evaluating the Model
#### Part 1
For part 1, I used the following.
2 hidden layers, the first layer had 80 neurons and the second layer had 30 neurons.
I used the relu activation function for the 2 hidden layer and sigmoid action function for the output later. Total number of epochs used was 100 in this case.
I was able to achieve an accuracy of 72.78 % and the model is saved in the following file AlphabetSoupCharity.h5

#### Part 2
For part 2, in an effort to improve the model, I used the keras-tuner library.
I used a lot of different combinations as described below.

##### Trial 1
Number of neuros to try: 1,3,5,7,9
Have the tuner choose between any of the three action fuunction i.e. relu, tanh, sigmoid
For the number of hidden layer, I had provided a range from 1-6
Epochs 20
For the training and tuning ended, the best accuracy i could get with this round was 73.43%
Number of input dimensions (features) i.e. scaled data was 43 in this case.

Below are the results for top 3 models along with their accuracy
###### Hyper Params for top 3 best models
{'activation': 'sigmoid', 'first_units': 3, 'num_layers': 5, 'units_0': 3, 'units_1': 3, 'units_2': 3, 'units_3': 9, 'units_4': 7, 'units_5': 7, 'tuner/epochs': 17, 'tuner/initial_epoch': 6, 'tuner/bracket': 2, 'tuner/round': 1, 'tuner/trial_id': '0144'}
{'activation': 'sigmoid', 'first_units': 9, 'num_layers': 1, 'units_0': 1, 'units_1': 9, 'units_2': 3, 'units_3': 7, 'units_4': 5, 'units_5': 3, 'tuner/epochs': 17, 'tuner/initial_epoch': 6, 'tuner/bracket': 2, 'tuner/round': 1, 'tuner/trial_id': '0055'}
{'activation': 'sigmoid', 'first_units': 9, 'num_layers': 2, 'units_0': 5, 'units_1': 5, 'units_2': 1, 'units_3': 5, 'units_4': 3, 'units_5': 5, 'tuner/epochs': 17, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}

###### Loss and accuracy results
268/268 - 1s - 2ms/step - accuracy: 0.7343 - loss: 0.5752
Loss: 0.5751794576644897, Accuracy: 0.7343440055847168
268/268 - 1s - 2ms/step - accuracy: 0.7336 - loss: 0.5750
Loss: 0.5749614238739014, Accuracy: 0.7336443066596985
268/268 - 1s - 2ms/step - accuracy: 0.7336 - loss: 0.5733
Loss: 0.5733080506324768, Accuracy: 0.7336443066596985

##### Trial 2
For trail 2, all the search params were same as above in trail 1 but I only changed the epochs and this time, I used 50.

Below are the results for top 3 models along with their accuracy
###### Hyper Params for top 3 best models
Results:
{'activation': 'relu', 'first_units': 7, 'num_layers': 2, 'units_0': 1, 'units_1': 9, 'units_2': 1, 'units_3': 3, 'units_4': 1, 'units_5': 7, 'tuner/epochs': 17, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}
{'activation': 'sigmoid', 'first_units': 1, 'num_layers': 1, 'units_0': 5, 'units_1': 7, 'units_2': 3, 'units_3': 9, 'units_4': 9, 'units_5': 9, 'tuner/epochs': 17, 'tuner/initial_epoch': 6, 'tuner/bracket': 2, 'tuner/round': 1, 'tuner/trial_id': '0148'}
{'activation': 'sigmoid', 'first_units': 7, 'num_layers': 6, 'units_0': 5, 'units_1': 9, 'units_2': 9, 'units_3': 7, 'units_4': 3, 'units_5': 7, 'tuner/epochs': 50, 'tuner/initial_epoch': 17, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0067'}

###### Loss and accuracy results
268/268 - 1s - 2ms/step - accuracy: 0.7338 - loss: 0.5569
Loss: 0.556861162185669, Accuracy: 0.7337609529495239
268/268 - 1s - 2ms/step - accuracy: 0.7334 - loss: 0.5731
Loss: 0.5731139779090881, Accuracy: 0.7334110736846924
268/268 - 1s - 3ms/step - accuracy: 0.7333 - loss: 0.5752
Loss: 0.5751731395721436, Accuracy: 0.7332944869995117

##### Trial 3
For trial 3, i decided to drop some more columns that i feel are not contributing much to the model.
I dropped APPLICATION_TYPE and CLASSIFICATION.
I again used keras tuner to search for best hyper parameters and used the ranges similar to Trial 1 but the only thing changed was input dimensions as the number of features reduced and for this one the number of input dimensions was 28 and epochs 20.

Here are the results for this Trail.
###### Hyper Params for top 3 best models
Results:
{'activation': 'tanh', 'first_units': 4, 'num_layers': 4, 'units_0': 1, 'units_1': 19, 'units_2': 7, 'units_3': 16, 'units_4': 19, 'units_5': 1, 'units_6': 7, 'units_7': 16, 'tuner/epochs': 50, 'tuner/initial_epoch': 17, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0071'}
{'activation': 'sigmoid', 'first_units': 16, 'num_layers': 2, 'units_0': 19, 'units_1': 16, 'units_2': 19, 'units_3': 4, 'units_4': 13, 'units_5': 16, 'units_6': 16, 'units_7': 13, 'tuner/epochs': 17, 'tuner/initial_epoch': 6, 'tuner/bracket': 3, 'tuner/round': 2, 'tuner/trial_id': '0039'}
{'activation': 'sigmoid', 'first_units': 10, 'num_layers': 5, 'units_0': 10, 'units_1': 13, 'units_2': 7, 'units_3': 7, 'units_4': 1, 'units_5': 19, 'units_6': 13, 'units_7': 13, 'tuner/epochs': 50, 'tuner/initial_epoch': 17, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0080'}

###### Loss and accuracy results
268/268 - 1s - 3ms/step - accuracy: 0.7030 - loss: 0.5993
Loss: 0.5993413925170898, Accuracy: 0.7029737830162048
268/268 - 1s - 3ms/step - accuracy: 0.7029 - loss: 0.5987
Loss: 0.5987154245376587, Accuracy: 0.7028571367263794
268/268 - 1s - 3ms/step - accuracy: 0.7029 - loss: 0.6033
Loss: 0.6033048629760742, Accuracy: 0.7028571367263794

### Summary
As described above, I tried to improve the model by using a number of different combinations and also use keras-tuner to search for the best hyper parameters, I was not able to get to 75% accuracy as i think i used very less number of neurons especially from what i used in part 1.

Based on my results, I cannot provide a direct recommendation on which model to use in order to solve this problem but I would defintely recommend using the model from Part 1 which is also saved as AlphabetSoupCharity.h5 as that has higher accuracy and would also like ot train another model with more number of neurons in each layer than what I used in Part 1.

Based on my results, model from Part 1 with accuracy of 72.78% is the best model to use for this problem.