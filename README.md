###  Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.his includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.




## Neural Network Model

Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.his includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.

![1-1](https://github.com/user-attachments/assets/400c61c1-63dc-4287-956f-a14be6e9b9b7)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:THENMOZHI P
### Register Number:212221230116

## importing Required Packages
````
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
````
## Authenticate the Google sheet
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet=gc.open('Untitled spreadsheet').sheet1
data=worksheet.get_all_values()

dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'Input':float})
dataset1=dataset1.astype({'Output':float})
dataset1.head()
X=dataset1[['Input']].values
y=dataset1[['Output']].values
```

## Split the testing and training data
```

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)

```
## Build the Deep learning Model
```
ai_brain=Sequential([
     Dense(units=8,activation='relu'),
     Dense(units=10,activation='relu'),
     Dense(1)
])

ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()


```
## Evaluate the Model
```
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[4]]
X_n1=Scaler.transform(X_n1)
ai_brain.predict(X_n1)

```
`
## Dataset Information

Include screenshot of the dataset

## OUTPUT

## DATA SET

![dl 1](https://github.com/user-attachments/assets/3b919f72-a515-4b3d-8856-a9934d28d977)




### Training Loss Vs Iteration Plot

![training loss](https://github.com/user-attachments/assets/777b6030-8c4b-447e-92de-f69c1b44ebfc)


### Test Data Root Mean Squared Error


![test data](https://github.com/user-attachments/assets/4b967c70-f6a0-4d6b-b55f-1a8239472832)

### New Sample Data Prediction

![new sample](https://github.com/user-attachments/assets/645f0be0-8e2f-46ca-911c-e9bf3d7e4310)


## RESULT

Thus a basic neural network regression model for the given dataset is written and executed successfull
