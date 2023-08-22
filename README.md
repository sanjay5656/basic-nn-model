# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/sanjay5656/basic-nn-model/assets/115128955/0dcd72d3-2b54-4b36-b972-4260a85f1087)

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

```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df

X = df[['INPUT']].values
Y = df[['OUTPUT']].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=2200)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1 = [[20]]
X_n1_1 value = Scaler.transform(X_n1)
model.predict(X_n1_1 value)

```

## Dataset Information

![Screenshot 2023-08-22 221819](https://github.com/sanjay5656/basic-nn-model/assets/115128955/37c988ec-b7aa-4aed-be79-1550030988d5)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/sanjay5656/basic-nn-model/assets/115128955/9edf092e-38ca-4516-93b1-1ff701236bc3)

### Test Data Root Mean Squared Error

![image](https://github.com/sanjay5656/basic-nn-model/assets/115128955/1730a82a-9a98-49bd-bc4a-f7a0fd3cfb1b)

### New Sample Data Prediction

![image](https://github.com/sanjay5656/basic-nn-model/assets/115128955/86a01034-446d-4996-9589-07b14c8d1d6e)

## RESULT
Thus the neural network regression model for the given dataset is executed successfully.
