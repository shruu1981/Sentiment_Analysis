#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS= set(stopwords.words("english"))


# In[4]:


get_ipython().system('pip install xgboost')


# In[6]:


get_ipython().system('pip install wordcloud')


# In[7]:


from wordcloud import WordCloud


# In[9]:


#Loading data

df = pd.read_csv(r"Downloads/archive (3)/amazon_alexa.tsv", delimiter = '\t', quoting = 3)
df.head(10)


# In[11]:


df.shape


# In[14]:


print(df.describe())
print(df.info())


# In[15]:


df.columns


# In[16]:


df.isnull().sum()


# In[22]:


df.dtypes


# In[17]:


df.dropna(inplace=True) #dropping null values


# *rating, feedback and length are integer values*
# *date, variation and verified_reviews are string values*

# In[19]:


df['length'] = df['verified_reviews'].apply(len)


# In[21]:


print(f"'verified_reviews' column value: {df.iloc[10]['verified_reviews']}") #Original value
print(f"Length of review : {len(df.iloc[10]['verified_reviews'])}") #Length of review using len()
print(f"'length' column value : {df.iloc[10]['length']}") #Value of the column 'length'


# **Analyzing 'rating' column**
# 
# *This column refers to the rating of the variation given by the user*

# In[23]:


df['rating'].value_counts()


# In[27]:


df['rating'].value_counts().plot.bar(color = 'lightblue')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()
#ploting bar plot


# In[34]:


#now lets plot pie chat

fig = plt.figure(figsize=(7,7))

tags = df['rating'].value_counts()/df.shape[0]

explode=(0.1,0.1,0.1,0.1,0.1)

tags.plot(kind='pie', autopct="%1.1f%%",explode=explode, label='Percentage wise distrubution of rating')


# **Analyzing 'variation' column**
# 
# *This column refers to the variation or type of Amazon Alexa product. Example - Black Dot, Charcoal Fabric etc.*

# In[36]:


df['variation'].value_counts()


# In[38]:


#Bar graph to visualize the total counts of each variation

df['variation'].value_counts().plot.bar(color = 'lightgreen')
plt.title('Variation distribution count')
plt.xlabel('Variation')
plt.ylabel('Count')
plt.show()


# In[41]:


df['variation'].value_counts()/df.shape[0]*100,2
#percentage distrbution


# In[43]:


df.groupby('variation')['rating'].mean()


# In[45]:


df.groupby('variation')['rating'].mean().sort_values().plot.bar(color = 'darkgreen', figsize=(11, 6))
plt.title("Mean rating according to variation")
plt.xlabel('Variation')
plt.ylabel('Mean rating')
plt.show()


# **Analyzing 'feedback' column**
# 
# *This column refers to the feedback of the verified review*

# In[46]:


df['feedback'].value_counts()


# In[47]:


#Extracting the 'verified_reviews' value for one record with feedback = 0

review_0 = df[df['feedback'] == 0].iloc[1]['verified_reviews']
print(review_0)


# In[48]:


#Extracting the 'verified_reviews' value for one record with feedback = 1

review_1 = df[df['feedback'] == 1].iloc[1]['verified_reviews']
print(review_1)


# **From the above 2 examples we can see that feedback 0 is negative review and 1 is positive review**
# 
# *Let's plot the feedback value count in a bar graph*

# In[50]:


#Bar graph to visualize the total counts of each feedback

df['feedback'].value_counts().plot.bar(color = 'pink')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()


# In[52]:


df['feedback'].value_counts()/df.shape[0]*100,2


# **Feedback distribution**
# 
# *91.87% reviews are positive
# 
# 8.13% reviews are negative*

# In[56]:


fig = plt.figure(figsize=(7,7))
wp = {'linewidth':1, "edgecolor":'black'}
tags = df['feedback'].value_counts()/df.shape[0]
explode=(0.1,0.1)
tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of feedback')


# **Analyzing 'verified_reviews' column**
# 
# *This column contains the textual review given by the user for a variation for the product.*

# In[59]:


import seaborn as sns
sns.histplot(df['length'],color='yellow').set(title='Distribution of length of review ')


# In[60]:


sns.histplot(df[df['feedback']==0]['length'],color='brown').set(title='Distribution of length of review if feedback = 0')


# In[67]:


cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(df.verified_reviews)


# In[66]:


# Combine all reviews
reviews = " ".join([review for review in df['verified_reviews']])
                        
# Initialize wordcloud object
wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[71]:


# find the unique words in each feedback category

# Combine all reviews for each feedback category and splitting them into individual words
neg_reviews = " ".join([review for review in df[df['feedback'] == 0]['verified_reviews']])
neg_reviews = neg_reviews.lower().split()

pos_reviews = " ".join([review for review in df[df['feedback'] == 1]['verified_reviews']])
pos_reviews = pos_reviews.lower().split()

#Finding words from reviews which are present in that feedback category only
unique_negative = [x for x in neg_reviews if x not in pos_reviews]
unique_negative = " ".join(unique_negative)

unique_positive = [x for x in pos_reviews if x not in neg_reviews]
unique_positive = " ".join(unique_positive)
wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('Wordcloud for negative reviews', fontsize=10)
plt.axis('off')
plt.show()


# **Negative words can be seen in the above word cloud - garbage, pointless, poor, horrible, repair etc**

# In[72]:


wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('Wordcloud for positive reviews', fontsize=10)
plt.axis('off')
plt.show()


# **Positive words can be seen in the above word cloud - good, enjoying, amazing, best, great etc**

# **Preprocessing and Modelling**
# 
# To build the corpus from the 'verified_reviews' we perform the following -
# 
# Replace any non alphabet characters with a space
# Covert to lower case and split into words
# Iterate over the individual words and if it is not a stopword then add the stemmed form of the word to the corpus

# In[74]:


corpus = []
stemmer = PorterStemmer()
for i in range(0, df.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', df.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)


# **Using Count Vectorizer to create bag of words**

# In[78]:


cv = CountVectorizer(max_features = 2500)
import pickle
#Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = df['feedback'].values


# In[84]:


#Saving the Count Vectorizer
import os
os.makedirs('Models', exist_ok=True)
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")


# In[81]:


print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")


# In[82]:


scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)


# In[85]:


#Saving the scaler model
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))


# In[86]:


#Fitting scaled X_train and y_train on Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)


# In[87]:


print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))


# In[88]:


#Predicting on the test set
y_preds = model_rf.predict(X_test_scl)


# In[89]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)


# In[90]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
cm_display.plot()
plt.show()


# 
# K fold cross-validation

# In[91]:


accuracies = cross_val_score(estimator = model_rf, X = X_train_scl, y = y_train, cv = 10)

print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())


# In[92]:


params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}


# In[93]:


cv_object = StratifiedKFold(n_splits = 2)

grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_scl, y_train.ravel())


# In[94]:


#Getting the best parameters from the grid search


print("Best Parameter Combination : {}".format(grid_search.best_params_))


# 
# XgBoost

# In[95]:


model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)


# In[96]:


#Accuracy of the model on training and testing data
 
print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))


# In[97]:


y_preds = model_xgb.predict(X_test)


# In[98]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)


# In[99]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
cm_display.plot()
plt.show()


# In[100]:


#Saving the XGBoost classifier
pickle.dump(model_xgb, open('Models/model_xgb.pkl', 'wb'))


# DecisionTreeClassifier

# In[101]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)


# In[102]:


#Accuracy of the model on training and testing data
 
print("Training Accuracy :", model_dt.score(X_train_scl, y_train))
print("Testing Accuracy :", model_dt.score(X_test_scl, y_test))


# In[103]:


y_preds = model_dt.predict(X_test)


# In[104]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)


# In[105]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
cm_display.plot()
plt.show()


# In[ ]:




