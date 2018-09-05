
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('cd', '~/p/kaggle/yongkuk/180903_Digit_recognizer/src')


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn import svm


# In[7]:


labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, 
                                                                       train_size=0.8, random_state=0)


# In[10]:


i=4
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


# In[23]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# In[22]:


test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

