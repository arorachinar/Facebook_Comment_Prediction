
# coding: utf-8

# # ML Assignment: Submitted by Chinar Arora(CXA180005)

# In[484]:


import os
from fnmatch import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[485]:


root = '/Users/chinararora/Documents/Semester_2@UTD/ML/Assignment-1/Dataset/Data'
pattern = "*.csv"
csv_list = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            csv_list.append(os.path.join(path, name))


# In[486]:


csv_out = 'consolidated.csv'
csv_header = 'Page_Popularity,Page_Checkins,Page_talking_about,Page_category,Min_E1,Max_E1,Avg_E1,Median_E1,Std_E1,Min_E2,Max_E2,Avg_E2,Median_E2,Std_E2,Min_E3,Max_E3,Avg_E3,Median_E3,Std_E3,Min_E4,Max_E4,Avg_E4,Median_E4,Std_E4,Min_E5,Max_E5,Avg_E5,Median_E5,Std_E5,CC1,CC2,CC3,CC4,CC5,Base_Time,Post_length,Post_share_count,Post_promotion_status,H_local,Post_Sunday,Post_Monday,Post_Tuesday,Post_Wednesday,Post_Thursday,Post_Friday,Post_Saturday,Base_Sunday,Base_Monday,Base_Tuesday,Base_Wednesday,Base_Thursday,Base_Friday,Base_Saturday,Comments_Next_Hhours'
csv_merge = open(csv_out, 'w')
csv_merge.write(csv_header)
csv_merge.write('\n')

for file in csv_list:
    
    csv_in = open(file)
    for line in csv_in: 
        csv_merge.write(line)


# In[487]:


df=pd.read_csv(csv_out)
df.head()


# In[488]:


df.shape


# In[489]:


df.columns


# In[7]:


#df.info()


# #### Randomly partitioning dataset using 70/30 split rule 

# In[490]:


xTrain, xTest, yTrain, yTest = train_test_split(df.iloc[:,0:53],df.iloc[:,-1], test_size = 0.3, random_state = 0)


# In[491]:


xTrain.head()


# In[492]:


xTrain.shape


# In[493]:


xTest.shape


# #### Exploratory Analysis- To understand correlation between predictors and response variable

# ##### From the correlation plot obtained, it can be interpreted that Min E5,Base_Time is negatively correlated with the target variable.Other features do not seem to be correlated strongly with the target variable.

# In[494]:


plt.subplots(figsize=(20,15))
sns.heatmap(df.corr())


# #### Based on exploratory data analysis and intuition, it seems that most important variable in predicting the comments in next  H hours should be post features like- likes,shares,length of post. Intrinsic features like when the post was made(CC1,CC2.CC3,CC4,CC5) also seem relevant. These have been included in model to experiment, later a legitimate approach is used to determine true relevant features of target variable.

# In[495]:


colsSelected=['Page_Popularity','Page_Checkins','Page_talking_about','Base_Time','Post_length',
       'Post_share_count','Post_promotion_status','CC1','CC2','CC3','CC4','CC5','H_local']


# In[496]:


xTest_sub,xTrain_sub=xTest[colsSelected],xTrain[colsSelected]
xTrain_sub.head()


# In[497]:


xTrain_sub.shape


# In[498]:


xTest_sub.shape


# In[499]:


xTrain_matrix=np.array(xTrain_sub)
xTest_matrix=np.array(xTest_sub)
yTrain_vector=np.array(yTrain)
yTest_vector=np.array(yTest)
yTrain_vector.shape


# In[500]:


yTest_vector.shape


# In[501]:


xTrain_matrix.shape


# #### Plotting the frequency distribution of the selected variables to understand the spread of the data

# In[502]:


def feature_distribution(xTrain):
    plt.figure(figsize=(10,4))
    plt.grid(True)
    plt.xlim([-100,5000])
    plt.ylim([0,100])
    plt.subplot(2,2,1)
    myplot = plt.hist(xTrain[:,0],label = 'Page_Popularity')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    myplot = plt.legend()
    plt.subplot(2,2,2)
    myplot = plt.hist(xTrain[:,1],label = 'Page_Checkins')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    myplot = plt.legend()
    plt.subplot(2,2,3)
    myplot = plt.hist(xTrain[:,2],label = 'Page_talking_about')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    myplot = plt.legend()
    plt.subplot(2,2,4)
    myplot = plt.hist(xTrain[:,4],label = 'Post_length')
    #plt.title('Histogram for columns 1,2,3,4')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    myplot = plt.legend()


# In[503]:


feature_distribution(xTrain_matrix)


# #### Due to varying magnitude of features, we need to perform feature normalisation. The method below subtracts the mean and divides the value by standard deviation to scale the values

# In[504]:


def normaliseData(X):
    scaler=preprocessing.StandardScaler().fit(X)
    Xnorm=scaler.transform(X)
    return Xnorm


# In[505]:


xTrain_matrix=normaliseData(xTrain_matrix)
xTrain_matrix


# In[506]:


xTest_matrix=normaliseData(xTest_matrix)
xTest_matrix


# In[507]:


feature_distribution(XtrainNorm)


# In[508]:


#Add column of ones for bias initialisation


# In[509]:


XtrainNorm=np.hstack((np.ones((xTrain_matrix.shape[0],1)),xTrain_matrix))
XtestNorm=np.hstack((np.ones((xTest_matrix.shape[0],1)),xTest_matrix))
XtrainNorm


# In[510]:


XtrainNorm.shape


# In[511]:


XtestNorm


# In[512]:


XtrainMatrix= np.matrix(XtrainNorm) 
yTrain = np.matrix(yTrain_vector).T


# In[513]:


XtestMatrix = np.matrix(XtestNorm)  
yTest = np.matrix(yTest_vector).T


# In[514]:


def computeCost(X,y,theta):
    error=np.power(((X * theta.T) - y), 2)
    return np.sum(error) / (2 * len(X))


# In[515]:


def gradientDescent(Xtrain, ytrain,Xtest,ytest, theta, alpha, iters,threshold=0.001):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.shape[1])
    trainCost = np.zeros(iters)
    testCost=np.zeros(iters)
    for i in range(iters):
        error=(Xtrain * theta.T) - ytrain
        for j in range(parameters):
            term=np.multiply(error, Xtrain[:,j])
            temp[0,j]=theta[0,j] - ((alpha / len(Xtrain)) * np.sum(term))    
        theta=temp
        trainCost[i]=computeCost(Xtrain, ytrain, theta)
        testCost[i]=computeCost(Xtest,ytest,theta)
        
        if i!=0 and trainCost[i-1]-trainCost[i]<threshold:
            print('Converged in ',str(i),' iterations.')
            break
        
    return theta,trainCost,testCost


# In[516]:


def plotLearningCurve(trainCost,testCost):
    size=trainCost.size-trainCost[trainCost==0].size-1
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(list(range(0,size)), trainCost[0:size], 'r')
    ax.plot(list(range(0,size)), testCost[0:size], 'b')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Training/Testing Cost VS Iterations')


# #### Experiment 1:

# In[517]:


alpha=0.01
iterations=1000
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost= gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations)
theta


# In[288]:


computeCost(XtrainMatrix,yTrain,theta)


# In[289]:


computeCost(XtestMatrix,yTest,theta)


# In[290]:


plotLearningCurve(trainCost,testCost)


# In[466]:


alpha=0.001
iterations=3000
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost = gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations)
theta


# In[206]:


computeCost(XtrainMatrix,yTrain,theta)


# In[207]:


computeCost(XtestMatrix,yTest,theta)


# In[208]:


plotLearningCurve(trainCost,testCost)


# In[467]:


alpha=0.05
iterations=1000
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost = gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations)
theta


# In[210]:


computeCost(XtrainMatrix,yTrain,theta)


# In[211]:


computeCost(XtestMatrix,yTest,theta)


# In[212]:


plotLearningCurve(trainCost,testCost)


# In[468]:


alpha=0.1
iterations=1000
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost = gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations)
theta


# In[214]:


computeCost(XtrainMatrix,yTrain,theta)


# In[215]:


computeCost(XtestMatrix,yTest,theta)


# In[216]:


plotLearningCurve(trainCost,testCost)


# #### Experiment 2:

# In[474]:


convergence_criteria=0.0001
iterations=5000
alpha=0.01
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost= gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations,convergence_criteria)
theta


# In[475]:


computeCost(XtrainMatrix,yTrain,theta)


# In[476]:


computeCost(XtestMatrix,yTest,theta)


# In[219]:


plotLearningCurve(trainCost,testCost)


# In[477]:


convergence_criteria=0.001
iterations=5000
alpha=0.01
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost= gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations,convergence_criteria)
theta


# In[225]:


plotLearningCurve(trainCost,testCost)


# In[478]:


convergence_criteria=0.1
iterations=1000
alpha=0.01
theta=np.zeros((XtrainMatrix.shape[1],1)).T
theta,trainCost,testCost= gradientDescent(XtrainMatrix,yTrain,XtestMatrix,yTest,theta,alpha,iterations,convergence_criteria)
theta


# In[479]:


computeCost(XtrainMatrix,yTrain,theta)


# In[480]:


computeCost(XtestMatrix,yTest,theta)


# In[373]:


plotLearningCurve(trainCost,testCost)


# #### Experiment 3:

# In[357]:


randomCols=['Page_Popularity','Page_Checkins','Page_talking_about','H_local','Post_length']


# In[403]:


XtrainRandom=xTrain[randomCols]
XtrainRandom=np.matrix(XtrainRandom)
XtrainRandom=normaliseData(XtrainRandom)
XtrainRandom.shape


# In[404]:


XtestRandom=xTest[randomCols]
XtestRandom=np.matrix(XtestRandom)
XtestRandom=normaliseData(XtestRandom)
XtestRandom.shape


# In[405]:


XtrainRandom=np.hstack((np.ones((XtrainRandom.shape[0],1)),XtrainRandom))
XtrainRandom=np.matrix(XtrainRandom)
XtestRandom=np.hstack((np.ones((XtestRandom.shape[0],1)),XtestRandom))
XtestRandom=np.matrix(XtestRandom)


# In[408]:


iterations=1000
alpha=0.01
theta=np.zeros((XtrainRandom.shape[1],1)).T
theta,trainCost,testCost= gradientDescent(XtrainRandom,yTrain,XtestRandom,yTest,theta,alpha,iterations)
theta


# In[409]:


computeCost(XtrainRandom,yTrain,theta)


# In[410]:


computeCost(XtestRandom,yTest,theta)


# In[411]:


plotLearningCurve(trainCost,testCost)


# #### Experiment 4:

# In[453]:


from sklearn.ensemble import ExtraTreesClassifier
#To suppres exponenetial form of decimal values
np.set_printoptions(suppress=True)
model = ExtraTreesClassifier()
model.fit(xTrain,yTrain)
print(model.feature_importances_)


# In[462]:


arr[::-1][0:6]


# In[454]:


arr=np.sort(model.feature_importances_,axis=None)
model.feature_importances_.argsort()[-5:][::-1]


# In[435]:


xSelectTrain=xTrain[['CC5','CC1','CC4','Post_length','Base_Time']]
xSelectTrain=normaliseData(xSelectTrain)
xSelectTrain=np.hstack((np.ones((xSelectTrain.shape[0],1)),xSelectTrain))
xSelectTrain=np.matrix(xSelectTrain)
xSelectTrain


# In[436]:


xSelectTest=xTest[['CC5','CC1','CC4','Post_length','Base_Time']]
xSelectTest=normaliseData(xSelectTest)
xSelectTest=np.hstack((np.ones((xSelectTest.shape[0],1)),xSelectTest))
xSelectTest=np.matrix(xSelectTest)
xSelectTest


# In[481]:


iterations=1000
alpha=0.1
theta=np.zeros((XtrainRandom.shape[1],1)).T
theta,trainCost,testCost= gradientDescent(xSelectTrain,yTrain,xSelectTest,yTest,theta,alpha,iterations)
theta


# In[482]:


computeCost(xSelectTrain,yTrain,theta)


# In[483]:


computeCost(xSelectTest,yTest,theta)


# In[440]:


plotLearningCurve(trainCost,testCost)


# ## Final Equation:                                                               Comments_In_Next_Hhours=7.15+11.43*CC5+5.73*CC1+6.26*CC4+0.145*Post_Length-5.69*Base_time

# In[ ]:


#End

