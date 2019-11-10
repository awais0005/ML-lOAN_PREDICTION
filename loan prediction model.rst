.. code:: ipython3

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    %matplotlib inline
    import warnings
    warnings.filterwarnings('ignore')

.. code:: ipython3

    import os
    os.getcwd()




.. parsed-literal::

    '/Users/mohammadawais'



.. code:: ipython3

    os.chdir('/Users/mohammadawais/desktop/p_data')

.. code:: ipython3

    os.getcwd()




.. parsed-literal::

    '/Users/mohammadawais/Desktop/p_data'



.. code:: ipython3

    train=pd.read_csv('QL_Train.csv')
    test=pd.read_csv('QL_Test.csv')

.. code:: ipython3

    train_original=train.copy
    test_original=test.copy

.. code:: ipython3

    train.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001002</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001003</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>N</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001005</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001006</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001008</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    test.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001015</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5720</td>
          <td>0</td>
          <td>110.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001022</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>3076</td>
          <td>1500</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001031</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5000</td>
          <td>1800</td>
          <td>208.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001035</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2340</td>
          <td>2546</td>
          <td>100.0</td>
          <td>360.0</td>
          <td>NaN</td>
          <td>Urban</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001051</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>3276</td>
          <td>0</td>
          <td>78.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    train.dtypes




.. parsed-literal::

    Loan_ID               object
    Gender                object
    Married               object
    Dependents            object
    Education             object
    Self_Employed         object
    ApplicantIncome        int64
    CoapplicantIncome    float64
    LoanAmount           float64
    Loan_Amount_Term     float64
    Credit_History       float64
    Property_Area         object
    Loan_Status           object
    dtype: object



.. code:: ipython3

    test.dtypes




.. parsed-literal::

    Loan_ID               object
    Gender                object
    Married               object
    Dependents            object
    Education             object
    Self_Employed         object
    ApplicantIncome        int64
    CoapplicantIncome      int64
    LoanAmount           float64
    Loan_Amount_Term     float64
    Credit_History       float64
    Property_Area         object
    dtype: object



.. code:: ipython3

    train.shape




.. parsed-literal::

    (614, 13)



.. code:: ipython3

    test.shape




.. parsed-literal::

    (367, 12)



Exploratory Data Analysis
=========================

univariate analysis: studying one variable at a time
----------------------------------------------------

target variable: loan_status

.. code:: ipython3

    train['Loan_Status'].value_counts()




.. parsed-literal::

    Y    422
    N    192
    Name: Loan_Status, dtype: int64



.. code:: ipython3

    train['Loan_Status'].value_counts(normalize=True)




.. parsed-literal::

    Y    0.687296
    N    0.312704
    Name: Loan_Status, dtype: float64



.. code:: ipython3

    train['Loan_Status'].value_counts().plot.bar()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a21fb1b10>




.. image:: output_17_1.png


visualization: independent variable(categorical)
------------------------------------------------

categorical: these features have cateogries(gender, married,self_employed,loan_status)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    train.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001002</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001003</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>N</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001005</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001006</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001008</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>Y</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    train['Gender'].value_counts(normalize=True).plot.bar(title='gender')





.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a227e0c90>




.. image:: output_21_1.png


.. code:: ipython3

    train['Married'].value_counts(normalize=True).plot.bar(title='married')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a21fb1510>




.. image:: output_22_1.png


.. code:: ipython3

    train['Self_Employed'].value_counts(normalize=True).plot.bar(title='self_employed')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a229f4910>




.. image:: output_23_1.png


.. code:: ipython3

    train['Credit_History'].value_counts(normalize=True).plot.bar(title='credit_history')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a22ad48d0>




.. image:: output_24_1.png


visualization: independent variable(ordinal)
============================================

ordinal: variable in cateogrical feature having some order involved(dependents, education, property_area)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    train['Dependents'].value_counts(normalize=True).plot.bar(title='dependents')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a22bad950>




.. image:: output_27_1.png


.. code:: ipython3

    train['Education'].value_counts(normalize=True).plot.bar(title='education')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a21513c10>




.. image:: output_28_1.png


.. code:: ipython3

    train['Property_Area'].value_counts(normalize=True).plot.bar(title='property area')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a21edf310>




.. image:: output_29_1.png


visualization: independent variable(numerical)
==============================================

.. code:: ipython3

    sns.distplot(train['ApplicantIncome'])
    # this is not a normal distribuition because it is on the distribuition is mostly on the left side




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a22e89dd0>




.. image:: output_31_1.png


.. code:: ipython3

    train['ApplicantIncome'].plot.box()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a22f64a10>




.. image:: output_32_1.png


.. code:: ipython3

    train.boxplot(column='ApplicantIncome',by='Education')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a230e6d50>




.. image:: output_33_1.png


.. code:: ipython3

    sns.distplot(train['CoapplicantIncome'])
    # it is also not a normally distributed





.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a231f4710>




.. image:: output_34_1.png


.. code:: ipython3

    train['CoapplicantIncome'].plot.box()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a232d5a10>




.. image:: output_35_1.png


co-applicant income visualization is almost same as the applicantincome
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    df=train.dropna()# NaN values can't be plottable
    sns.distplot(df['LoanAmount'])




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a234426d0>




.. image:: output_37_1.png


.. code:: ipython3

    train['LoanAmount'].plot.box()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23599650>




.. image:: output_38_1.png


this distribuition is fairly normal but with a lot of outliers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bivariate analysis:
===================

::

   this analysis involves studying two variable and their relationship.

categorical independent variable vs Target Variable
---------------------------------------------------

.. code:: ipython3

    gender=pd.crosstab(train['Gender'],train['Loan_Status'])
    gender.div(gender.sum(1).astype(float),axis=0).plot(kind='bar',stacked=False,figsize=(4,4))




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23594d50>




.. image:: output_42_1.png


.. code:: ipython3

    gender=pd.crosstab(train['Gender'],train['Loan_Status'])
    gender.div(gender.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a2379c150>




.. image:: output_43_1.png


.. code:: ipython3

    married=pd.crosstab(train['Married'],train['Loan_Status'])
    married.div(married.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a2379c310>




.. image:: output_44_1.png


.. code:: ipython3

    dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
    dependents.div(dependents.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23933b90>




.. image:: output_45_1.png


.. code:: ipython3

    education=pd.crosstab(train['Education'],train['Loan_Status'])
    education.div(education.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23a35810>




.. image:: output_46_1.png


.. code:: ipython3

    selfemp=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
    selfemp.div(selfemp.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a21cfdb50>




.. image:: output_47_1.png


.. code:: ipython3

    credithistory=pd.crosstab(train['Credit_History'],train['Loan_Status'])
    credithistory.div(credithistory.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23b1e250>




.. image:: output_48_1.png


here the credithistory will highly affect the chances of the loan status
========================================================================

.. code:: ipython3

    propertyarea=pd.crosstab(train['Property_Area'],train['Loan_Status'])
    propertyarea.div(propertyarea.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23989490>




.. image:: output_50_1.png


numerical independent variable vs target variable
=================================================

.. code:: ipython3

    train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a23dbae90>




.. image:: output_52_1.png


.. code:: ipython3

    bins=[0,2500,4000,6000,81000]
    group=['low','average','high','very high']
    train['income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)
    income_bin=pd.crosstab(train['income_bin'],train['Loan_Status'])
    income_bin.div(income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
    plt.xlabel('ApplicantIncome')
    plt.ylabel('Percentage')
    plt.show()



.. image:: output_53_0.png


.. code:: ipython3

    bins=[0,1000,1800,2500,]
    group=['low','average','high']
    train['coapplicant_income']=pd.cut(df['CoapplicantIncome'],bins,labels=group)
    coapplicant_income=pd.crosstab(train['coapplicant_income'],train['Loan_Status'])
    coapplicant_income.div(coapplicant_income.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
    plt.xlabel('coapplicant_income')
    plt.ylabel('percentage')
    plt.show()



.. image:: output_54_0.png


.. code:: ipython3

    train['total_income']=train['ApplicantIncome']+train['CoapplicantIncome']

.. code:: ipython3

    bins=[0,2500,4000,6000,8100]
    groups=['low','average','high','very high']
    train['total_income']=pd.cut(train['total_income'],bins,labels=groups)
    total_income=pd.crosstab(train['total_income'],train['Loan_Status'])
    total_income.div(total_income.sum(1).astype(float),axis=0).plot(kind='bar',stacked=False)
    plt.xlabel('total_income')
    plt.ylabel('percentage')
    plt.show()



.. image:: output_56_0.png


.. code:: ipython3

    bins=[0,100,200,700]
    groups=['low','average','high']
    train['loan_amount']=pd.cut(df['LoanAmount'],bins,labels=groups)
    loan_amount=pd.crosstab(train['loan_amount'],train['Loan_Status'])
    loan_amount.div(loan_amount.sum(1).astype(float),axis=0).plot(kind='bar',stacked=False)
    plt.xlabel('loan_amount')
    plt.ylabel('percentage')
    plt.show()



.. image:: output_57_0.png


.. code:: ipython3

    train.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>count</td>
          <td>614.000000</td>
          <td>614.000000</td>
          <td>592.000000</td>
          <td>600.00000</td>
          <td>564.000000</td>
        </tr>
        <tr>
          <td>mean</td>
          <td>5403.459283</td>
          <td>1621.245798</td>
          <td>146.412162</td>
          <td>342.00000</td>
          <td>0.842199</td>
        </tr>
        <tr>
          <td>std</td>
          <td>6109.041673</td>
          <td>2926.248369</td>
          <td>85.587325</td>
          <td>65.12041</td>
          <td>0.364878</td>
        </tr>
        <tr>
          <td>min</td>
          <td>150.000000</td>
          <td>0.000000</td>
          <td>9.000000</td>
          <td>12.00000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <td>25%</td>
          <td>2877.500000</td>
          <td>0.000000</td>
          <td>100.000000</td>
          <td>360.00000</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <td>50%</td>
          <td>3812.500000</td>
          <td>1188.500000</td>
          <td>128.000000</td>
          <td>360.00000</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <td>75%</td>
          <td>5795.000000</td>
          <td>2297.250000</td>
          <td>168.000000</td>
          <td>360.00000</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <td>max</td>
          <td>81000.000000</td>
          <td>41667.000000</td>
          <td>700.000000</td>
          <td>480.00000</td>
          <td>1.000000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    train=train.drop('income_bin',axis=1)

.. code:: ipython3

    train=train.drop('loan_amount',axis=1)

.. code:: ipython3

    train=train.drop('coapplicant_income',axis=1)

.. code:: ipython3

    train=train.drop('total_income',axis=1)

.. code:: ipython3

    train['Dependents'].replace('3+', 3,inplace=True) 
    
    test['Dependents'].replace('3+', 3,inplace=True) 
    
    train['Loan_Status'].replace('N', 0,inplace=True) 
    
    train['Loan_Status'].replace('Y', 1,inplace=True)


.. code:: ipython3

    train.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001002</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001003</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>0</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001005</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001006</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001008</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



HEAT MAP
========

.. code:: ipython3

    matrix=train.corr()
    ax=plt.subplots()
    sns.heatmap(matrix,vmax=.8,square=True,cmap='BuPu')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a240e4310>




.. image:: output_66_1.png


applicant_income+loan_amount && credit_history+loan_status have the strong relations in the heat map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

missing values and outliers
===========================

.. code:: ipython3

    train.isnull().sum()




.. parsed-literal::

    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed        32
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    dtype: int64



for numerical variables: imputation using mean or median
--------------------------------------------------------

for categorical variables: imputation using mode
------------------------------------------------

.. code:: ipython3

    train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
    train['Married'].fillna(train['Married'].mode()[0],inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)

.. code:: ipython3

    train.isnull().sum()




.. parsed-literal::

    Loan_ID               0
    Gender                0
    Married               0
    Dependents            0
    Education             0
    Self_Employed         0
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History        0
    Property_Area         0
    Loan_Status           0
    dtype: int64



.. code:: ipython3

    train['Loan_Amount_Term'].value_counts()




.. parsed-literal::

    360.0    512
    180.0     44
    480.0     15
    300.0     13
    84.0       4
    240.0      4
    120.0      3
    36.0       2
    60.0       2
    12.0       1
    Name: Loan_Amount_Term, dtype: int64



360 is repeating 512 times
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)

.. code:: ipython3

    train.isnull().sum()




.. parsed-literal::

    Loan_ID               0
    Gender                0
    Married               0
    Dependents            0
    Education             0
    Self_Employed         0
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term      0
    Credit_History        0
    Property_Area         0
    Loan_Status           0
    dtype: int64



loan_amount has many outliers so that it is not an easy task to fill with the mean, i approach towards the median
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)

.. code:: ipython3

    train.isnull().sum()




.. parsed-literal::

    Loan_ID              0
    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    Loan_Status          0
    dtype: int64



.. code:: ipython3

    test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
    
    test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
    test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
    
    test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
    test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
    test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

.. code:: ipython3

    test.isnull().sum()




.. parsed-literal::

    Loan_ID              0
    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    dtype: int64



log transformation: ln(loanamount)=value
========================================

after log transformation the distribuition becomes normal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    train['loanamount_log']=np.log(train['LoanAmount'])
    train['loanamount_log'].hist(bins=20)
    # after log transformation the distribuition becomes normal





.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a2442a710>




.. image:: output_85_1.png


.. code:: ipython3

    train




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001002</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001003</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>0</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001005</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.189655</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001006</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.787492</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001008</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.948760</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>609</td>
          <td>LP002978</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2900</td>
          <td>0.0</td>
          <td>71.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>1</td>
          <td>4.262680</td>
        </tr>
        <tr>
          <td>610</td>
          <td>LP002979</td>
          <td>Male</td>
          <td>Yes</td>
          <td>3</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4106</td>
          <td>0.0</td>
          <td>40.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>1</td>
          <td>3.688879</td>
        </tr>
        <tr>
          <td>611</td>
          <td>LP002983</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>8072</td>
          <td>240.0</td>
          <td>253.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>5.533389</td>
        </tr>
        <tr>
          <td>612</td>
          <td>LP002984</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>7583</td>
          <td>0.0</td>
          <td>187.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>5.231109</td>
        </tr>
        <tr>
          <td>613</td>
          <td>LP002990</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>4583</td>
          <td>0.0</td>
          <td>133.0</td>
          <td>360.0</td>
          <td>0.0</td>
          <td>Semiurban</td>
          <td>0</td>
          <td>4.890349</td>
        </tr>
      </tbody>
    </table>
    <p>614 rows × 14 columns</p>
    </div>



.. code:: ipython3

    train['LoanAmount'].hist(bins=20)
    # what it looks like before log transformation, i.e not a normal distribuition




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a21f31850>




.. image:: output_87_1.png


.. code:: ipython3

    test['loanamount_log']=np.log(test['LoanAmount'])
    test['loanamount_log'].hist(bins=20)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1a246d6b10>




.. image:: output_88_1.png


.. code:: ipython3

    train




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001002</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001003</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>0</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001005</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.189655</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001006</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.787492</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001008</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.948760</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>609</td>
          <td>LP002978</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2900</td>
          <td>0.0</td>
          <td>71.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>1</td>
          <td>4.262680</td>
        </tr>
        <tr>
          <td>610</td>
          <td>LP002979</td>
          <td>Male</td>
          <td>Yes</td>
          <td>3</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4106</td>
          <td>0.0</td>
          <td>40.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>1</td>
          <td>3.688879</td>
        </tr>
        <tr>
          <td>611</td>
          <td>LP002983</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>8072</td>
          <td>240.0</td>
          <td>253.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>5.533389</td>
        </tr>
        <tr>
          <td>612</td>
          <td>LP002984</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>7583</td>
          <td>0.0</td>
          <td>187.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>5.231109</td>
        </tr>
        <tr>
          <td>613</td>
          <td>LP002990</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>4583</td>
          <td>0.0</td>
          <td>133.0</td>
          <td>360.0</td>
          <td>0.0</td>
          <td>Semiurban</td>
          <td>0</td>
          <td>4.890349</td>
        </tr>
      </tbody>
    </table>
    <p>614 rows × 14 columns</p>
    </div>



.. code:: ipython3

    test




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Loan_ID</th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>LP001015</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5720</td>
          <td>0</td>
          <td>110.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.700480</td>
        </tr>
        <tr>
          <td>1</td>
          <td>LP001022</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>3076</td>
          <td>1500</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>2</td>
          <td>LP001031</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5000</td>
          <td>1800</td>
          <td>208.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.337538</td>
        </tr>
        <tr>
          <td>3</td>
          <td>LP001035</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2340</td>
          <td>2546</td>
          <td>100.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.605170</td>
        </tr>
        <tr>
          <td>4</td>
          <td>LP001051</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>3276</td>
          <td>0</td>
          <td>78.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.356709</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>362</td>
          <td>LP002971</td>
          <td>Male</td>
          <td>Yes</td>
          <td>3</td>
          <td>Not Graduate</td>
          <td>Yes</td>
          <td>4009</td>
          <td>1777</td>
          <td>113.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.727388</td>
        </tr>
        <tr>
          <td>363</td>
          <td>LP002975</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4158</td>
          <td>709</td>
          <td>115.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.744932</td>
        </tr>
        <tr>
          <td>364</td>
          <td>LP002980</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>3250</td>
          <td>1993</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Semiurban</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>365</td>
          <td>LP002986</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5000</td>
          <td>2393</td>
          <td>158.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>5.062595</td>
        </tr>
        <tr>
          <td>366</td>
          <td>LP002989</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>9200</td>
          <td>0</td>
          <td>98.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>4.584967</td>
        </tr>
      </tbody>
    </table>
    <p>367 rows × 13 columns</p>
    </div>



drop loan_id
============

.. code:: ipython3

    train=train.drop('Loan_ID',axis=1)

.. code:: ipython3

    test=test.drop('Loan_ID',axis=1)

.. code:: ipython3

    train.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>Loan_Status</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>1</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>0</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.189655</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.787492</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>1</td>
          <td>4.948760</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    test.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5720</td>
          <td>0</td>
          <td>110.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.700480</td>
        </tr>
        <tr>
          <td>1</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>3076</td>
          <td>1500</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5000</td>
          <td>1800</td>
          <td>208.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.337538</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2340</td>
          <td>2546</td>
          <td>100.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.605170</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>3276</td>
          <td>0</td>
          <td>78.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.356709</td>
        </tr>
      </tbody>
    </table>
    </div>



y=target variable
=================

.. code:: ipython3

    y=train.iloc[:,11]

.. code:: ipython3

    y




.. parsed-literal::

    0      1
    1      0
    2      1
    3      1
    4      1
          ..
    609    1
    610    1
    611    1
    612    1
    613    0
    Name: Loan_Status, Length: 614, dtype: int64



x–> only independent variable
=============================

.. code:: ipython3

    x=train.drop('Loan_Status',axis=1)

.. code:: ipython3

    x




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>1</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.189655</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.787492</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.948760</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>609</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2900</td>
          <td>0.0</td>
          <td>71.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>4.262680</td>
        </tr>
        <tr>
          <td>610</td>
          <td>Male</td>
          <td>Yes</td>
          <td>3</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4106</td>
          <td>0.0</td>
          <td>40.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>3.688879</td>
        </tr>
        <tr>
          <td>611</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>8072</td>
          <td>240.0</td>
          <td>253.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.533389</td>
        </tr>
        <tr>
          <td>612</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>7583</td>
          <td>0.0</td>
          <td>187.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.231109</td>
        </tr>
        <tr>
          <td>613</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>4583</td>
          <td>0.0</td>
          <td>133.0</td>
          <td>360.0</td>
          <td>0.0</td>
          <td>Semiurban</td>
          <td>4.890349</td>
        </tr>
      </tbody>
    </table>
    <p>614 rows × 12 columns</p>
    </div>



.. code:: ipython3

    y




.. parsed-literal::

    0      1
    1      0
    2      1
    3      1
    4      1
          ..
    609    1
    610    1
    611    1
    612    1
    613    0
    Name: Loan_Status, Length: 614, dtype: int64



removing null values from x
---------------------------

.. code:: ipython3

    y.isnull().sum()




.. parsed-literal::

    0



.. code:: ipython3

    x.isnull().sum()




.. parsed-literal::

    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    loanamount_log       0
    dtype: int64



.. code:: ipython3

    x['LoanAmount'].value_counts()




.. parsed-literal::

    128.0    33
    120.0    20
    110.0    17
    100.0    15
    187.0    12
             ..
    570.0     1
    300.0     1
    376.0     1
    117.0     1
    311.0     1
    Name: LoanAmount, Length: 203, dtype: int64



.. code:: ipython3

    x['LoanAmount'].fillna(x['LoanAmount'].mode()[0],inplace=True)

.. code:: ipython3

    x.isnull().sum()




.. parsed-literal::

    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    loanamount_log       0
    dtype: int64



.. code:: ipython3

    x['loanamount_log'].value_counts()




.. parsed-literal::

    4.852030    33
    4.787492    20
    4.700480    17
    4.605170    15
    5.075174    12
                ..
    5.198497     1
    5.463832     1
    5.111988     1
    4.276666     1
    5.497168     1
    Name: loanamount_log, Length: 203, dtype: int64



.. code:: ipython3

    x['loanamount_log'].fillna(x['loanamount_log'].mode()[0],inplace=True)

.. code:: ipython3

    x.isnull().sum()




.. parsed-literal::

    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    loanamount_log       0
    dtype: int64



.. code:: ipython3

    x




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5849</td>
          <td>0.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>1</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.189655</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.787492</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.948760</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>609</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2900</td>
          <td>0.0</td>
          <td>71.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>4.262680</td>
        </tr>
        <tr>
          <td>610</td>
          <td>Male</td>
          <td>Yes</td>
          <td>3</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4106</td>
          <td>0.0</td>
          <td>40.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>3.688879</td>
        </tr>
        <tr>
          <td>611</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>8072</td>
          <td>240.0</td>
          <td>253.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.533389</td>
        </tr>
        <tr>
          <td>612</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>7583</td>
          <td>0.0</td>
          <td>187.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.231109</td>
        </tr>
        <tr>
          <td>613</td>
          <td>Female</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>4583</td>
          <td>0.0</td>
          <td>133.0</td>
          <td>360.0</td>
          <td>0.0</td>
          <td>Semiurban</td>
          <td>4.890349</td>
        </tr>
      </tbody>
    </table>
    <p>614 rows × 12 columns</p>
    </div>




x–> numerical change
====================

.. code:: ipython3

    from sklearn.preprocessing import LabelEncoder

.. code:: ipython3

    number=LabelEncoder()

.. code:: ipython3

    x['Gender']=number.fit_transform(x['Gender'].astype('str'))

.. code:: ipython3

    x['Married']=number.fit_transform(x['Married'].astype('str'))

.. code:: ipython3

    x['Education']=number.fit_transform(x['Education'].astype('str'))

.. code:: ipython3

    x['Self_Employed']=number.fit_transform(x['Self_Employed'].astype('str'))

.. code:: ipython3

    x['Property_Area']=number.fit_transform(x['Property_Area'].astype('str'))

.. code:: ipython3

    x




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5849</td>
          <td>0.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>4583</td>
          <td>1508.0</td>
          <td>128.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>4.852030</td>
        </tr>
        <tr>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>3000</td>
          <td>0.0</td>
          <td>66.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.189655</td>
        </tr>
        <tr>
          <td>3</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>2583</td>
          <td>2358.0</td>
          <td>120.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.787492</td>
        </tr>
        <tr>
          <td>4</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>6000</td>
          <td>0.0</td>
          <td>141.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.948760</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>609</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2900</td>
          <td>0.0</td>
          <td>71.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>4.262680</td>
        </tr>
        <tr>
          <td>610</td>
          <td>1</td>
          <td>1</td>
          <td>3</td>
          <td>0</td>
          <td>0</td>
          <td>4106</td>
          <td>0.0</td>
          <td>40.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>3.688879</td>
        </tr>
        <tr>
          <td>611</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>8072</td>
          <td>240.0</td>
          <td>253.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>5.533389</td>
        </tr>
        <tr>
          <td>612</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>0</td>
          <td>7583</td>
          <td>0.0</td>
          <td>187.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>5.231109</td>
        </tr>
        <tr>
          <td>613</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>4583</td>
          <td>0.0</td>
          <td>133.0</td>
          <td>360.0</td>
          <td>0.0</td>
          <td>1</td>
          <td>4.890349</td>
        </tr>
      </tbody>
    </table>
    <p>614 rows × 12 columns</p>
    </div>



splitting the dataset x & y in ratio 80:20
==========================================

.. code:: ipython3

    from sklearn.model_selection import train_test_split

.. code:: ipython3

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

.. code:: ipython3

    x_train




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>90</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>2958</td>
          <td>2900.0</td>
          <td>131.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>4.875197</td>
        </tr>
        <tr>
          <td>533</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>11250</td>
          <td>0.0</td>
          <td>196.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>5.278115</td>
        </tr>
        <tr>
          <td>452</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>3948</td>
          <td>1733.0</td>
          <td>149.0</td>
          <td>360.0</td>
          <td>0.0</td>
          <td>0</td>
          <td>5.003946</td>
        </tr>
        <tr>
          <td>355</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>3813</td>
          <td>0.0</td>
          <td>116.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.753590</td>
        </tr>
        <tr>
          <td>266</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>0</td>
          <td>4708</td>
          <td>1387.0</td>
          <td>150.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>5.010635</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>277</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>3103</td>
          <td>1300.0</td>
          <td>80.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.382027</td>
        </tr>
        <tr>
          <td>9</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>12841</td>
          <td>10968.0</td>
          <td>349.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>5.855072</td>
        </tr>
        <tr>
          <td>359</td>
          <td>1</td>
          <td>1</td>
          <td>3</td>
          <td>0</td>
          <td>0</td>
          <td>5167</td>
          <td>3167.0</td>
          <td>200.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>5.298317</td>
        </tr>
        <tr>
          <td>192</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>6033</td>
          <td>0.0</td>
          <td>160.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>5.075174</td>
        </tr>
        <tr>
          <td>559</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>4180</td>
          <td>2306.0</td>
          <td>182.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>5.204007</td>
        </tr>
      </tbody>
    </table>
    <p>491 rows × 12 columns</p>
    </div>



.. code:: ipython3

    x_test




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>454</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>7085</td>
          <td>0.0</td>
          <td>84.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>4.430817</td>
        </tr>
        <tr>
          <td>52</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>4230</td>
          <td>0.0</td>
          <td>112.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>4.718499</td>
        </tr>
        <tr>
          <td>536</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>6133</td>
          <td>3906.0</td>
          <td>324.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>5.780744</td>
        </tr>
        <tr>
          <td>469</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>4333</td>
          <td>2451.0</td>
          <td>110.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.700480</td>
        </tr>
        <tr>
          <td>55</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>0</td>
          <td>2708</td>
          <td>1167.0</td>
          <td>97.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>4.574711</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>337</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>1</td>
          <td>2500</td>
          <td>4600.0</td>
          <td>176.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>5.170484</td>
        </tr>
        <tr>
          <td>376</td>
          <td>1</td>
          <td>1</td>
          <td>3</td>
          <td>0</td>
          <td>0</td>
          <td>8750</td>
          <td>4996.0</td>
          <td>130.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>4.867534</td>
        </tr>
        <tr>
          <td>278</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>14583</td>
          <td>0.0</td>
          <td>436.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>6.077642</td>
        </tr>
        <tr>
          <td>466</td>
          <td>1</td>
          <td>1</td>
          <td>3</td>
          <td>1</td>
          <td>0</td>
          <td>2947</td>
          <td>1664.0</td>
          <td>70.0</td>
          <td>180.0</td>
          <td>0.0</td>
          <td>2</td>
          <td>4.248495</td>
        </tr>
        <tr>
          <td>303</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1625</td>
          <td>1803.0</td>
          <td>96.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.564348</td>
        </tr>
      </tbody>
    </table>
    <p>123 rows × 12 columns</p>
    </div>



.. code:: ipython3

    y_train




.. parsed-literal::

    90     1
    533    0
    452    0
    355    1
    266    1
          ..
    277    1
    9      0
    359    1
    192    0
    559    1
    Name: Loan_Status, Length: 491, dtype: int64



.. code:: ipython3

    y_test




.. parsed-literal::

    454    1
    52     0
    536    1
    469    0
    55     1
          ..
    337    1
    376    1
    278    1
    466    0
    303    1
    Name: Loan_Status, Length: 123, dtype: int64




multiple linear regression model
================================

.. code:: ipython3

    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    





.. parsed-literal::

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



.. code:: ipython3

    # Predicting the Test set results
    y_pred_mr = regressor.predict(x_test)

.. code:: ipython3

    for i in y_pred_mr:
        print('predicted value',i)


.. parsed-literal::

    predicted value 0.7821778884082171
    predicted value 0.7734849150463134
    predicted value 0.7412317132118237
    predicted value 0.8159822420608238
    predicted value 0.8480304675131294
    predicted value -0.0035496762789888336
    predicted value 0.8740322124385985
    predicted value 0.6909626935957124
    predicted value 0.025685151825700386
    predicted value 0.7780647880473668
    predicted value 0.7800981073507254
    predicted value 0.8538298352680986
    predicted value 0.7320967029278762
    predicted value 0.7824009041631746
    predicted value 0.8457845494381839
    predicted value 0.875561573540567
    predicted value 0.669965623550414
    predicted value 0.6575677036326696
    predicted value 0.8067879341021601
    predicted value 0.009110941534726608
    predicted value 0.10294359698872001
    predicted value 0.7834819504436783
    predicted value 0.8381551949424839
    predicted value 0.7934471756583157
    predicted value 0.7921230836124229
    predicted value 0.7811420310401721
    predicted value 0.6774233907674738
    predicted value 0.738408441589
    predicted value 0.14662877515921927
    predicted value 0.06717202997530945
    predicted value 0.7971914953176557
    predicted value 0.6870288682510646
    predicted value 0.7428639961995438
    predicted value 0.7671725528981997
    predicted value 0.7774216640930933
    predicted value -0.006871977378971494
    predicted value 0.7634794735228392
    predicted value 0.6340426907418774
    predicted value 0.8213512023992838
    predicted value 0.7832767039210926
    predicted value 0.8025450071715862
    predicted value 0.11311604695491745
    predicted value 0.9333918569266167
    predicted value 0.7607151281397759
    predicted value 0.7696553616932771
    predicted value 0.6798487196438976
    predicted value 0.839045434557249
    predicted value 0.773099498457088
    predicted value 0.7988695622258187
    predicted value 0.7006222111971623
    predicted value 0.8143782317603931
    predicted value 0.8119580853792661
    predicted value 0.7078532181017082
    predicted value 0.7138809708431264
    predicted value 0.7648956997122331
    predicted value 0.7311326069903317
    predicted value 0.8298890793718241
    predicted value 0.661628004051084
    predicted value 0.8453262633274566
    predicted value 0.735724818993836
    predicted value -0.05135270795222799
    predicted value 0.7533695571751139
    predicted value 0.7377790148673055
    predicted value 0.09391270970242012
    predicted value 0.8558686957118821
    predicted value 0.741893809230393
    predicted value 0.7675426313746834
    predicted value 0.7192857799240125
    predicted value 0.7441730424909865
    predicted value 0.6555858156364694
    predicted value 0.6905575591761658
    predicted value 0.8231324222955376
    predicted value 0.7350542580827657
    predicted value 0.8112954282819231
    predicted value 0.8100799203104013
    predicted value 0.7367069056751201
    predicted value 0.3928989058180719
    predicted value 0.7131770928123213
    predicted value 0.8891923291166496
    predicted value 0.061747876481387395
    predicted value 0.6863077678321855
    predicted value 0.7774746840945619
    predicted value 0.7438452321633123
    predicted value 0.863443493917824
    predicted value 0.7727398758080384
    predicted value 0.8632442028371728
    predicted value 0.7771957904809712
    predicted value 0.7588304246265878
    predicted value 0.8231989631949806
    predicted value 0.8169865271942707
    predicted value 0.7447928480743505
    predicted value 0.7897428240625955
    predicted value 0.8092616389606239
    predicted value 0.10817581554114836
    predicted value 0.7637801738523258
    predicted value -0.00975688893677501
    predicted value 0.7811740802984076
    predicted value 0.7149370035293761
    predicted value 0.7936028395830992
    predicted value 0.8484701650531453
    predicted value 0.7524309483219288
    predicted value 0.8232717186686574
    predicted value 0.781471354494559
    predicted value 0.7666461492983984
    predicted value 0.8958428313700467
    predicted value 0.7766679670208431
    predicted value 0.6490997688270076
    predicted value 0.8002266505283624
    predicted value 0.7544564966386543
    predicted value 0.8188316569409564
    predicted value 0.7257156559621841
    predicted value 0.7445560976281966
    predicted value 0.76045231513472
    predicted value 0.8836210169345646
    predicted value -0.03644966397564098
    predicted value 0.09415232259212486
    predicted value 0.7815652584971053
    predicted value 0.7230719494375085
    predicted value 0.7736123448718469
    predicted value 0.8017209018710161
    predicted value 0.7494607033679753
    predicted value 0.13314688661390867
    predicted value 0.841511384182793


.. code:: ipython3

    for i in y_test:
        print('actual value',i)


.. parsed-literal::

    actual value 1
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 0
    actual value 1
    actual value 1
    actual value 0
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 0
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 0
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1
    actual value 0
    actual value 0
    actual value 1
    actual value 0
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 1
    actual value 0
    actual value 1


logistic regression
===================

.. code:: ipython3

    from sklearn.linear_model import LogisticRegression 
    
    model = LogisticRegression() 
    model.fit(x_train, y_train)





.. parsed-literal::

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



.. code:: ipython3

    y_pred_lr = model.predict(x_test)


.. code:: ipython3

    for i in y_pred_lr:
        print('predicted value',i)


.. parsed-literal::

    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 0
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 1
    predicted value 0
    predicted value 1


accuracy testing
================

.. code:: ipython3

    from sklearn.metrics import accuracy_score
    accuracy_score(y_test,y_pred_lr)




.. parsed-literal::

    0.8373983739837398



almost 84% of the prediction from the train dataset are accurate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

predicting the test dataset
===========================

.. code:: ipython3

    test




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5720</td>
          <td>0</td>
          <td>110.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.700480</td>
        </tr>
        <tr>
          <td>1</td>
          <td>Male</td>
          <td>Yes</td>
          <td>1</td>
          <td>Graduate</td>
          <td>No</td>
          <td>3076</td>
          <td>1500</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5000</td>
          <td>1800</td>
          <td>208.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>5.337538</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Male</td>
          <td>Yes</td>
          <td>2</td>
          <td>Graduate</td>
          <td>No</td>
          <td>2340</td>
          <td>2546</td>
          <td>100.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.605170</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Not Graduate</td>
          <td>No</td>
          <td>3276</td>
          <td>0</td>
          <td>78.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.356709</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>362</td>
          <td>Male</td>
          <td>Yes</td>
          <td>3</td>
          <td>Not Graduate</td>
          <td>Yes</td>
          <td>4009</td>
          <td>1777</td>
          <td>113.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.727388</td>
        </tr>
        <tr>
          <td>363</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>4158</td>
          <td>709</td>
          <td>115.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Urban</td>
          <td>4.744932</td>
        </tr>
        <tr>
          <td>364</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>3250</td>
          <td>1993</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Semiurban</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>365</td>
          <td>Male</td>
          <td>Yes</td>
          <td>0</td>
          <td>Graduate</td>
          <td>No</td>
          <td>5000</td>
          <td>2393</td>
          <td>158.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>5.062595</td>
        </tr>
        <tr>
          <td>366</td>
          <td>Male</td>
          <td>No</td>
          <td>0</td>
          <td>Graduate</td>
          <td>Yes</td>
          <td>9200</td>
          <td>0</td>
          <td>98.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>Rural</td>
          <td>4.584967</td>
        </tr>
      </tbody>
    </table>
    <p>367 rows × 12 columns</p>
    </div>



.. code:: ipython3

    from sklearn.preprocessing import LabelEncoder
    number=LabelEncoder()
    test['Gender']=number.fit_transform(test['Gender'].astype('str'))
    test['Married']=number.fit_transform(test['Married'].astype('str'))
    test['Education']=number.fit_transform(test['Education'].astype('str'))
    test['Self_Employed']=number.fit_transform(test['Self_Employed'].astype('str'))
    test['Property_Area']=number.fit_transform(test['Property_Area'].astype('str'))


.. code:: ipython3

    test




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Gender</th>
          <th>Married</th>
          <th>Dependents</th>
          <th>Education</th>
          <th>Self_Employed</th>
          <th>ApplicantIncome</th>
          <th>CoapplicantIncome</th>
          <th>LoanAmount</th>
          <th>Loan_Amount_Term</th>
          <th>Credit_History</th>
          <th>Property_Area</th>
          <th>loanamount_log</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5720</td>
          <td>0</td>
          <td>110.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.700480</td>
        </tr>
        <tr>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>3076</td>
          <td>1500</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>0</td>
          <td>5000</td>
          <td>1800</td>
          <td>208.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>5.337538</td>
        </tr>
        <tr>
          <td>3</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>0</td>
          <td>2340</td>
          <td>2546</td>
          <td>100.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.605170</td>
        </tr>
        <tr>
          <td>4</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>3276</td>
          <td>0</td>
          <td>78.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.356709</td>
        </tr>
        <tr>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <td>362</td>
          <td>1</td>
          <td>1</td>
          <td>3</td>
          <td>1</td>
          <td>1</td>
          <td>4009</td>
          <td>1777</td>
          <td>113.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.727388</td>
        </tr>
        <tr>
          <td>363</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>4158</td>
          <td>709</td>
          <td>115.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>2</td>
          <td>4.744932</td>
        </tr>
        <tr>
          <td>364</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>3250</td>
          <td>1993</td>
          <td>126.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>1</td>
          <td>4.836282</td>
        </tr>
        <tr>
          <td>365</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5000</td>
          <td>2393</td>
          <td>158.0</td>
          <td>360.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>5.062595</td>
        </tr>
        <tr>
          <td>366</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>9200</td>
          <td>0</td>
          <td>98.0</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>0</td>
          <td>4.584967</td>
        </tr>
      </tbody>
    </table>
    <p>367 rows × 12 columns</p>
    </div>



.. code:: ipython3

    pred_test=model.predict(test)

.. code:: ipython3

    pred_test




.. parsed-literal::

    array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
           0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0,
           1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])



