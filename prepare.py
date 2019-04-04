import pandas as pd
import numpy as np
import math as m
import io
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# This is the list of object columns from the Telco_DB
# These would need to be encoded if they need to be used as some form of metric

col = ['gender',
 'partner',
 'dependents',
 'phone_service',
 'multiple_lines',
 'online_security',
 'online_backup',
 'device_protection',
 'tech_support',
 'streaming_tv',
 'streaming_movies',
 'paperless_billing',
 'total_charges',
 'churn',
 'contract_type',
 'internet_service_type',
 'payment_type']


# This is ORION's for-loop for the label encoder
# Something went wrong when i tried using it for our purposes
# It was giving 3 sets of label encoded fields for every field regardless of whether or not it was an object or int

def prep_telco_data(df):
    df['total_charges'] = df['total_charges'].convert_objects(convert_numeric=True)
    df.total_charges.dropna(0, inplace=True)
    for col in df.drop(columns=(['customer_id', 'total_charges', 'monthly_charges'])):
    #  print('encoding ' + col) -- error checking
        encoder = LabelEncoder()
        encoder.fit(df[col])
        new_col = col + '_e'
        df[new_col] = encoder.transform(df[col])
    return df



# This is the main Telco DB prep function
    # Drops erroneous fields
    # Converts total_charges to a float from a string
    # Gives tenure as a year
    # Gives a family plan which by our means is a custm=omer who has a dependent, partner, and multiple lines
    # Gives household, phone_id, streaming_services, and online_security fields. Which are a combination of other fields of like groups
    # Encodes all object fields
    # Reindexes the dataframe to be easier to read. Puts important fields to the front and like fields next to each other
    # Drops null values

def prep_telco(df_telco):
    df = df_telco.copy()

    df.drop(['payment_type_id.1', 'contract_type_id.1', 'internet_service_type_id.1'], inplace=True, axis=1)

    df.total_charges = pd.to_numeric(df.total_charges, downcast='float', errors='coerce')

    df['tenure_year']= (df['tenure']/12).round(decimals=2)
    # this gives the 'year' that the customer is in i.e. 1st year 2nd year etc. 
    # so if they are 0.75 years tenure they are in the 1st year if they are 3.3 in the 4th year
    df['in_tenure_year']= np.ceil(df['tenure']/12)

    df['family_plan'] = (df.partner == 'Yes') & (df.dependents == 'Yes') & (df.multiple_lines == 'Yes')
    df.loc[df.family_plan == True, 'family_plan'] = 1
    df.loc[df.family_plan == False, 'family_plan'] = 0

    conditions_1 = [
    (df['dependents'] == 'No') & (df['partner'] == 'No'),
    (df['dependents'] == 'Yes') & (df['partner'] == 'No'),
    (df['dependents'] == 'No') & (df['partner'] == 'Yes'),
    (df['dependents'] == 'Yes') & (df['partner'] == 'Yes')]
    choices_1 = [0,1,2,3]
    df['household'] = np.select(conditions_1, choices_1)

    conditions_2 = [
    (df['multiple_lines'] == 'Yes') & (df['phone_service'] == 'Yes'),
    (df['multiple_lines'] == 'No') & (df['phone_service'] == 'Yes'),
    (df['multiple_lines'] == 'No phone service') & (df['phone_service'] == 'No'),]
    choices_2 = [2,1,0]
    df['phone_id'] = np.select(conditions_2, choices_2)

    conditions_3 =[
    (df['streaming_tv']=='No internet service')& (df['streaming_movies']=='No internet service'),
    (df['streaming_tv']=='No')& (df['streaming_movies']=='No'),
    (df['streaming_tv']=='No')& (df['streaming_movies']=='Yes'),
    (df['streaming_tv']=='Yes')& (df['streaming_movies']=='No'),
    (df['streaming_tv']=='Yes')& (df['streaming_movies']=='Yes')]
    choices_3 = [0,1,2,3,4]
    df['streaming_services'] = np.select(conditions_3, choices_3)

    conditions_4 =[
    (df['online_security']=='No internet service')& (df['online_backup']=='No internet service'),
    (df['online_security']=='No')& (df['online_backup']=='No'),
    (df['online_security']=='No')& (df['online_backup']=='Yes'),
    (df['online_security']=='Yes')& (df['online_backup']=='No'),
    (df['online_security']=='Yes')& (df['online_backup']=='Yes')]
    choices_4 = [0,1,2,3,4]
    df['online_security_backup'] = np.select(conditions_4, choices_4)

    conditions_5 = [
    (df['contract_type'] == 'Month-to-month'),
    (df['contract_type'] != 'Month-to-Month')]
    choices_5 = [1,0]
    df['monthly'] = np.select(conditions_5, choices_5)

    encoder_payment_type = LabelEncoder()
    encoder_payment_type.fit(df.payment_type)
    df = df.assign(payment_type_encode=encoder_payment_type.transform(df.payment_type))

    encoder_internet_service_type = LabelEncoder()
    encoder_internet_service_type.fit(df.internet_service_type)
    df = df.assign(internet_service_type_encode=encoder_internet_service_type.transform(df.internet_service_type))

    encoder_contract_type = LabelEncoder()
    encoder_contract_type.fit(df.contract_type)
    df = df.assign(contract_type_encode=encoder_contract_type.transform(df.contract_type))

    encoder_churn = LabelEncoder()
    encoder_churn.fit(df.churn)
    df = df.assign(churn_encode=encoder_churn.transform(df.churn))

    encoder_paperless_billing = LabelEncoder()
    encoder_paperless_billing.fit(df.paperless_billing)
    df = df.assign(paperless_billing_encode=encoder_paperless_billing.transform(df.paperless_billing))

    encoder_streaming_movies = LabelEncoder()
    encoder_streaming_movies.fit(df.streaming_movies)
    df = df.assign(streaming_movies_encode=encoder_streaming_movies.transform(df.streaming_movies))

    encoder_streaming_tv = LabelEncoder()
    encoder_streaming_tv.fit(df.streaming_tv)
    df = df.assign(streaming_tv_encode=encoder_streaming_tv.transform(df.streaming_tv))

    encoder_tech_support = LabelEncoder()
    encoder_tech_support.fit(df.tech_support)
    df = df.assign(tech_support_encode=encoder_tech_support.transform(df.tech_support))

    encoder_device_protection = LabelEncoder()
    encoder_device_protection.fit(df.device_protection)
    df = df.assign(device_protection_encode=encoder_device_protection.transform(df.device_protection))

    encoder_online_backup = LabelEncoder()
    encoder_online_backup.fit(df.online_backup)
    df = df.assign(online_backup_encode=encoder_online_backup.transform(df.online_backup))

    encoder_online_security = LabelEncoder()
    encoder_online_security.fit(df.online_security)
    df = df.assign(online_security_encode=encoder_online_security.transform(df.online_security))

    encoder_multiple_lines = LabelEncoder()
    encoder_multiple_lines.fit(df.multiple_lines)
    df = df.assign(multiple_lines_encode=encoder_multiple_lines.transform(df.multiple_lines))

    encoder_phone_service = LabelEncoder()
    encoder_phone_service.fit(df.phone_service)
    df = df.assign(phone_service_encode=encoder_phone_service.transform(df.phone_service))

    encoder_dependents = LabelEncoder()
    encoder_dependents.fit(df.dependents)
    df = df.assign(dependents_encode=encoder_dependents.transform(df.dependents))    

    encoder_gender = LabelEncoder()
    encoder_gender.fit(df.gender)
    df = df.assign(gender_encode=encoder_gender.transform(df.gender))
    
    encoder_partner = LabelEncoder()
    encoder_partner.fit(df.partner)
    df = df.assign(partner_encode=encoder_partner.transform(df.partner))

    df = df.reindex([
    'customer_id', 'churn','tenure', 'tenure_year', 'in_tenure_year', 'monthly_charges','total_charges', 
    'payment_type','payment_type_id',   'contract_type', 'contract_type_id','internet_service_type', 'internet_service_type_id','paperless_billing',
    'gender', 'senior_citizen', 'partner', 'dependents','phone_service', 'multiple_lines', 'streaming_tv', 'streaming_movies',  'online_security', 'online_backup', 
    'device_protection', 'tech_support','family_plan','household', 'phone_id', 'streaming_services', 'online_security_backup', 'monthly' ,'payment_type_encode', 'internet_service_type_encode',
    'contract_type_encode', 'churn_encode', 'paperless_billing_encode','streaming_movies_encode', 'streaming_tv_encode', 'tech_support_encode','device_protection_encode', 
    'online_backup_encode','online_security_encode', 'multiple_lines_encode','phone_service_encode', 'dependents_encode', 'gender_encode','partner_encode'
    ], axis='columns')

    df = df.dropna()

    return df


# This is a train_test_split function that should be used after all the data prep has been done i.e. using the prep_telco function
def split_telco(df_telco):
    return train_test_split(df_telco, train_size=0.7, 
    random_state=123, stratify=df_telco[["churn"]])

# This function need to be fixed
# When calling this function in another file (jupyter notebook) the selected dataframe doesn't become scaled or transformed.
# Took the guts of this function and put it into the notebook and it worked
def min_max_scale_telco(df_train, df_test):
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(df_train[['monthly_charges', 'total_charges']])
    
    df_train_scaled[['monthly_charges', 'total_charges']] = scaler.fit_transform(df_train[['monthly_charges', 'total_charges']])
    df_test_scaled[['monthly_charges', 'total_charges']] = scaler.fit_transform(df_test[['monthly_charges', 'total_charges']])
        
    ## This is the code that worked in the jupyter notebook
    # scaler = MinMaxScaler()
    # scaler.fit(train_df[['monthly_charges', 'total_charges']])

    # train_df[['monthly_charges', 'total_charges']] = scaler.transform(train_df[['monthly_charges', 'total_charges']])
    # test_df[['monthly_charges', 'total_charges']] = scaler.transform(test_df[['monthly_charges', 'total_charges']])

    return df_train_scaled, df_test_scaled


###### Following is a series of functions for viewing the prepped data

def df_print_missing_vals(df):
    # any missing values?
    # prints the number of missing values 
    # only for columns with missing values
    print("\nMissing Values:\n")
    null_counts = df.isnull().sum()
    if len(null_counts[null_counts > 0]) == 0:
        print("No missing values")
    else:
        print(null_counts[null_counts > 0])


def df_percent_missing_vals(df):
    return (df.isnull().sum() / df.shape[0]) * 100

def df_metadata(df) -> tuple:
    """
    return a tuple of (size, shape, info)
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    return (df.size, df.shape, buffer.getvalue())

def df_print_metadata(df) -> None:
    """
    print metadata of dataframe
    """
    size, shape, info = df_metadata(df)
    print("DATAFRAME METADATA")
    print(f"Size: {size}")
    print()
    print(f"Shape: {shape[0]} x {shape[1]}")
    print()
    print("Info:")
    print(info, end="")
    df_print_missing_vals(df)
    print("Percentages of Missing Values are:")
    print(df_percent_missing_vals(df))


##### Following are functions for viewing bassic statistical data

def df_print_r_and_p_values(X, y):
    r_and_p_values = {col: stats.pearsonr(X[col], y) for col in X.columns}
    print("PEARSON'S R")
    for k, v in r_and_p_values.items():
        col = k
        r, p = v
        print(f"{col}:")
        print(f"\tPearson's R is {r:.2f} ")
        print(f"\twith a significance p-value of {p: .3}\n")

def peekatdata(df):
    print('First five rows of the dataframe:')
    head_df = df.head()
    print(head_df)
    print('\n')
    print('Last five rows of the dataframe:')
    tail_df = df.tail()
    print(tail_df)
    print('\n')
    print('Shape of the dataframe:')
    shape_tuple = df.shape
    print(shape_tuple)
    print('\n')
    print('Describe dataframe:')
    describe_df = df.describe()
    print(describe_df)
    print('\n')
    print('Data types of each column:')
    print(df.dtypes)
    print('\n')
    print('Non-nulls in each column:')
    print(df.info())
    print('\n')
    print('Memory usage:')
    print(df.memory_usage)