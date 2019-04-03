import pandas as pd
import numpy as np
import math as m
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# def prep_telco_data(df):
#     df['total_charges'] = df['total_charges'].convert_objects(convert_numeric=True)
#     df.total_charges.dropna(0, inplace=True)
#     for col in df.drop(columns=(['customer_id', 'total_charges', 'monthly_charges'])):
# #         print('encoding ' + col) -- error checking
#         encoder = LabelEncoder()
#         encoder.fit(df[col])
#         new_col = col + '_e'
#         df[new_col] = encoder.transform(df[col])
#     return df

def prep_telco(df_telco):
    df = df_telco.copy()

    df['tenure_year']= (df['tenure']/12).round(decimals=2)
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

    return df




####
def split_telco(df_telco):
    return train_test_split(df_telco, train_size=0.7, 
    random_state=123, stratify=df_telco[["survived"]])


def min_max_scale_telco(df_train, df_test):
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(df_train[['age', 'fare']])
    
    df_train_scaled[["age", "fare"]] = scaler.fit_transform(df_train[["age", "fare"]])
    df_test_scaled[["age", "fare"]] = scaler.fit_transform(df_test[["age", "fare"]])
    
    return df_train_scaled, df_test_scaled