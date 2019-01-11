import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

"""
########################
 Porfolio preprocessing
########################
"""

def add_portfolio_name(portfolio_df):
    """
    INPUT :
        portfolio_df : portfolio df

    RETURN :
        portfolio_df : updated portfolio df with addtional col of name of each offer ids
    """
    portfolio_df = portfolio_df.copy()
    portfolio_df['name'] = portfolio_df.offer_type.astype(str) + "_" + portfolio_df.difficulty.astype(str) +\
                      "_" + portfolio_df.reward.astype(str) + \
                     "_" + portfolio_df.duration.astype(str)
    return portfolio_df

def one_hot_channels(portfolio_df):
    """
    INPUT :
        portfolio_df : portfolio df

    RETURN :
        portfolio_df : updated portfolio df with addtional columsn of one hot encoded of channels columns
    """
    portfolio_df = portfolio_df.copy()
    channels = ['web', 'email', 'mobile', 'social']

    for channel in channels:
        portfolio_df[channel] = portfolio_df.channels.apply(lambda x: channel in x)*1

    #drop channels column
    portfolio_df = portfolio_df.drop('channels', axis=1)
    return portfolio_df

def portfolio_preprocessing(portfolio_df):
    """
    INPUT :
        portfolio_df : portfolio df

    RETURN :
        portfolio_df : updated preprocessed portfolio df
    """
    portfolio_df = portfolio_df.copy()
    # add portfolio add_portfolio_name
    portfolio_df = add_portfolio_name(portfolio_df)
    # one_hot_channels
    portfolio_df = one_hot_channels(portfolio_df)
    return portfolio_df

"""
######################
Profile preprocessing
######################
"""

def profile_parse_dates(profile_df):
    """
    INPUT :
        portfolio_df : profile df with became_member_on as int

    RETURN :
        profile_df : updated portfolio df with parsed dates as datetime in became_member_on
    """

    profile_df = profile_df.copy()

    #convert to string
    profile_df['became_member_on'] = profile_df.became_member_on.apply(lambda x: str(x))
    #convert to datetime format
    profile_df['became_member_on'] = pd.to_datetime(profile_df.became_member_on)
    return profile_df

"""
########################
Transcript preprocessing
########################
"""

def encode_offer_id(x):
    """
    This function return a value of "offer id" or "offer_id"
    """
    try :
        return x['offer id']
    except:
        return x['offer_id']

def transcript_encoded(transcript_df):
    """
    To encode column :
        - event : received, reviewed, completed
        - value : offer_id, amount

    INPUT :
        transcript_df : transcript def
    RETURN :
        transcript_encoded : encoded transcript df of value column

    """

    transcript_encoded = transcript_df.copy()
    transcript_encoded['offer_id'] = transcript_df[transcript_df.event.isin(['offer received', 'offer viewed', 'offer completed'])]\
                                                             .value.apply(encode_offer_id)
    transcript_encoded['amount'] =  transcript_df[transcript_df.event.isin(['transaction'])].value.apply(lambda x: x['amount'])
    return transcript_encoded

def merge_transcript_profile(transcript_df, profile_df):
    """
    This function is to merge profile df to transcript df
    INPUT:
        transcript_df : transcript df
        profile_df : profile df
    RETURN :
        transcript_profile_df : a merge of transcript and profile df
    """
    profile_df = profile_parse_dates(profile_df)
    transcript_encoded_df = transcript_encoded(transcript_df)
    transcript_profile_df = pd.merge(transcript_encoded_df, profile_df, left_on=['person'],
                                      right_on = ['id'], how ='left')
    transcript_profile_df = transcript_profile_df.drop(['id'], axis=1)

    return transcript_profile_df

def merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df):
    """
    This function is to merge profile to a merged df of profile & transcript df
    INPUT:
        transcript_df : transcript df
        profile_df : profile df
        portfolio_df : portfolio df
    RETURN :
        transcript_profile_porto: a merge of transcript and profile, and portfolio df
    """

    portfolio_df = portfolio_preprocessing(portfolio_df)
    transcript_profile_df = merge_transcript_profile(transcript_df, profile_df)
    transcript_profile_porto = pd.merge(transcript_profile_df, portfolio_df, left_on = 'offer_id', right_on ='id', how='left').drop('id', axis=1)
    #parse date became_member_on
    transcript_profile_porto = profile_parse_dates(transcript_profile_porto)

    return transcript_profile_porto

def find_invalid_index(transcript_df, profile_df, portfolio_df):
    """
    INPUT : transcript, profile, portfolio dataframe
    RETURN : a list of invalid index in transcript dataframe
    """

    #merge transcript, profile, and portfolio dataframe
    trascript_merge_df = merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df)

    # list of invalid index of offer completed
    invalid_index = []

    #iterate over profile_id (person)
    for profile_id in tqdm(trascript_merge_df.person.unique()):
        # take a subest_df for profile_id person
        subset_df = trascript_merge_df[trascript_merge_df.person == profile_id]
        # take a subset of 'offer completed'
        completed_df = subset_df[subset_df.event == 'offer completed']

        # iterate over the completed offer_id
        for offer in completed_df.offer_id.unique():
            # take a subset df of completed offer
            comp = completed_df[completed_df.offer_id == offer]
            # estimate the offer received time in hours using the offer duration (in days)
            start_time = comp.time.values - (comp.duration.values *24)
            # the offer completed time
            comp_time = comp.time.values
            # take the subset within start_time and comp_time
            subset = subset_df[(subset_df.time >= start_time[0]) & (subset_df.time <= comp.time.values[0])]
            # take only event of offer viewed for the given offer_id
            subset_viewed = subset[(subset.event == 'offer viewed') & ( subset.offer_id == offer)]

            # check whether subset_viewed is empty or not, if it is empty then the offer completed is not valid
            # because the offer is completed before the customer viewed it,
            # it means  that the customer was not affected by the portfolio campaign
            if subset_viewed.shape[0] == 0 :
                invalid_index.extend(comp.index)

    return invalid_index


def transcript_cleaning(transcript_df, profile_df, portfolio_df):
    """
    INPUT :
        transcript_df : transcript df
        profile_df : profile df
        portfolio_df : portfolio df
    RETURN :
        transcript_clean_df : a clean transcript df that has additinal column to mark the invalid offer completed

    This function will check whether a saved "transcript_clean.csv" is available and use it if available
    Ohter wise, the function will continue to execute the next block code to clean the dataframe,
    and  save a clean transcript df as "transcript_clean.csv".

    The function will mark where the invalid offer completed as 1, else 0.
    The invalid offer completed is the offer completed when the customer never viewed the offer.
    """
    try:
        transcript_clean_df = load_file('data/transcript_clean.csv')
        print("The transcript_clean.csv and transcript_merge.csv file are available at local folder")

    except:
        transcript_clean_df = merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df)
        invalid_index = find_invalid_index(transcript_df, profile_df, portfolio_df)

        #marking invalid in transcript_merge_df
        transcript_clean_df.loc[transcript_clean_df.index.isin(invalid_index),"invalid"] = 1
        transcript_clean_df.loc[~transcript_clean_df.index.isin(invalid_index),"invalid"] = 0

        #saving df
        transcript_clean_df.to_csv('data/transcript_clean.csv')

    return transcript_clean_df

def transcript_preprocessing(transcript_df, profile_df, portfolio_df):
    """
    INPUT : transcript_df, profile_df, portfolio_df : DataFrame
    RETURN :
        transcript_valid_df : transcript df that only contains the valid offer, mark as 0 in invalid column
        transcript_all_df : transcript all df as return by transcript_cleaning function
    """
    transcript_all_df = transcript_cleaning(transcript_df, profile_df, portfolio_df)
    transcript_valid_df = transcript_all_df[transcript_all_df.invalid == 0]
    return transcript_valid_df, transcript_all_df

def load_file(filepath):
    """Load file csv"""
    df_clean = pd.read_csv(filepath)
    df_clean = df_clean.set_index(df_clean.columns[0])
    df_clean = profile_parse_dates(df_clean)
    return df_clean


"""
################################
FEATURES EXTRACTION
################################
"""

def get_response_time(df, profile_id):
    """
    INPUT :
        df : DataFrame, clean merge transcript df
        profile_id : profile id
    RETURN :
        response_time_series : a Series of response_time of offering for given profile_id

    Response time is caluclated from the time delta between time(hour) of offer viewed to offer completed
    """
    subset_offer_typ = df[df.event == 'offer completed']['name'].unique().tolist()

    response_time_series = pd.Series(name=profile_id)

    for offer in subset_offer_typ:
        completed_time = df[(df.name == offer) & (df.event == 'offer completed')]['time'].values
        reviewed_time = df[(df.name == offer) & (df.event == 'offer viewed')]['time'].values
        if (completed_time.shape[0] != reviewed_time.shape[0]) and (reviewed_time.shape[0] != 0):
            reviewed_time_clean = np.array([])
            for t in completed_time:
                reviewed_time_clean = np.append(reviewed_time_clean, reviewed_time[reviewed_time <= t].max())
            response_time = completed_time - reviewed_time_clean
        else:
            response_time = completed_time - reviewed_time
            response_time = response_time[response_time > 0]

        if response_time.shape[0] != 0 :
            response_time_avg = response_time.mean()
        else:
            response_time_avg = np.nan
        response_time_series[offer +'_' +'response_time_avg'] = response_time_avg

    return response_time_series

def get_spending_series(df, profile_id):
    """
    INPUT :
        df : DataFrame, clean merge transcript df
        profile_id : profile id
    RETURN :
        spending_series : a Series of spending for a given profile_id (avg, transaction_count, and sum_spending)

    """
    avg_spending = df.amount.mean()
    transaction_count = df.amount.count()
    sum_spending = df.amount.sum()

    spending_series = pd.Series([avg_spending, transaction_count, sum_spending], index=["avg_spending", "transaction_count", 'sum_spending'], name=profile_id)
    return spending_series


def get_event_typ_series(df, profile_id):
    """
    INPUT :
        df : DataFrame, clean merge transcript df
        profile_id : profile id
    RETURN :
        event_typ_series : a Series of event_type value counts for given profile_id
    """
    event_typ_series = (df.event + "_" + df.name).value_counts()
    event_typ_series.name = profile_id
    return event_typ_series

def get_attributes_series(df, profile_id):
    """
    INPUT :
        df : DataFrame, clean merge transcript df
        profile_id : profile id
    RETURN :
        attributes_series : a Series of attributes for given profile_id

    """
    event_typ_series = get_event_typ_series(df, profile_id)
    response_time_series = get_response_time(df, profile_id)
    spending_series = get_spending_series(df, profile_id)
    attributes_series = pd.concat([event_typ_series, response_time_series, spending_series], axis=0)
    return attributes_series

def generate_attributes(portfolio_df):
    """
    INPUT :
        portfolio_df : portfolio df
    RETURN :
        attributes: a list of attributes name
    """

    portfolio_df = portfolio_preprocessing(portfolio_df)
    events = ['offer received', 'offer viewed', 'offer completed']
    portfolio_names = [ event +"_"+ name for event in events for name in portfolio_df.name.tolist() ]

    response_time_attributes = [name +'_' +'response_time_avg' for name in portfolio_df.name.tolist() ]
    attributes = portfolio_names + response_time_attributes + ["avg_spending", "transaction_count", "sum_spending"]
    return attributes

def feature_fillna(profile_updated_df):
    """
    This function is to fill missing value with zero (0) for selected feature
    INPUT: profile_updated_df with missing values
    RETURN : profile_updated_df with no missing values
    """
    profile_updated_df = profile_updated_df.copy()

    cols_to_fillna = ['offer received_bogo_10_10_7',
                     'offer received_bogo_10_10_5',
                     'offer received_informational_0_0_4',
                     'offer received_bogo_5_5_7',
                     'offer received_discount_20_5_10',
                     'offer received_discount_7_3_7',
                     'offer received_discount_10_2_10',
                     'offer received_informational_0_0_3',
                     'offer received_bogo_5_5_5',
                     'offer received_discount_10_2_7',
                     'offer viewed_bogo_10_10_7',
                     'offer viewed_bogo_10_10_5',
                     'offer viewed_informational_0_0_4',
                     'offer viewed_bogo_5_5_7',
                     'offer viewed_discount_20_5_10',
                     'offer viewed_discount_7_3_7',
                     'offer viewed_discount_10_2_10',
                     'offer viewed_informational_0_0_3',
                     'offer viewed_bogo_5_5_5',
                     'offer viewed_discount_10_2_7',
                     'offer completed_bogo_10_10_7',
                     'offer completed_bogo_10_10_5',
                     'offer completed_informational_0_0_4',
                     'offer completed_bogo_5_5_7',
                     'offer completed_discount_20_5_10',
                     'offer completed_discount_7_3_7',
                     'offer completed_discount_10_2_10',
                     'offer completed_informational_0_0_3',
                     'offer completed_bogo_5_5_5',
                     'offer completed_discount_10_2_7',
                     'avg_spending',
                     'sum_spending'
                     ]

    col_null_frac = profile_updated_df.isnull().sum() / profile_updated_df.shape[0]
    cols_to_drop = col_null_frac[col_null_frac ==1].index.tolist()
    profile_updated_df[cols_to_fillna] = profile_updated_df[cols_to_fillna].fillna(0)
    profile_updated_df = profile_updated_df.drop(cols_to_drop, axis=1)
    return profile_updated_df

def add_invalid_feature(profile_updated_df, transcript_merge_df):
    """
    INPUT :
        profile_updated_df : updated profile df
        transcript_merge_df : transcript_all_df as return by transcrip_preprocessing function
    RETURN :
        profile_updated_df : updated profile df with invalid columns that  is a count how many each profile id made an invalid offer completed

    """

    profile_updated_df = profile_updated_df.copy()

    person_invalid = transcript_merge_df[transcript_merge_df.invalid == 1].person.value_counts()
    # create new feature 'invalid', how many invalid transaction made by customer (transaction that not influenced by offer)
    profile_updated_df['invalid'] = person_invalid
    profile_updated_df['invalid'] = profile_updated_df['invalid'].fillna(0)

    return profile_updated_df

def add_feature_rate_portfolio_type(profile_updated_df):
    """
    Create features
        - Total Count of the offer received, reviewed, completed for each type of portfolio (bogo, discount, and informational)
        - Rates for each offering type (bogo/discount/informational) :
            - rate_review = total reviewed / total received
            - rate_completed_reviewed = total completed / total reviewed
            - rate_completed_received = tatal completed / total received

    """
    profile_updated = profile_updated_df.copy()

    for offer in ['bogo', 'discount', 'informational']:
        received_cols = profile_updated.columns[(profile_updated.columns.str.contains('received_' + offer)) & \
                                                (~profile_updated.columns.str.contains('rate' ))].tolist()
        profile_updated[offer +'_received'] = profile_updated[received_cols].sum(axis=1).fillna(0)

        viewed_cols = profile_updated.columns[(profile_updated.columns.str.contains('viewed_' + offer)) & \
                                                (~profile_updated.columns.str.contains('rate'))].tolist()
        profile_updated[offer +'_viewed'] = profile_updated[viewed_cols].sum(axis=1).fillna(0)

        profile_updated['rate_viewed_' + offer] = (profile_updated[offer +'_viewed'] / profile_updated[offer +'_received']).fillna(0)

        if offer != 'informational':
            completed_cols = profile_updated.columns[(profile_updated.columns.str.contains('completed_' + offer)) & \
                                                (~profile_updated.columns.str.contains('rate' ))].tolist()
            profile_updated[offer +'_completed'] = profile_updated[completed_cols].sum(axis=1).fillna(0)

            profile_updated['rate_completed_viewed_' + offer] = \
                        (profile_updated[offer +'_completed'] /profile_updated[offer +'_viewed']).fillna(0)
            profile_updated['rate_completed_received_' + offer] = \
                        (profile_updated[offer +'_completed'] / profile_updated[offer +'_received']).fillna(0)

    return profile_updated

def add_feature_rate_overall(profile_updated_df):

    """
    Create Feature :
    - Total count of received, viewed, completed
    - Overall Rates :
            - rate_review = total reviewed / total received
            - rate_completed_reviewed = total completed / total reviewed
            - rate_completed_received = tatal completed / total received

    """
    profile_updated = profile_updated_df.copy()

    profile_updated['offer_received_total'] = profile_updated.bogo_received + profile_updated.discount_received + \
                                            profile_updated.informational_received

    profile_updated['offer_viewed_total'] = profile_updated.bogo_viewed + profile_updated.discount_viewed + \
                                            profile_updated.informational_viewed

    profile_updated['offer_completed_total'] = profile_updated.bogo_completed + profile_updated.discount_completed

    profile_updated['rate_offer_viewed_overall'] = \
            (profile_updated['offer_viewed_total'] / profile_updated['offer_received_total']).fillna(0)

    profile_updated['rate_offer_completed_received_overall'] = \
            (profile_updated['offer_completed_total'] / profile_updated['offer_received_total']).fillna(0)

    profile_updated['rate_offer_completed_viewed_overall'] = \
            (profile_updated['offer_completed_total'] / profile_updated['offer_viewed_total']).fillna(0)


    return profile_updated


def add_feature_rate_portfolio_id(profile_updated_df, portfolio_df):

    """
    Create Feature :
    - Rates for each offering in portfolio  :
            - rate_review = total reviewed / total received
            - rate_completed_reviewed = total completed / total reviewed
            - rate_completed_received = tatal completed / total received
    """
    profile_updated = profile_updated_df.copy()
    portfolio_updated = portfolio_preprocessing(portfolio_df)

    for offer_name in portfolio_updated.name.tolist():
        profile_updated['rate_offer_viewed_' + offer_name ] = \
            (profile_updated['offer viewed_' + offer_name] / profile_updated['offer received_' + offer_name]).fillna(0)

        if offer_name not in portfolio_updated[portfolio_updated.name.str.contains('informational')]['name'].tolist() :
            profile_updated['rate_offer_completed_viewed_' + offer_name ] = \
                (profile_updated['offer completed_' + offer_name] / profile_updated['offer viewed_' + offer_name]).fillna(0)

            profile_updated['rate_offer_completed_received_' + offer_name ] = \
                (profile_updated['offer completed_' + offer_name] / profile_updated['offer received_' + offer_name]).fillna(0)

    return profile_updated

def add_feature_transaction_completed_ratio(profile_updated_df):
    """
    Create feature transcation count to offer completed ratio

    to avoid np.inf as a result of division, a 0.1 number was added to the denominator
    """
    profile_updated = profile_updated_df.copy()

    profile_updated['transaction_completed_ratio'] = \
            profile_updated.transaction_count / (profile_updated.offer_completed_total + 0.1)

    return profile_updated

def feature_extraction(transcript_clean_df, transcript_all_df, profile_df, portfolio_df):
    """
    INPUT :
        transcript_clean_df : a clean transcript df
        transcript_all_df : transcript all df as return by transcript_preprocessing function
        portfolio_df : portfolio df
    RETURN :
        profile_updated : profile updated df with 92 features

    This function will check first whether the saved "profile_updated.csv" is available
    If not available, the next function code block will be execute, then save it.
    """

    try:
        profile_updated = load_file('data/profile_updated.csv')
        print("The profile_updated.csv file is available at local folder.")
    except:
        attributes_df = pd.DataFrame(index=generate_attributes(portfolio_df))

        for profile_id in tqdm(transcript_clean_df.person.unique()):
            subset_df = transcript_clean_df[transcript_clean_df.person == profile_id]
            subset_attributes_series = get_attributes_series(subset_df, profile_id)
            attributes_df[profile_id] = subset_attributes_series

        #parse dates became_member_on in profile_df
        profile_df = profile_parse_dates(profile_df)

        #df concatenation
        profile_updated = pd.concat([profile_df.set_index('id'),attributes_df.T ], axis=1, sort=False)

        # re-encode selected features as they should be zero instead of NaN as they did not received any offer
        profile_updated = feature_fillna(profile_updated)

        # create new FEATURES

        # add feature whether the customer made a valid or invalid transaction of offer completed
        profile_updated = add_invalid_feature(profile_updated, transcript_all_df)

        # add feature rate per portfolio type (bogo/discount/informational)
        profile_updated = add_feature_rate_portfolio_type(profile_updated)

        # add feature rate overall by event type (offer received, viewed, completed)
        profile_updated = add_feature_rate_overall(profile_updated)

        # add feature rate for individual portfolio id
        profile_updated = add_feature_rate_portfolio_id(profile_updated, portfolio_df)

        # add feature transaction to offer completed ratio
        profile_updated = add_feature_transaction_completed_ratio(profile_updated)

        #saving
        profile_updated.to_csv('data/profile_updated.csv')

    return profile_updated


"""
#######################
FEATURE PREPROCESSING
#######################
"""



def separate_profile(profile_updated_df):
    """
    INPUT :
        profile_updated_df : dataframe of profile

    RETURN :
        profile_updated_main : updated profile df for main profile, age < 100
        profile_updated_sp : updated profile df for special profile, age >= 100
    """
    # sparate data with age < 100 and age >= 100, missing value on gender and income

    #main profile
    profile_updated_main = profile_updated_df[profile_updated_df.age < 100]

    #special profile
    profile_updated_sp = profile_updated_df[profile_updated_df.age >= 100]
    profile_updated_sp = profile_updated_sp.drop(['gender', 'income', 'age'], axis=1)

    return profile_updated_main, profile_updated_sp


def encode_member_day(profile_updated_df):
    """
    INPUT :
        profile_updated_df : profile df

    RETURN :
        profile_updated_df : updated profile df, with additional col of 'member_year'

    It calculate delta days 31 dec 2018 and became_member_on date
    """
    profile_updated_df = profile_updated_df.copy()
    profile_updated_df['member_days_since'] = (datetime.datetime(2018,12,31) - profile_updated_df.became_member_on).dt.days
    profile_updated_df['member_year'] = profile_updated_df.became_member_on.dt.year.astype(str)
    profile_updated_df = profile_updated_df.drop('became_member_on', axis=1)
    return profile_updated_df

def feature_preprocessing(profile_updated_df, transcript_all_df, portfolio_df):
    """
    INPUT :
        profile_updated_df : updated profile df
        transcript_all_df : transcript df that contains both invalid and valid profile as output of transcrip_preprocessing function
        portfolio_df : portfolio df

    RETURN :
        profile_onehot_main : main profile df with one_hot enconded
        profile_onehot_sp : sp profile df with one_hot enconded

    """
    #drop features that have more than 50% missing values
    col_null = profile_updated_df.isnull().sum()
    col_null_frac = col_null / profile_updated_df.shape[0]
    cols_to_drop = col_null_frac[col_null_frac > 0.5].index.tolist()
    profile_updated_df = profile_updated_df.drop(cols_to_drop, axis=1)

    # remove row data that have age > 100 years, missing values on income and gender
    profile_updated_main, profile_updated_sp = separate_profile(profile_updated_df)

    # re-encode became_member_on to member_day (how may days since become member from 31 dec 2018)
    profile_updated_clean = encode_member_day(profile_updated_main)
    profile_updated_sp = encode_member_day(profile_updated_sp)

    # one-hot the categorical features
    profile_onehot_main = pd.get_dummies(profile_updated_clean)
    profile_onehot_sp = pd.get_dummies(profile_updated_sp)

    return profile_onehot_main, profile_onehot_sp

"""
#######################
SAVE & LOAD MODEL
#######################
"""

def save(model, filename):
    """
    This function is to save the sklearn object
    INPUT :
        model : sklearn object
        filename : filepath to saved
    RETURN : none
    """
    import pickle
    pickle.dump(model, open(filename,'wb'))

def load(filename):
    """
    This function is to load the saved sklearn object
    INPUT : filename : filepath
    RETURN : loaded sklearn object
    """
    import pickle
    return pickle.load(open(filename, 'rb'))

"""
##################################
SPOT CHECK ML Supervised Alogrithm

ref : https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
##################################
"""

import warnings
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


# create a dict of standard models to evaluate {name:object}
def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for a in alpha:
		models['lasso-'+str(a)] = Lasso(alpha=a)
	for a in alpha:
		models['ridge-'+str(a)] = Ridge(alpha=a)
	for a1 in alpha:
		for a2 in alpha:
			name = 'en-' + str(a1) + '-' + str(a2)
			models[name] = ElasticNet(a1, a2)
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	models['theil'] = TheilSenRegressor()
	# non-linear models
	n_neighbors = range(1, 21)
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsRegressor(n_neighbors=k)
	models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
	models['svml'] = SVR(kernel='linear')
	models['svmp'] = SVR(kernel='poly')
	c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for c in c_values:
		models['svmr'+str(c)] = SVR(C=c)
	# ensemble models
	n_trees = 100
	models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	models['bag'] = BaggingRegressor(n_estimators=n_trees)
	models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
	print('Defined %d models' % len(models))
	return models

# create a dict of standard models to evaluate {name:object} for MultiOutputRegressor
def get_models_multioutput(models=dict()):
    # linear models
    models['lr'] = MultiOutputRegressor(LinearRegression())
    alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models['lasso-'+str(a)] = MultiOutputRegressor(Lasso(alpha=a))
    for a in alpha:
        models['ridge-'+str(a)] = MultiOutputRegressor(Ridge(alpha=a))
    for a1 in alpha:
        for a2 in alpha:
            name = 'en-' + str(a1) + '-' + str(a2)
            models[name] = MultiOutputRegressor(ElasticNet(a1, a2))
    models['huber'] = MultiOutputRegressor(HuberRegressor())
    models['lars'] = MultiOutputRegressor(Lars())
    models['llars'] = MultiOutputRegressor(LassoLars())
    models['pa'] = MultiOutputRegressor(PassiveAggressiveRegressor(max_iter=1000, tol=1e-3))
    models['ranscac'] = MultiOutputRegressor(RANSACRegressor())
    models['sgd'] = MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3))
    models['theil'] = MultiOutputRegressor(TheilSenRegressor())
    # non-linear models
    n_neighbors = range(1, 21)
    for k in n_neighbors:
        models['knn-'+str(k)] = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=k))
    models['cart'] = MultiOutputRegressor(DecisionTreeRegressor())
    models['extra'] = MultiOutputRegressor(ExtraTreeRegressor())
    models['svml'] = MultiOutputRegressor(SVR(kernel='linear'))
    models['svmp'] = MultiOutputRegressor(SVR(kernel='poly'))
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in c_values:
        models['svmr'+str(c)] = SVR(C=c)
    # ensemble models
    n_trees = 100
    models['ada'] = MultiOutputRegressor(AdaBoostRegressor(n_estimators=n_trees))
    models['bag'] = MultiOutputRegressor(BaggingRegressor(n_estimators=n_trees))
    models['rf'] = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_trees))
    models['et'] = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=n_trees))
    models['gbm'] = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=n_trees))
    print('Defined %d models' % len(models))
    return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline

# evaluate a single model
def evaluate_model(X, y, model, folds, metric):
    # create the pipeline
    pipeline = make_pipeline(model)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    return scores

# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, folds, metric):
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric)
    except:
        scores = None
    return scores

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, folds=10, metric='accuracy'):
    results = dict()
    for name, model in models.items():
        # evaluate the model
        scores = robust_evaluate_model(X, y, model, folds, metric)
        # show process
        if scores is not None:
            # store a result
            results[name] = scores
            mean_score, std_score = np.mean(scores), np.std(scores)
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
        else:
            print('>%s: error' % name)
    return results

# print and plot the top n results
def summarize_results(results, maximize=True, top_n=10):
    # check for no results
    if len(results) == 0:
        print('no results')
        return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k,np.mean(v)) for k,v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # print the top n
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name]), std(results[name])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
    # boxplot for the top n
    pyplot.boxplot(scores, labels=names)
    _, labels = pyplot.xticks()
    pyplot.setp(labels, rotation=90)
    #pyplot.savefig('spotcheck.png')
