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
    portfolio_df = portfolio_df.copy()
    portfolio_df['name'] = portfolio_df.offer_type.astype(str) + "_" + portfolio_df.difficulty.astype(str) +\
                      "_" + portfolio_df.reward.astype(str) + \
                     "_" + portfolio_df.duration.astype(str)
    return portfolio_df

def one_hot_channels(portfolio_df):
    portfolio_df = portfolio_df.copy()
    channels = ['web', 'email', 'mobile', 'social']

    for channel in channels:
        portfolio_df[channel] = portfolio_df.channels.apply(lambda x: channel in x)*1

    #drop channels column
    portfolio_df = portfolio_df.drop('channels', axis=1)
    return portfolio_df

def portfolio_preprocessing(portfolio_df):
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
    profile_df = profile_df.copy()

    #convert to string
    profile_df['became_member_on'] = profile_df.became_member_on.apply(lambda x: str(x))
    #convert to datetime format
    profile_df['became_member_on'] = pd.to_datetime(profile_df.became_member_on)
    return profile_df

""" Transcript preprocessing """

def encode_offer_id(x):
    try :
        return x['offer id']
    except:
        return x['offer_id']

def transcript_encoded(transcript_df):
    """
    To encode column :
        - event : received, reviewed, completed
        - value : offer_id, amount
    """

    transcript_encoded = transcript_df.copy()
    transcript_encoded['offer_id'] = transcript_df[transcript_df.event.isin(['offer received', 'offer viewed', 'offer completed'])]\
                                                             .value.apply(encode_offer_id)
    transcript_encoded['amount'] =  transcript_df[transcript_df.event.isin(['transaction'])].value.apply(lambda x: x['amount'])
    return transcript_encoded

def merge_transcript_profile(transcript_df, profile_df):
    profile_df = profile_parse_dates(profile_df)
    transcript_encoded_df = transcript_encoded(transcript_df)
    transcript_profile_df = pd.merge(transcript_encoded_df, profile_df, left_on=['person'],
                                      right_on = ['id'], how ='left')
    transcript_profile_df = transcript_profile_df.drop(['id'], axis=1)

    return transcript_profile_df

def merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df):
    portfolio_df = portfolio_preprocessing(portfolio_df)
    transcript_profile_df = merge_transcript_profile(transcript_df, profile_df)
    transcript_profile_porto = pd.merge(transcript_profile_df, portfolio_df, left_on = 'offer_id', right_on ='id', how='left').drop('id', axis=1)
    #parse date became_member_on
    transcript_profile_porto = profile_parse_dates(transcript_profile_porto)

    return transcript_profile_porto

def find_invalid_index(transcript_df, profile_df, portfolio_df):
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
    try:
        transcript_clean_df = load_file('transcript_clean.csv')
        print("The transcript_clean.csv and transcript_merge.csv file are available at local folder")

    except:
        transcript_clean_df = merge_transcript_profile_portfolio(transcript_df, profile_df, portfolio_df)
        invalid_index = find_invalid_index(transcript_df, profile_df, portfolio_df)

        #marking invalid in transcript_merge_df
        transcript_clean_df.loc[transcript_clean_df.index.isin(invalid_index),"invalid"] = 1
        transcript_clean_df.loc[~transcript_clean_df.index.isin(invalid_index),"invalid"] = 0

        #saving df
        transcript_clean_df.to_csv('transcript_clean.csv')

    return transcript_clean_df

def transcript_preprocessing(transcript_df, profile_df, portfolio_df):
    transcript_all_df = transcript_cleaning(transcript_df, profile_df, portfolio_df)
    transcript_valid_df = transcript_all_df[transcript_all_df.invalid == 0]
    return transcript_valid_df, transcript_all_df

def load_file(filepath):
    df_clean = pd.read_csv(filepath)
    df_clean = df_clean.set_index(df_clean.columns[0])
    df_clean = profile_parse_dates(df_clean)
    return df_clean


################################
""" FEATURES EXTRACTION """
################################

def get_response_time(df, profile_id):
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
    avg_spending = df.amount.mean()
    transaction_count = df.amount.count()
    sum_spending = df.amount.sum()

    spending_series = pd.Series([avg_spending, transaction_count, sum_spending], index=["avg_spending", "transaction_count", 'sum_spending'], name=profile_id)
    return spending_series

# def get_offer_typ_series(df, profile_id):
#     offer_typ_series = df.offer_type.value_counts()
#     offer_typ_series.name = profile_id
#     return offer_typ_series

def get_event_typ_series(df, profile_id):
    event_typ_series = (df.event + "_" + df.name).value_counts()
    event_typ_series.name = profile_id
    return event_typ_series

def get_attributes_series(df, profile_id):
    #offer_typ_series = get_offer_typ_series(df, profile_id)
    event_typ_series = get_event_typ_series(df, profile_id)
    response_time_series = get_response_time(df, profile_id)
    spending_series = get_spending_series(df, profile_id)
    attributes_series = pd.concat([event_typ_series, response_time_series, spending_series], axis=0)
    return attributes_series

def generate_attributes(portfolio_df):
    portfolio_df = portfolio_preprocessing(portfolio_df)
    events = ['offer received', 'offer viewed', 'offer completed']
    portfolio_names = [ event +"_"+ name for event in events for name in portfolio_df.name.tolist() ]
    #offer_type_list = portfolio_df.offer_type.unique().tolist()
    response_time_attributes = [name +'_' +'response_time_avg' for name in portfolio_df.name.tolist() ]
    attributes = portfolio_names + response_time_attributes + ["avg_spending", "transaction_count", "sum_spending"]
    return attributes

def feature_fillna(profile_updated_df):
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
    try:
        profile_updated = load_file('profile_updated.csv')
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
        profile_updated.to_csv('profile_updated.csv')

    return profile_updated


"""
#######################
FEATURE PREPRPOCESSING
#######################
"""

def add_invalid_feature(profile_updated_df, transcript_merge_df):
    profile_updated_df = profile_updated_df.copy()

    person_invalid = transcript_merge_df[transcript_merge_df.invalid == 1].person.value_counts()
    # create new feature 'invalid', how many invalid transaction made by customer (transaction that not influenced by offer)
    profile_updated_df['invalid'] = person_invalid
    profile_updated_df['invalid'] = profile_updated_df['invalid'].fillna(0)

    return profile_updated_df

def encode_member_day(profile_updated_df):
    profile_updated_df = profile_updated_df.copy()
    profile_updated_df['member_day'] = (datetime.datetime(2018,12,31) - profile_updated_df.became_member_on).dt.days
    profile_updated_df = profile_updated_df.drop('became_member_on', axis=1)
    return profile_updated_df

"""
#######################
SAVE & LOAD MODEL
#######################
"""

def save(model, filename):
    import pickle
    pickle.dump(model, open(filename,'wb'))

def load(filename):
    import pickle
    return pickle.load(open(filename, 'rb'))
