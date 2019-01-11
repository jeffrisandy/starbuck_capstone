
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Anaconda distribution of Python version 3.6 or later
2. [eli5](https://eli5.readthedocs.io/en/latest/)
3. [PDPBox](https://pdpbox.readthedocs.io/en/latest/)

## Project Motivation<a name="motivation"></a>

Understanding our customers is the key providing them a good service and sustain a profitable business. To understand them well, we need to pay attention on their purchase behaviour. One way we can collect and analyse their purchasing behaviour through an app, then identify their needs based on demographics.

The Starbucks [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Capstone challenge data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Periodically, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). However, some users might not receive any offer during certain weeks.
Using the data, I aim to :
- Gain understanding what types of customer characteristics and demographics are there.
- What offer should be sent to each customer ?
- How well can we predict customer response to offer ?

An unsupervised machine learning model with K-Means algorithm is used to cluster the customers. The number of clusters is chosen with 2 metrics - the higher Silhouette score and the lower Inertia / SSE value.

A supervised machine learning model with regression is used to predict the customer response rate to offer. The used metrics are mean squared error (MSE) and coefficent of determination (R^2)


## File Descriptions <a name="files"></a>

### Notebooks
There  are 3 notebooks available:
- part 1 about data understanding and cleaning process
- part 2 about EDA, feature preprocessing and unsupervised Machine Learning with KMeans.
- part 3 about supervised machine learning model to predict how well customer response to offer

Markdown cells were used to assist in walking through the thought process for individual steps.

### Helpers function
There is a `helpers.py` as utilities and also to extract features from the available data.

### Dataset
The dataset is in folder data, contained in three files:

- `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
- `profile.json` - demographic data for each customer
- `transcript.json` - records for transactions, offers received, offers viewed, and offers completed
Here is the schema and explanation of each variable in the files:

`portfolio.json`
- `id` (string) - offer id
- `offer_type` (string) - type of offer ie BOGO, discount, informational
- `difficulty` (int) - minimum required spend to complete an offer
- `reward` (int) - reward given for completing an offer
- `duration` (int) -
- `channels` (list of strings)

`profile.json`
- `age` (int) - age of the customer
- `became_member_on` (int) - date when customer created an app account
- `gender` (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- `id` (str) - customer id
- `income` (float) - customer's income

`transcript.json`
- `event` (str) - record description (ie transaction, offer received, offer viewed, etc.)
- `person` (str) - customer id
- `time` (int) - time in hours. The data begins at time t=0
- `value` - (dict of strings) - either an offer id or transaction amount depending on the record


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@jeffrisandy/investigating-starbucks-customers-segmentation-using-unsupervised-machine-learning-10b2ac0cfd3b). Github is available at [here](https://github.com/jeffrisandy/starbuck_capstone)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity and Starbucks for the data as part of final capstone project of [Udacity Data Science Nanodegree.](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
