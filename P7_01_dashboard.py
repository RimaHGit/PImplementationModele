

# Prérequis exécuter le notebook pour la génération des données et du modèle (dans le même répértoire)
# Déploiement de l'application (installation streamlit + SHAP + lancement de l'application)
# pip install streamlit
# pip install shap
# streamlit run "P7_01_dashboard.py"

import streamlit as st
import pandas as pd
#import shap
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Bank Customer Prediction App
This app predicts the **Bank customer Prediction**!
""")
st.write('---')

#pkl_filename = "pickle_bestmodel_bank.pkl"
#scaler_filename = "pickle_scaler_bank.pkl"
#imputer_filename = "pickle_imputer_bank.pkl"

#pickle_model = pickle.load(open(pkl_filename, 'rb'))

pickle_model  = RandomForestClassifier(n_estimators = 200, max_depth = 15, bootstrap = True, min_samples_leaf = 4, min_samples_split = 10)


trainX = pd.read_csv('trainX.csv')
testX = pd.read_csv('testX.csv')
trainy = pd.read_csv('trainy.csv')
testy = pd.read_csv('testy.csv')

pickle_model.fit(trainX, trainy)

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

app_train=0
choosendf=0

if uploaded_file is not None:
    st.write('**---Choix par Import CSV---**')
    app_train = pd.read_csv(uploaded_file, sep = ",")
    app_train = app_train.fillna(0)
    maked_choice = st.sidebar.selectbox('Select your Customer (ID):', app_train['SK_ID_CURR'])
    st.write('Choosen Customer : ', maked_choice)
    choosendf=app_train.loc[app_train['SK_ID_CURR'] == maked_choice]
    choosendf = choosendf.loc[:, ~choosendf.columns.str.contains('^Unnamed')]
    choosendf = choosendf.loc[:, ~choosendf.columns.str.contains('^TARGET')]
    st.write('Selected Row: ', choosendf.shape)
    st.write(choosendf)
    st.write('Of Head: ')
    st.write(app_train)
    X = pd.DataFrame(app_train, columns=app_train.columns)
    Y = pd.DataFrame(app_train, columns=["TARGET"])
    predictions_choosen1 = pickle_model.predict_proba(choosendf)[:, 1]
    st.write(predictions_choosen1)
    st.write('DECISION :')
    if (predictions_choosen1<0.34):
     st.write('Prêt accrodé')
    else :
     st.write('Prêt refusé')
    st.write("--------------------------------------------------------------------------------------")

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('2. Upload your Input Parameters')
st.write('**--- Globalité des données de la bank ---**')
st.write('Training Head: ')
app_train = pd.read_csv("./df_selected_features_100.csv", sep = ",")
st.write(app_train.head())
X = pd.DataFrame(app_train, columns=app_train.columns)
Y = pd.DataFrame(app_train, columns=["TARGET"])

@st.cache(suppress_st_warning=True)
def computeGeneralData(pickle_model, app_train):
 # Make predictions on the test data
 predictions = pickle_model.predict_proba(testX)[:, 1]
 df1 = pd.DataFrame(testy)
 df2 = pd.DataFrame({'SCORE' : predictions, 'DECISION' : predictions}, index = df1.index )
 df2.loc[(df2.DECISION < 0.34),'DECISION']='Prêt accrodé'
 df2.loc[(df2.DECISION != 'Prêt accrodé'),'DECISION']='Prêt refusé'
 st.header('Prediction of TARGET')
 st.write(df2)
 st.write('---')

computeGeneralData(pickle_model,app_train)

st.write("--------------------------------------------------------------------------------------")

def user_input_features():
    EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', 0.03, 0.83, 0.40)
    EXT_SOURCE_2 = st.sidebar.slider('EXT_SOURCE_2', 0.03, 0.83, 0.40)
    DAYS_EMPLOYED = st.sidebar.slider('DAYS_EMPLOYED', -12203.00, -100.00, -6000.00)
    AMT_REQ_CREDIT_BUREAU_YEAR = st.sidebar.slider('AMT_REQ_CREDIT_BUREAU_YEAR', 0.00, 8.00, 4.00)
    OWN_CAR_AGE = st.sidebar.slider('OWN_CAR_AGE', 1.00, 30.00, 15.00)
    OBS_60_CNT_SOCIAL_CIRCLE = st.sidebar.slider('OBS_60_CNT_SOCIAL_CIRCLE', 0.00, 10.00, 5.00)
    OBS_30_CNT_SOCIAL_CIRCLE = st.sidebar.slider('OBS_30_CNT_SOCIAL_CIRCLE', 0.00, 10.00, 5.00)
    AMT_ANNUITY = st.sidebar.slider('AMT_ANNUITY', 6750.00, 67500.00, 30000.00)
    DAYS_REGISTRATION = st.sidebar.slider('DAYS_REGISTRATION', -14166.00, -51.00, -7000.00)
    DAYS_LAST_PHONE_CHANGE = st.sidebar.slider('DAYS_LAST_PHONE_CHANGE', -2800.00, 0.00, -1400.00)
    EXT_SOURCE_1 = st.sidebar.slider('EXT_SOURCE_1',0.06, 0.88, 0.44)
    REGION_POPULATION_RELATIVE = st.sidebar.slider('REGION_POPULATION_RELATIVE', 0.00, 0.07, 0.04)
    AMT_INCOME_TOTAL = st.sidebar.slider('AMT_INCOME_TOTAL', 45000.00, 990000.00, 500000.00)
    HOUR_APPR_PROCESS_START = st.sidebar.slider('HOUR_APPR_PROCESS_START', 4.00, 20.00, 12.00)
    AMT_GOODS_PRICE = st.sidebar.slider('AMT_GOODS_PRICE', 67500.00, 1800000.00, 900000.00)
    DAYS_BIRTH = st.sidebar.slider('DAYS_BIRTH', 7712.00, 25010.00, 15000.00)
    DAYS_ID_PUBLISH = st.sidebar.slider('DAYS_ID_PUBLISH', -6021.00, -20.00, -3000.00)
    AMT_CREDIT = st.sidebar.slider('AMT_CREDIT', 67500.00, 2013840.00, 1000000.00)
    SK_ID_CURR = st.sidebar.slider('SK_ID_CURR', 103679.00, 448119.00, 250000.00)
    data = {'EXT_SOURCE_3': EXT_SOURCE_3,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'DAYS_BIRTH': DAYS_BIRTH,
            'DAYS_EMPLOYED': DAYS_EMPLOYED,
            'DAYS_ID_PUBLISH': DAYS_ID_PUBLISH,
            'AMT_ANNUITY': AMT_ANNUITY,
            'DAYS_REGISTRATION': DAYS_REGISTRATION,
            'DAYS_LAST_PHONE_CHANGE': DAYS_LAST_PHONE_CHANGE,
            'AMT_CREDIT': AMT_CREDIT,
            'SK_ID_CURR': SK_ID_CURR,
            'EXT_SOURCE_1': EXT_SOURCE_1,
            'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
            'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
            'HOUR_APPR_PROCESS_START': HOUR_APPR_PROCESS_START,
            'AMT_REQ_CREDIT_BUREAU_YEAR': AMT_REQ_CREDIT_BUREAU_YEAR,
            'OWN_CAR_AGE': OWN_CAR_AGE,
            'OBS_60_CNT_SOCIAL_CIRCLE': OBS_60_CNT_SOCIAL_CIRCLE,
            'OBS_30_CNT_SOCIAL_CIRCLE': OBS_30_CNT_SOCIAL_CIRCLE
            }
    features = pd.DataFrame(data, index=[0])
    return features

st.write('**--- Nouveau Client (Slider) ---**')
df = user_input_features()
predictions_choosen = pickle_model.predict_proba(df)[:, 1]
st.write('Selected Row: ', df.shape)
st.write(df)
st.write(predictions_choosen)

st.write('DECISION :')
if (predictions_choosen<0.34):
 st.write('Prêt accrodé')
else :
 st.write('Prêt refusé')

st.write("--------------------------------------------------------------------------------------")
st.write('**--- Comparison between Slider and Général DATA Bank ---**')

st.write(df)
app_train2=app_train
app_train2 = app_train2.loc[:, ~app_train2.columns.str.contains('^Unnamed')]
app_train2 = app_train2.loc[:, ~app_train2.columns.str.contains('^TARGET')]

st.write(app_train2.describe(include = 'all'))

st.write("--------------------------------------------------------------------------------------")
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
#explainer = shap.TreeExplainer(pickle_model)
#shap_values = explainer.shap_values(df)

#st.header('Feature Importance')
#plt.title('Feature importance based on SHAP values')
#shap.summary_plot(shap_values, df)
#st.pyplot(bbox_inches='tight')
#st.write('---')
