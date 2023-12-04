"""
Prepared By: Clark P. Necciai Jr.
For: Dr. Gregory Lyon - Programming II
Task: Logistic Regression Model - LinkedIn User

To run application:
1. Open terminal w/ streamlit-linkedin-clark environment active
2. Navigate to corresponding directory
3. Input: "streamlit run streamlit-appip.py" command

To exit:
In terminal, `ctrl + c`
"""

#Import necessary libraries
import streamlit as st 
import pandas as pd
import numpy as np
import warnings
import plotly.express as px 
import time
warnings.filterwarnings("ignore")

#Import Data
s = pd.read_csv("social_media_usage.csv")

#----------------DataFrame Prep/Data Type Transformation----------------# 

#If x == 1, return x=1 else return x=0
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)

#Lambda func for correct check for female (female == 2 in social_media_usage_README.txt)
gender_female = lambda g: 1 if g == 2 else 0

#Create `ss` DataFrame with Specified Variables
ss = pd.DataFrame({
    'sm_li': s['web1h'].apply(clean_sm),
    'income':  np.where(s['income'] > 9, np.NaN, s['income']),
    'education': np.where(s['educ2'] > 8, np.NaN, s['educ2']),
    'parent': s['par'].apply(clean_sm), # No missing
    'married': s['marital'].apply(clean_sm), #No missing
    'female': s['gender'].apply(gender_female), #No missing
    'age': np.where(s['age'] > 98, np.NaN, s['age']),
})

#Remove Decimal with Int64 then convert to Categorical
ss['sm_li'] = ss['sm_li'].astype('str')
ss['income'] = pd.to_numeric(ss['income'], errors='coerce').astype('Int64').astype('category')
ss['education'] = pd.to_numeric(ss['education'], errors='coerce').astype('Int64').astype('category')
ss['parent'] = ss['parent'].astype('category')
ss['married'] = ss['married'].astype('category')
ss['female'] = ss['female'].astype('category')

#Drop missing Values
ss.dropna(inplace=True)

#----------------Logistic Regression Model Specification----------------#

#Assign Target and Feature Set
y = ss['sm_li']
X = ss.drop('sm_li', axis = 1)

#80/20 Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8675309)

#Instantiate Logistic Regression Model and Fit
from sklearn.linear_model import LogisticRegression as LogReg
logistic_model = LogReg(random_state=0, class_weight='balanced').fit(X_train, y_train)

#----------------STREAMLIT OUTPUT----------------#

st.title("Who's A User?")
st.markdown("#### A Machine Learning Python Application to Predict LinkedIn Usage") 
st.markdown("###### Prepared by: Clark P. Necciai")
st.divider()
st.markdown("\tSometimes, is can seem as though the whole world uses \
    **LinkedIn!**")

st.markdown("Of course, while it is a popular social networking \
    site and useful for marketing purposes, not *everyone* uses\
    it. Through a granular analysis of up-to-date Pew usage data, \
    our application now provides an interactive environment to \
    produce real-time predictions as to whether or not someone \
    uses LinkedIn!")

st.markdown("Making a prediction using our application returns not only \
    the predicted classification, but also the probability used in making \
    that classification!")

st.markdown("Let's see an example of predicting LinkedIn usage for a high income, \
    high education, married, non-parent female who is 42 years old and see \
    what our Logistic Model predicts!")

video_file = open('streamlit-usage.mp4', 'rb')
video = video_file.read()
st.divider()
st.video(video)
st.divider()
st.markdown("### ~ Application Below ~")
st.markdown("###### Choose options then click [Calculate Prediction!] to retrieve results!")


col_i, col_e = st.columns([1,1])

#Income Prompt
with col_i: 
    answer_i = st.selectbox( "Select Income Range :moneybag: :money_mouth_face:", 
                            ( "1 - Less than $10,000", "2 - $10,000 to under $20,000",
                                                "3 - $20,000 to under $30,000", "4 - $30,000 to under $40,000",
                                                "5 - $40,000 to under $50,000", "6 - $50,000 to under $75,000", 
                                                "7 - $75,000 to under $100,000", "8 - $100,000 to under $150,000",
                                                "9 - $150,000 or more!"))
    
    #Income Variable Assignment
    match answer_i:
        case "1 - Less than $10,000":
            user_income = 1
        case "2 - $10,000 to under $20,000":
            user_income = 2
        case "3 - $20,000 to under $30,000":
            user_income = 3
        case "4 - $30,000 to under $40,000":
            user_income = 4
        case "5 - $40,000 to under $50,000":
            user_income = 5
        case "6 - $50,000 to under $75,000":
            user_income = 6
        case "7 - $75,000 to under $100,000":
            user_income = 7
        case "8 - $100,000 to under $150,000":
            user_income = 8
        case "9 - $150,000 or more!":
            user_income = 9

#Education Prompt
with col_e: 
    answer_e = st.selectbox("Select Education Level :mortar_board:",
                                ("1 - Less than high school (Grades 1-8 or no formal schooling)",
                                "2 - High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                                "3 - High school graduate (Grade 12 with diploma or GED certificate)",
                                "4 - Some college, no degree (includes some community college)",
                                "5 - Two-year associate degree from a college or university Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                                "6 - Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                                "7 - Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                                "8 - Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"))

    #Education Variable Assignment
    match answer_e:
        case "1 - Less than high school (Grades 1-8 or no formal schooling)":
            user_education = 1
        case "2 - High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
            user_education = 2
        case "3 - High school graduate (Grade 12 with diploma or GED certificate)":
            user_education = 3
        case "4 - Some college, no degree (includes some community college)":
            user_education = 4
        case "5 - Two-year associate degree from a college or university Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
            user_education = 5
        case "6 - Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
            user_education = 6
        case "7 - Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
            user_education = 7
        case "8 - Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
            user_education = 8


#Present Radio Buttons Inline
col1, col2, col3 = st.columns([1,1,1])

#Parent Prompt and Variable Assignment
with col1:     
    answer_p = st.radio("Is a parent? :baby:", ["Yes", "No"])
    user_parent = 1 if answer_p == "Yes" else 0

#Married Prompt and Variable Assignment
with col2: 
    answer_m = st.radio("Is married? :ring:", ["Yes", "No"])
    user_married = 1 if answer_m == "Yes" else 0

#Female Prompt and Variable Assignment
with col3:
    answer_g = st.radio("Is female? :woman:", ["Yes", "No"])
    user_gender = 1 if answer_g == "Yes" else 0

#Age Prompt and Variable Assignment
user_age = st.slider("Select age :child::adult::older_adult:", int(ss["age"].min()), 98)

col_l, col_p, col_r = st.columns([1,1,1])

#Prompt User to Make Prediction
with col_p:
    user_make_prediction = st.button("Calculate Prediction!")

    #Show output 
    if user_make_prediction == True: 

        with st.spinner(text = "Making a prediction..."):
            time.sleep(1.5)
            
        #Insert User Input into Dataframe
        pred_obs = pd.DataFrame({
            'income' : user_income,
            'education' : user_education,
            'parent' : user_parent,
            'married' : user_married, 
            'female' : user_gender,
            'age' : user_age}, index=[0])

        y_pred = logistic_model.predict(pred_obs)

        #Probability of LinkedIn User
        probability_user = logistic_model.predict_proba(pred_obs)[0][1]
        
        #Format probability
        ans = "{:.2f}%".format(probability_user*100)
        
        st.divider()
        
        #Display Classification Result
        color = ""
        if (probability_user*100) >= 50.0:
            st.write("Our model would classify this person as :blue[LinkedIn User]!")
        else:
            st.write("Our model would classify this person as :red[Not LinkedIn User]!")
        
        #Display Probability Result used in determining classification
        st.metric(label = "Probability this person uses LinkedIn:", value=ans)

        prob_ans = dict(
            labels = ["LinkedIn User",
                      "Not LinkedIn User"], 
            values = [probability_user * 100, (1 - probability_user) * 100],
        )
        
        #Donut Plot Displaying Competing Probabilities
        fig = px.pie(prob_ans, 
                     values = 'values',
                     names = 'labels', 
                     hover_data=None,
                     hole=.60,
                     category_orders={"labels": ["LinkedIn User", "Not LinkedIn User" ] })
        
        fig.update_layout(
            title = "Model Outcome Probabilities",
            margin = {'l' : 0, 'r' : 450, 'b' : 175, 't' : 65}
            
        )
        
        #Display
        st.plotly_chart(fig)
        
