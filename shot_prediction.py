import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Shot prediction",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Basketball.svg/120px-Basketball.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Predicting NBA shots")

st.write(
"""
---

"""
)

@st.cache(suppress_st_warning=True)
def data_loader():
    shots = pd.read_csv('datasets/shot_logs_cleaned.csv')

    return shots

st.sidebar.header('User Input Features')

shots= data_loader()

final_margin = st.sidebar.slider('Final margin', int(shots['FINAL_MARGIN'].min()), 
                                int(shots['FINAL_MARGIN'].max()), int(shots['FINAL_MARGIN'].median()))
fig1 = st.sidebar.checkbox('Display this data', key="fig1")
shot_number = st.sidebar.slider('Shot number', int(shots['SHOT_NUMBER'].min()), 
                                int(shots['SHOT_NUMBER'].max()), int(shots['SHOT_NUMBER'].median()))
fig2 = st.sidebar.checkbox('Display this data', key="fig2")
shot_clock = st.sidebar.slider('Shot clock', float(shots['SHOT_CLOCK'].min()), 
                                float(shots['SHOT_CLOCK'].max()), float(shots['SHOT_CLOCK'].mean()))
fig3 = st.sidebar.checkbox('Display this data', key="fig3")
touch_time = st.sidebar.slider('Touch time', float(shots['TOUCH_TIME'].min()), 
                                float(shots['TOUCH_TIME'].max()), float(shots['TOUCH_TIME'].mean()))
fig4 = st.sidebar.checkbox('Display this data', key="fig4")
shot_dist = st.sidebar.slider('Shot distance', float(shots['SHOT_DIST'].min()), 
                                float(shots['SHOT_DIST'].max()), float(shots['SHOT_DIST'].mean()))
fig5 = st.sidebar.checkbox('Display this data', key="fig5")
close_def_distance = st.sidebar.slider('Closest defender distance', float(shots['CLOSE_DEF_DIST'].min()), 
                                float(shots['CLOSE_DEF_DIST'].max()), float(shots['CLOSE_DEF_DIST'].mean()))
fig6 = st.sidebar.checkbox('Display this data', key="fig6")
game_time = st.sidebar.slider('Game time', int(shots['GAME_TIME'].min()), 
                                int(shots['GAME_TIME'].max()), int(shots['GAME_TIME'].median()))
fig7 = st.sidebar.checkbox('Display this data', key="fig7")
                                
data = {'FINAL_MARGIN': final_margin,
        'SHOT_NUMBER': shot_number,
        'SHOT_CLOCK': shot_clock,
        'TOUCH_TIME': touch_time,
        'SHOT_DIST': shot_dist,
        'CLOSE_DEF_DIST': close_def_distance,
        'GAME_TIME': game_time
        }
input_df = pd.DataFrame(data, index=[0])
st.sidebar.write("""
---

---
*Class: Introduction to data science*

*Authors: Dominik Ćurić, Dario Deković, Janko Vidaković*

*Inspiration: [Bret Mehan, Predicting NBA shots, Stanford University](http://cs229.stanford.edu/proj2017/final-reports/5132133.pdf)*

*Dataset: [NBA shot logs](https://www.kaggle.com/dansbecker/nba-shot-logs)*
""")

col1, _, col2 = st.columns((2, 0.1, 1))
with col1:
    st.header("Predictions from the model")
    st.write(
    """
    Model used: **XGBoost**

    Accuracy on test set: **68%**

    *To make predictions on your data adjust the sliders in the sidebar*
    """
    )
    # Displays the user input features
    st.subheader('User Input features')
    
    st.markdown(f"""
    | Final margin | Shot number | Shot clock | Touch time | Shot distance | Closest defender distance | Game time |
    | ----- | ------ | ------ | ----- | ------ | ------ | ------ |
    | {final_margin} | {shot_number} | {shot_clock:.2f} *sec* | {touch_time:.2f} *sec* |{shot_dist:.2f} *feet* | {close_def_distance:.2f} *feet* | {game_time} *sec* |
    """)

    # Reads in saved classification model
    load_clf:XGBClassifier = pickle.load(open('models/shots_clsf.pkl', 'rb'))
    load_scaler:StandardScaler = pickle.load(open('models/shots_scaler.pkl', 'rb'))

    input_df = load_scaler.transform(input_df)

    # Apply model to make predictions
    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df)


    st.subheader('Prediction')
    shot_result = np.array(['Shot missed','Shot made'])
    class_name = shot_result[prediction][0]
    st.markdown(
        f"""
        | Model prediction |
        | ---------------- |
        | {class_name} |
        """
        )

def plot_dist(x: str, xlabel:str) -> plt.axes:
    st.write(xlabel)
    f = sns.displot(data=shots, x=x, kind='kde', hue='SHOT_RESULT')
    plt.xlabel(xlabel)
    plt.ylabel("")
    f._legend.set_title("Shot result")
    st.pyplot(f)

with col2:
    st.header("Graphs from the data")
    if fig1:
        plot_dist("FINAL_MARGIN", "Final margin")
    if fig2:
        plot_dist("SHOT_NUMBER", "Shot number")
    if fig3:
        plot_dist("SHOT_CLOCK", "Shot clock")
    if fig4:
        plot_dist("TOUCH_TIME", "Touch time")
    if fig5:
        plot_dist("SHOT_DIST", "Shot distance")
    if fig6:
        plot_dist("CLOSE_DEF_DIST", "Closest defender distance")
    if fig7:
        plot_dist("GAME_TIME", "Game time")
    
    if not(fig1 or fig2 or fig3 or fig4 or fig5 or fig6 or fig7):
        st.write(
        """
        *Select data points you want to visualize in the sidebar*
        """
        )



