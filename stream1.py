import streamlit as st
import joblib
model=joblib.load(r"C:\Users\VIVEKANANDA D\OneDrive\Documents\vivek\logistic-regression")
scaler=joblib.load(r"C:\Users\VIVEKANANDA D\OneDrive\Documents\vivek\scaler")
import numpy as np

def predict():
  pre_data=[gender1(gender)+chestpain1(chestpain)+restingecg1(restingecg)+agena(exercise)+slope1(slope)+fastingbs1(fastingbs)]
  h=np.append(standardize(),pre_data)
  result=model.predict([h])
  return result

def chestpain1(chestpain):
    if chestpain.upper()=='ATA':
        chestpain=[1.0,0.0,0.0]
    elif chestpain.upper()=='NAP':
         chestpain=[0.0,1.0,0.0]
    elif chestpain.upper()=='TA':
         chestpain=[0.0,0.0,1.0]
    elif chestpain.upper()=='ASY':
         chestpain=[0.0,0.0,0.0]
    return chestpain
def slope1(slope):
    if slope.lower()=='up':
        slope=[0.0,1.0]
    elif slope.lower()=='flat':
        slop=[1.0,0.0]
    elif slope.lower()=='down':
        slope=[0.0,0.0]
    return slope

def restingecg1(resting_ecg):
    if resting_ecg.lower()=='normal':
        resting_ecg=[1.0,0.0]
    elif resting_ecg.lower()=='st':
        resting_ecg=[0.0,1.0]
    elif resting_ecg.lower()=='lvh':
        resting_ecg=[0.0,0.0]
    return resting_ecg
def fastingbs1(fastingbs):
    if fastingbs.lower()=='low':
        return [0]
    elif fastingbs.lower()=='high':
        return [1]
def agena(exercise):
    if exercise.lower()=='yes':
        return [1]
    elif exercise.lower()=='no':
        return [0]
    

def standardize():
    c=scaler.transform([[age,restingbp,cholestrol,maxhr,oldpeak]])
    return c
def gender1(gender):
  if gender.lower()=='male':
    return [1]
  elif gender.lower()=='female':
    return [0]



def main():
    st.title("Heart Disease Prediction")
    html_temp="""<body><style>{background-color:black;}</style></body>
    """
    global age,gender,chestpain,restingbp,cholestrol,fastingbs,restingecg,maxhr,exercise,oldpeak,slope
    st.markdown(html_temp,unsafe_allow_html=True)
    age=st.number_input("AGE",min_value=0 ,max_value=100)
    gender=st.radio('select gender',("male","female"),horizontal=True)
    chestpain=st.selectbox("chose your chest pain type",('ATA','NAP','TA','ASY'))
    restingbp=st.slider("select your resting bp",min_value=10,max_value=200)
    cholestrol=st.number_input("CHOLESTROL",min_value=10,max_value=500)
    fastingbs=st.radio("select fasting blood sugar",('low','high'),horizontal=True)
    restingecg=st.selectbox("chose your resting ecg type",("NORMAL","LVH","ST"))
    maxhr=st.slider("select your maximum heart rate",min_value=35,max_value=250)
    exercise=st.radio('do you have angina ',("yes","no"),horizontal=True)
    oldpeak=st.number_input("enter your old peak value",min_value=-6.0,max_value=7.0)
    slope=st.selectbox("chose your slope type",('UP','DOWN','FLAT'))
    
    if(st.button("predict")):
        x=predict()
        if x==[1.0]:
          y="you may be attacked by Heart disease consult the doctor"
        else :
          y="you don't have risk of the Heart disease"
        
        st.text_area(label="Output Data:", value=y, height=100)
        



if __name__=='__main__':
    main()
