#Author: Ravish Garg
#Customer Engineer, Data Specialist

from __future__ import print_function
from inspect import FullArgSpec
from logging import disable
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import pandasql as ps
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os, io, cv2, sys, re
from datetime import datetime
import exifread
import base64
from google.cloud import firestore, storage
from google.cloud import vision
import gmaps
import googlemaps
import pyrebase
from tensorflow.python.util.tf_export import api_export
from flask import Flask, render_template
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
import SessionState
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

st.set_page_config(page_title='GCP DEMO - MDS4IC', page_icon=':smiley:')

API_KEY= << ... PUT YOUR OWN API KEYS ...>>
googlemaps=googlemaps.Client(key=API_KEY)
gmaps.configure(api_key=API_KEY)

firebaseConfig = {
 << ... PUT YOUR OWN ENVIRONMENT VARIABLES ...>>
    }

page_bg_img = """
<style>
.reportview-container {
background-image: linear-gradient(rgba(0, 0, 0, 0.7),
                       rgba(0, 0, 0, 0.7)),url("https://images.unsplash.com/photo-1576091358783-a212ec293ff3?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1825&q=80");
background-size: cover;
}
.sidebar .sidebar-content {
   display: flex;
   align-items: center;
   justify-content: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

def login():
    email=st.text_input('Enter registered E-Mail')
    password = st.text_input('Enter Password',type="password")
    state = None
    try:
        state = auth.sign_in_with_email_and_password(email, password)
    except:
        st.write("If you are not able to Login... Check with your Health officer.")
    return state

def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_readable_time(mytime):
    return datetime.fromtimestamp(mytime).strftime('%Y-%m-%d %H:%M:%S')

def findHorizontalLines(img):
    img = cv2.imread(img) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
    lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    return lineLocations

def pageSegmentation1(img, w, df_SegmentLocations):
    img = cv2.imread(img) 
    im2 = img.copy()
    segments = []
    for i in range(len(df_SegmentLocations)):
        y = df_SegmentLocations['SegmentStart'][i]
        h = df_SegmentLocations['Height'][i]
        cropped = im2[y:y + h, 0:w] 
        segments.append(cropped)      
    return segments

def CloudVisionTextExtractor(handwritings):
    # convert image from numpy to bytes for submittion to Google Cloud Vision
    _, encoded_image = cv2.imencode('.png', handwritings)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    # feed handwriting image segment to the Google Cloud Vision API
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)
    return response

def getTextFromVisionResponse(response):
    texts = []
    for page in response.full_text_annotation.pages:
        for i, block in enumerate(page.blocks):  
            for paragraph in block.paragraphs:       
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    texts.append(word_text)
    return ' '.join(texts)

def ocr(img_name):
    os.chdir(r'/Users/ravishgarg/Documents/Projects/Streamlit/OCR/')
    #img_uri = '/Users/ravishgarg/Documents/Projects/Streamlit/OCR/Prescrip.png'
    img_uri = img_name
    lineLocations = findHorizontalLines(img_uri)

    df_lineLocations = pd.DataFrame(lineLocations.sum(axis=1)).reset_index()
    df_lineLocations.columns = ['rowLoc', 'LineLength']
    df_lineLocations[df_lineLocations['LineLength'] > 0]
    df_lineLocations['line'] = 0
    df_lineLocations['line'][df_lineLocations['LineLength'] > 100] = 1
    df_lineLocations['cumSum'] = df_lineLocations['line'].cumsum()

    query = '''
    select row_number() over (order by cumSum) as SegmentOrder
    , min(rowLoc) as SegmentStart
    , max(rowLoc) - min(rowLoc) as Height
    from df_lineLocations
    where line = 0
    --and CumSum !=0
    group by cumSum
    '''
    df_SegmentLocations  = ps.sqldf(query, locals())
    w = lineLocations.shape[1]
    segments = pageSegmentation1(img_uri, w, df_SegmentLocations)
    handwritings = segments[1]
    response = CloudVisionTextExtractor(handwritings)
    handwrittenText = getTextFromVisionResponse(response)

    handwritings0 = segments[0]
    response = CloudVisionTextExtractor(handwritings0)
    handwrittenText0 = getTextFromVisionResponse(response)

    return handwrittenText0, handwrittenText

def manualentry():
    st.write("## Enter your prescription details Manually")

def healthdetails():
    message = ""
    person_coord = pd.DataFrame()
    st.write("#### Upload your prescription and get the medicine availability details in real-time. ")
    st.write("\n\n\n")
    st.write("")
    ocr_text = None
    ocr_text0 = None
    hcdetails = st.text_input('Enter your Health-Card Details',type="password")
    username = st.text_input('Enter your Full Name')
    pincode = st.text_input('Enter Pin Code')
    image_file = st.file_uploader("Upload Prescription",type=["png","jpg","jpeg"])
    if image_file is not None:
        img = load_image(image_file)
        st.image(img)

    c1,c2 = st.beta_columns(2)
    with c1:
        if image_file is not None:
            file_details = {"Filename":image_file.name,"FileSize":image_file.size,"FileType":image_file.type}
            #st.write(file_details)
            statinfo = os.stat(image_file.readable())
            # st.write(statinfo)
            stats_details = {"Accessed_Time":get_readable_time(statinfo.st_atime),"Creation_Time":get_readable_time(statinfo.st_ctime),
            "Modified_Time":get_readable_time(statinfo.st_mtime)}
            #st.write(stats_details)
            file_details_combined = {"Filename":image_file.name,
            "FileSize":image_file.size,"FileType":image_file.type,"Accessed_Time":get_readable_time(statinfo.st_atime),"Creation_Time":get_readable_time(statinfo.st_ctime),
            "Modified_Time":get_readable_time(statinfo.st_mtime)}
            df_file_details = pd.DataFrame(list(file_details_combined.items()),columns=["Meta Tags","Value"])
            st.dataframe(df_file_details)
            st.write("Pincode Entered: ",pincode)
            st.write("Name Entered: ",username)
            st.write("Healthcard: ",hcdetails)
            # visualization : plotting on google maps
            geocode_result = googlemaps.geocode("110026")
            st.write("Coordinates Identified :",geocode_result[0]['geometry']['location'])

            person_latitude = geocode_result[0]['geometry']['location']['lat']
            person_longitude = geocode_result[0]['geometry']['location']['lng']
            person_coord = pd.DataFrame({'latitude':[person_latitude],'longitude':[person_longitude]})
    
    with c2:
        if image_file is not None:
            with st.form('Process'):
                process_state = SessionState.get(processed=False)
                st.write("Press SUBMIT to process the prescription.")
                if st.form_submit_button('SUBMIT') or process_state.processed:
                    st.write('Processing...')
                    process_state.processed = True
                    ocr_text0,ocr_text = ocr(image_file.name)    
    
    st.map(person_coord,zoom=15)
    if ocr_text is not None:    
        data = ocr_text
        re.split(r'\s+\d+\.\s+', data)

        m = re.search('NAME(.+?)ADDRESS', ocr_text0)
        if m:
            pname = m.group(1)

        n = re.search('ADDRESS(.+?)[0-9]', ocr_text0)
        if n:
            paddress = n.group(1)

        st.write("## Prescription Image has been processed:")
        st.write("Name on Prescription: "+ str(pname))
        st.write("Address on Prescription: "+ str(paddress))
        st.write(ocr_text)
        df = pd.DataFrame([x.split('.') for x in data.split('.')]).rename(columns={0:'OCR_Data'})
        df = df.astype(str)
        
        def search_words(text):
            result = re.findall(r'\b[^\d\W]+\b', text)
            return " ".join(result)

        def search_numbers(text):
            result = re.findall(r'\b[0-9]\w+', text)
            return " ".join(result)

        df['OCR_Name']=df['OCR_Data'].apply(lambda x : search_words(x))
        df['OCR_Name']=df['OCR_Name'].str.split(' ').str[:2]
        df['OCR_Name']=df['OCR_Name'].str.join(" ")
        df['OCR_Qty']=df['OCR_Data'].apply(lambda x : search_numbers(x))
        df['OCR_Name'].replace("", np.nan, inplace=True)
        df.dropna(inplace=True)
        st.write(df)
        c5,c6 = st.beta_columns(2)
        with c5:
            with st.form('Approve'):
                if st.form_submit_button('Approved'):
                    message = "Uploading data to central repository and checking for medicine availability..."
        with c6:
            with st.form('Decline'):
                if st.form_submit_button('Edit'):
                    message = "Initiate Manual upload using Manual Input from Sidebar Menu."
        st.write(message)

def main():
    st.title("Medicine Availability & Delivery System")
    st.write("""## ⚕ WELCOME ⚕ """)
    st.write("")
    st.sidebar.title("⚕ M.A.D.S. ⚕")
    menu = ["Home","Manual Input","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    st.sidebar.info("Click *Home* for Manual Entry !!")
    if choice == "Home":
        st.write("### Enter Login Details")
        status = login()
        #st.write(status)
        if status is not  None:
            healthdetails()
        else:
            st.write(" ")
    elif choice =="Manual Input":
        st.write("### Enter Login Details")
        status = login()
        #st.write(status)
        if status is not  None:
            manualentry()
        else:
            st.write(" ")
    elif choice =="About":
        st.write("")
        st.write("## Developed and managed by Google Cloud Platform Team !! ")

main()
