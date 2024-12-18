# Importing necessary modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import base64

# Read the CSV containing the music data
df = pd.read_csv(r"C:\Users\HP\Downloads\Emotion-based-music-recommendation-system-main\Emotion-based-music-recommendation-system-main\muse_v3.csv")

# Preprocess the dataframe for easy use
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Split the dataframe based on different emotions
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Function to return sample data based on detected emotions
def fun(list):
    data = pd.DataFrame()
    emotion_mapping = {
        'Neutral': df_neutral,
        'Angry': df_angry,
        'Fear': df_fear,
        'Happy': df_happy,
        'Sad': df_sad
    }

    times = [30, 20, 15, 10, 5]
    for i, emotion in enumerate(list):
        t = times[i] if i < len(times) else 5
        data = pd.concat([data, emotion_mapping.get(emotion, df_angry).sample(n=t)], ignore_index=True)
    
    return data

# Function to process the emotion list and return unique emotions based on frequency
def pre(l):
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    return [item for items, c in Counter(l).most_common() for item in [items] * c]

# Define the CNN model for emotion detection
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load pre-trained weights for the model
model.load_weights(r"C:\Users\HP\Downloads\Emotion-based-music-recommendation-system-main\Emotion-based-music-recommendation-system-main\model.h5")

# Emotion dictionary mapping output index to emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Set OpenCL usage for OpenCV
cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(r"C:\Users\HP\Downloads\Emotion-based-music-recommendation-system-main\Emotion-based-music-recommendation-system-main\haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

# Streamlit page settings
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion-based Music Recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

# Emotion detection button and loop
col1, col2, col3 = st.columns(3)
list = []  # Initialize emotion list

with col2:
    if st.button('SCAN EMOTION (Click here)'):
        count = 0
        list.clear()  # Reset the list on each click

        # Start video capture and emotion detection
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            count += 1
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))

                list.append(emotion_dict[max_index])
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('s'):  # Stop on 's' key press
                break

            if count >= 20:  # Stop after 20 frames
                break

        cap.release()
        cv2.destroyAllWindows()

        list = pre(list)  # Process the list to get the unique emotion order
        st.success("Emotions successfully detected") 

        # Get song recommendations based on detected emotions
        new_df = fun(list)

        # Display the recommended songs
        st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs with Artist Names</b></h5>", unsafe_allow_html=True)

        try:
            for l, a, n, i in zip(new_df["link"][:30], new_df['artist'][:30], new_df['name'][:30], range(30)):
                st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>""".format(l, i + 1, n), unsafe_allow_html=True)
                st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>".format(a), unsafe_allow_html=True)

        except Exception as e:
            print(f"Error: {e}")  # Handle any exceptions that occur
