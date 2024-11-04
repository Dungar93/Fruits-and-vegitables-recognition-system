# import streamlit as st
# import tensorflow as tf
# import numpy as np


# #Tensorflow Model Prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model("trained_model.h5")
#     image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) #convert single image to batch
#     predictions = model.predict(input_arr)
#     return np.argmax(predictions) #return index of max element

# #Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

# #Main Page
# if(app_mode=="Home"):
#     st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
#     image_path = "home_img.jpg"
#     st.image(image_path)

# #About Project
# elif(app_mode=="About Project"):
#     st.header("About Project")
#     st.subheader("About Dataset")
#     st.text("This dataset contains images of the following food items:")
#     st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
#     st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
#     st.subheader("Content")
#     st.text("This dataset contains three folders:")
#     st.text("1. train (100 images each)")
#     st.text("2. test (10 images each)")
#     st.text("3. validation (10 images each)")

# #Prediction Page
# elif(app_mode=="Prediction"):
#     st.header("Model Prediction")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image,width=4,use_column_width=True)
#     #Predict button
#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)
#         #Reading Labels
#         with open("labels.txt") as f:
#             content = f.readlines()
#         label = []
#         for i in content:
#             label.append(i[:-1])
#         st.success("Model is Predicting it's a {}".format(label[result_index]))
        
import streamlit as st
import tensorflow as tf
import numpy as np
import requests

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Function to Fetch Recipe Names
def get_recipe_names(ingredient):
    api_url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredient}&number=5&apiKey=88365dfca39046f3835027c7ad704718"
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        recipes = response.json()
        # Extract only the names of the recipes
        recipe_names = [recipe['title'] for recipe in recipes]
        return recipe_names
    else:
        # Handle the case if there is an error in the request
        return [f"Error: {response.status_code}"]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits - banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables - cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            labels = [i.strip() for i in content]
            
            # Display prediction result
            predicted_label = labels[result_index]
            st.success(f"Model is Predicting it's a {predicted_label}")
            
            # Fetch and display recipes
            st.subheader("Recipe Suggestions")
            recipes = get_recipe_names(predicted_label)
            
            if recipes:
                for recipe in recipes:
                    st.write(f"- {recipe}")
            else:
                st.write("No recipes found or an error occurred.")
