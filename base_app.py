"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu
# Images
from PIL import Image
import pickle

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("PARBI CLASSIFIER")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Explore", "Feature Engineering", "Prediction", "About Us", "Contact Us"]
	selection = st.sidebar.radio("Choose Option", options)

	# Building out the "Home" page
	if selection == "Home":
		image = Image.open('resources/imgs/home.jpg')
		st.image(image, caption='Climate Change')

		st.markdown("### Not just an app, it's a revolution!")
		st.write("The PARBI Classifier App is based on Machine Learning models that are able to classify whether or not a person believes in climate change based on their novel tweet data. This precise and robust response to people's perceptions of climate change lays the groundwork for the next level of business revolution, which is more focused on their customers.")

	# Building out the "Explore" page
	if selection == "Explore":
		st.markdown("### Exploratory Data Analysis (EDA)")
		# You can read a markdown file from supporting resources folder
		st.markdown("This section contains insights on the loaded dataset and its output")

		# Display the unprocessed data
		st.markdown("##### Raw Twitter data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		option = st.sidebar.selectbox('Select visualization', ('Barplots of common words', 'Word cloud of sentiments'))

		if st.checkbox('Show visualizations'):
			if option == 'Barplots of common words':
				image = Image.open('resources/imgs/most_used.png')
				st.image(image)
			else:
				with st.expander('Pro'):
					image = Image.open('resources/imgs/pro1.png')
					st.image(image)
				with st.expander('News'):
					image = Image.open('resources/imgs/news1.png')
					st.image(image)
				with st.expander('Neutral'):
					image = Image.open('resources/imgs/neutral1.png')
					st.image(image)
				with st.expander('Anti'):
					image = Image.open('resources/imgs/anti1.png')
					st.image(image)

	# Building out the "Feature Engineering" page
	if selection == "Feature Engineering":
		st.markdown("### Feature Engineering")
		# You can read a markdown file from supporting resources folder
		st.markdown("This section contains insights on the features that were added to the data")

		# Display the unprocessed data
		st.markdown("##### Balancing of data")
		if st.checkbox('Show unbalanced data'): # data is hidden if box is unchecked
			image = Image.open('resources/imgs/dist_sent.png')
			st.image(image)

		if st.checkbox('Show balanced data'): # data is hidden if box is unchecked
			image = Image.open('resources/imgs/balanced.png')
			st.image(image)

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		option = st.sidebar.selectbox(
            'Select the model from the Dropdown',
            ('Logistic Regression', 'Decision Tree', 'SVM', 'KNeighbors', 'Random Forest'))
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		# model selection options
		if option == 'Logistic Regression':
			model = "resources/Logistic_model.pkl"
		elif option == 'Decision Tree':
			model = "resources/Decision_tree_model.pkl"
		elif option == 'SVM':
			model = "resources/Support Vector_model.pkl"
		elif option == 'KNeighbors':
			model = "resources/KNeighbors_model.pkl"
		else:
			model = "resources/Random Forest_model.pkl"

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)

			word = ''
			if prediction == 0:
				word = '"**Neutral**". The tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				word = '"**Pro**". The tweet supports the belief of man-made climate change'
			elif prediction == 2:
				word = '**News**. The tweet links to factual news about climate change'
			else:
				word = '**Anti**. The tweet do not belief in man-made climate change'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(word))

	# Building out the About Us page
	if selection == "About Us":
		st.info("PARBI Tech Consultancy")
		st.write("PARBI Tech Consultancy provides data and analytics services that enable data-driven insights, well-timed and informed decisions that consistently position the company's clents ahead of the curve.")
		
		st.info("Our Vision:")
		st.write("To be the leading global data-driven solutions provider")

		st.info("Meet the team")
		Rumbie = Image.open('resources/imgs/rumbie.jpg')
		Rumbie1 = Rumbie.resize((150, 155))
		Isaac = Image.open('resources/imgs/isaac.jpg')
		Isaac1 = Isaac.resize((150, 155))
		Bongani = Image.open('resources/imgs/bongani.jpeg')
		Bongani1 = Bongani.resize((150, 155))
		Qudus = Image.open('resources/imgs/qudus.jpg')
		Qudus1 = Qudus.resize((150, 155))
		Peter = Image.open('resources/imgs/peter.jpg')
		Peter1 = Peter.resize((150, 155))

		col1, col2, col3 = st.columns(3)
		with col2:
			st.image(Rumbie1, width=150, caption="Rumbie: Team Lead")
		
		col1, col2, col3, col4= st.columns(4)
		with col1:
			st.image(Isaac1, width=150, caption="Isaac: Technical Lead")
		with col2:
			st.image(Bongani1, width=150, caption="Bongani: Project Manager")
		with col3:
			st.image(Qudus1, width=150, caption="Qudus: Senior Data Scientist")
		with col4:
			st.image(Peter1, width=150, caption="Peter: Technical Support")

	# Build the Contact us page
	if selection == "Contact Us":
		image = Image.open('resources/imgs/contactus.jpeg')
		st.image(image)
		
		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Contact info")
			st.write("82, Bush Willow Lane")
			st.write("Johannesburg, 2086, South Africa")
			st.write("Telephone:+234 7036172544")
			st.write("WhatsApp:+234 8093224263")
			st.write("Email: info@parbitech.com")
			
		with col2:
			st.subheader("Send Us")
			email = st.text_input("Enter your email")
			message = st.text_area("Enter your message")
			st.button("Send")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
