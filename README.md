# Disaster Response Pipelines
Create a machine learning pipeline to categorize messages sent during disaster events so that the messages are sent to an appropriate disaster relief agency

# Project Overview
We build an ETL pipeline to clean and merge different datasets. Following that messages are processed using nltk and then fed through a ML pipeline to train and test the data. Different machine learning models are tried and tested to predict message classification. Parameters have been fine tuned using GridDearch to get the highest accuracy. The project also includes a Flask web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

# Data Overview
We use the disaster dataset provided by Figure Eight to build a model for an API that classifies disaster messages

The data set has 2 main datasets:
- Messages.csv: Has the messages along with genre 
- Categories.csv: Has messages along with the category they belong to

# Instructions to run
Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans and stores data in SQLite database 
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

- To run ML pipeline that trains classifier
python models/train_classifier.py data/DisasterResponse.db models/model.pkl

- Run the following command in the app's directory to run your web app. 
python run.py

- Go to http://0.0.0.0:3001/

# Output Screenshot
<img width="1263" alt="Output" src="https://user-images.githubusercontent.com/10444093/77122184-35963b80-69fa-11ea-9a1c-03fb9ae4ddf5.png">

# Install
This project requires Python 3.6 and the following Python libraries installed:

- NumPy
- Pandas
- matplotlib
- seaborn
- sklearn
- nltk
- Flask
- plotly
- sqlalchemy

You will also need to have software installed to run and execute a Jupyter Notebook

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included. Make sure that you select Python 3.x installer.

# Licensing and Acknowledgements
Thanks to Udacity for providing access to Figure Eight disaster data used in this project and guidance
