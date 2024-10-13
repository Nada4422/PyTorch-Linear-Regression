
# PyTorch Linear Regression

## Objective

The objective of this task is to use an Artificial Neural Network (ANN) implemented in PyTorch to predict housing prices based on various features, including area, number of bedrooms, bathrooms, and other categorical attributes such as air conditioning, guestroom availability, and parking. The model is trained to perform regression to analyze the relationship between these features and housing prices. 

This includes:

1- Building and training a neural network to predict housing prices.

2- Evaluating the model's performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.

3- Visualizing the relationship between actual and predicted prices.

## Dataset Description
The dataset used contains the following columns:

1- price: Target variable (the price of the house).

2- area: The area of the house.

3- bedrooms: Number of bedrooms.

4- bathrooms: Number of bathrooms.

5- stories: Number of stories (floors) in the house.
mainroad: Whether the house is on the main road 

6- categorical: Yes/No).

7- guestroom: Whether the house has a guestroom (categorical: Yes/No).

8- basement: Whether the house has a basement (categorical: Yes/No).

9- hotwaterheating: Whether the house has hot water heating (categorical: Yes/No).

10- airconditioning: Whether the house has air conditioning (categorical: Yes/No).

11-parking: Number of parking spaces.

12-prefarea: Whether the house is in a preferred area (categorical: Yes/No).

13- furnishingstatus: Furnishing status of the house (categorical: 'furnished', 'semi-furnished', 'unfurnished').

**Categorical Columns**

The following columns are categorical and will be label-encoded in the preprocessing stage:

1- mainroad

2- guestroom

3- basement

4- hotwaterheating

5- airconditioning

6- parking

7- prefarea

8- furnishingstatus

## Steps to Run the Code in Google Colab
**Step 1: Set Up Your Environment**

1- Open Google Colab: Go to Google Colab and create a new notebook.

2- Upload Dataset: Upload your dataset (Housing-1.csv) in the Colab environment. You can do this by clicking on the "Files" icon on the left-hand side of Colab and selecting "Upload."

**Step 2: Install Dependencies**

Before running the code, ensure you have all the required libraries. Most of these are pre-installed in Google Colab, but you can use the following commands to ensure everything is set up:

!pip install torch torchvision
!pip install pandas scikit-learn matplotlib

**Step 3: Upload Your Dataset**

You will need to upload your dataset (Housing-1.csv) to your Colab environment. You can do this in the following way:

1- In the Colab menu, click on the Files tab.

2- Select Upload and upload the Housing-1.csv file.

Alternatively, if the file is stored in your Google Drive, you can mount the drive:

from google.colab import drive
drive.mount('/content/drive')

**Step 4: Copy and Run the Code**

Copy the code provided into the cells of the Colab notebook and run the cells sequentially.

**Step 5: Model Training and Evaluation**

After running the code, the model will train and provide evaluation metrics like MSE, MAE, and R² Score. Additionally, visualizations comparing actual and predicted prices will be generated.

## Dependencies and Installation Instructions
The code depends on several Python libraries. Ensure the following are installed before running the script:

1- PyTorch: pip install torch torchvision

2- Pandas: pip install pandas

3- Scikit-learn: pip install scikit-learn

4- Matplotlib: pip install matplotlib

These libraries handle data processing, model creation, training, and evaluation, as well as visualizing the results.

**List of Libraries**

1- PyTorch: For building and training the neural network.

2- Pandas: For reading and manipulating the dataset.

3- Scikit-learn: For preprocessing data (e.g., encoding categorical features, splitting data), scaling, and evaluating performance.

4- Matplotlib: For visualizing the results (actual vs. predicted prices).

## Additional Notes
1- If you're running the code on your local machine, ensure you have a compatible version of Python (3.7 or higher is recommended).

2- If missing values are present in the dataset, they are handled appropriately with df.dropna() in the code.

3- The dataset's categorical features are encoded using LabelEncoder to convert them into numeric values.

## Objective Recap
This README file is designed to:

1- Guide you through the task of using an ANN to perform regression on housing prices.

2- Explain how to preprocess the dataset.

3-Detail the steps to run the code in Google Colab.

4- List all required dependencies and how to install them.

5- Provide visualizations of the actual vs. predicted prices as part of the model evaluation process.
