
#imports for data frames
import pandas as pd
#imports to get data from Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import sys
from os.path import dirname, abspath
# Path to the current script (stock_data_mining.py)
current_script_path = dirname(abspath(__file__))

# Path to the 'Data_Mining' directory, which is the parent directory of 'real_world_stock_market_dataMining'
parent_directory = dirname(current_script_path)

# Add the parent directory to sys.path
sys.path.append(parent_directory)

# Import write_to_file from writer.py
from writer import write_to_file


def main():
    
    #get the dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target = iris.target
    write_to_file("Iris-dataMining-example/datashape.txt", [data.shape])
    write_to_file("targetshape.txt", [target.shape])
    
    #training and testing sets x and y
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    write_to_file("Iris-dataMining-example/raw_train.txt", ["x_train: ", x_train, "\n y_train: ", y_train])
    #precrocessing the x data
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    
    #write the data to a txt
    write_to_file("Iris-dataMining-example/preprocessed_train.txt", ["x_train: ", x_train, "\n y_train: ", y_train])
    
    #Creating and training a classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    
    #start making predictions on the test set
    y_pred = clf.predict(x_test)

    #evalutation on the model
    acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    #write the evaluation report to a txt
    write_to_file("Iris-dataMining-example/evaluation_report.txt",["accuracy score: ", acc, "\nconfusion matrix: ", conf, "\nreport: ", report] )

#end of main


# run the program
if __name__ == "__main__":
    main()