# Real-time-failure-detection

To re-run experiments presented in paper "Sensor fault detection and isolation for health monitoring
devices by learning from failures", clone this repository and clone the environment found in envs/conda_env.yml.

**The following steps are necessary to run the experiments:**\
Create a database called *final* in mysql command line:
```
create database final;
```
Then, save the *sql_databases/dump-final-tables-start-2021.sql* file in MySQL\MySQL Server {*version*}\bin folder. Open a command prompt with admin rights in the same folder and run:
```
mysql -u root -p final < dump-final-tables-start-2021.sql
```

Python version used is 3.7.1 and to run the experiments, the following conda environment should be created and activated:
```
cd envs
conda env create --name conda_env --file=conda_env.yml
conda activate conda_env
```
Download the folders with [recorded data](https://drive.google.com/file/d/1FaJK0pMIHg-x5dnGmrmts8z9h3-vbLpa/view?usp=sharing) and [written failure reports](https://drive.google.com/file/d/1gmovGbZ0iGfg5QCHeaK6wkSqllJ2qdbE/view?usp=sharing).
Now, run the python scripts to generate tables with features for sensor data and for word documents.\
To generate the sensor features run the *extracting_{sensor}_features.py* files from main folder. As an example:
```
python extracting_gsr_features.py --sql_password {insert here your sql password} --data_folder_location {location of folder with recorded data .csv files}
```
To generate the tfidf vectors run the code below:
```
python tfidf_vector_extraction.py --sql_password {insert here your sql password} --report_folder_location {location of folder with recorded data .docx files}
```
Lastly, to generate working features from multiple working sensor features, run this code:
```
python get_working_device.py --sql_password {insert here your sql password}
```
Once all sql feature tables have been created, the last step is training and testing Naive Bayes, SVMs and Neural Networks on these features. Depending on the number of labeled files we want to provide, the number for the *provided_labels* should be varied. To use only one labeled example of each follow the example below:
```
python ComparisonBetweenLabelledAndNLPLabels.py --sql_password {insert here your sql password} --provided_labels 1 --results_folder {folder where plots and summary .txt files are generated}
```
To generate only fully supervised classification results for reproducing tables found in section 7, run the following code:
```
python data_classification_only.py --sql_password {insert here your sql password} --results_folder {folder where plots and summary .txt files are generated}
```
To generate only NLP clustering results from section 5 run the following code with the number of required labeled examples of each failure type:
```
python nlp_clustering_accuracy.py --sql_password {sql password} --database_name final_bare --provided_labels 1 --results_folder {folder where plots and summary .txt files are generated}
```
To print information about features in tables found in the paper, various useful functions can be found in *misc/getresultsforNLPjournal.py* file.
