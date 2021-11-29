# Knn-Preceptrom-Svm-Pa_Algorithms
During machine learning course I implemented 4 supervised learning models with associated learning algorithms:<br/>
* Knn - k nearest neighbors
* Preceptrom
* Svm - support-vector machines
* Pa - passive aggressive (`online` machine learning algorithm)

The algorithms are writed in `ex2.py` file, in separated classes. <br/>

##  Dataset
dataset is the Iris flower classification. In this dataset, we have five features per<br/>
instance (all the features are numerical) and three labels `0`,`1`,`2`, which are correspond <br/>
to the Iris flower species.

##  Run
The code should get as input four arguments.<br/>
The run command to the program should be:<br/>
$ python ex2.py <train_x_path> <train_y_path> <test_x_path> <output_log_name> <br/>

For example: <br/>
$ python ex2.py train_x.txt train_y.txt test_x.txt out.txt <br/>

The first parameter will be the training examples ( train x.txt ), <br/>
the second param is the training labels (train y.txt), <br/>
the third param will be the testing examples (test x.txt), <br/>
and the fourth one will be the output file name. <br/>

