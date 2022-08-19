## IMDb Rating Prediction Algorithm

---
###UNIFESP Artificial Intelligence Final Project 

The [IEEE Document](https://github.com/LewisDamy/IMDB-Prediction-Algorithm-/blob/main/Luis_Damy_TrabalhoFinal.pdf) containing the images and more deep explanation 
of the project can be found in inside the repository once it was
required by the professor to do it as an academic paper best practice.

Note: This document were written in Portuguese. 

--- 

Python project to predict the rating of a movie based on:
- Genre
- Duration
- Rating
- Age Rating
- Actor Likes
- Movie Likes
- Director Likes
- Cast Total Likes
- User Reviews
- Critic Reviews

## Approach:

---

1. Data Extraction
   
    It has brought to my attention that the IMDb has the best
    dataset. Knowing that, the source from this project were 
    from the [IMDB Dataset Interface](https://www.imdb.com/interfaces/)
   

2. Data Preprocessing
   
    Through the 5 thousand lines and 30 columns brought from the
    extraction by a cleaning and filtering relevant content. 
    This step was done with help of [Pandas Library](https://github.com/pandas-dev/pandas)
    when filtering and saving to another csv file 
    while dumping the futile information
   

3. Machine Learning Techniques
    
    During this process it was used the [Sklearn Library](https://github.com/scikit-learn/scikit-learn)
    that provided the functions for training and running the test from [KNN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
    In addition to that, it was also used the [Stats Models Library](https://github.com/statsmodels/statsmodels) to run the [Linear Regression Algorithm](https://en.wikipedia.org/wiki/Linear_regression)
    as well as documenting the results in a .txt file.

## Usage:

---

- Creating virtual environment:
```python
python -m venv venv
```
- Installing requirements:
```makefile
make install
```
- Running the application:
```python
python main.py
```

## Results:

---

|Algorithm               | Accuracy | Time Execution | Test Size %   |
| ---------------------- | -------- | -------------- | ------------- |
| **KNN**                |  0.7895  |     0.0016s    |      40       |
| **Linear Regression**  |  0.6202  |     4.741s     |      40       |

Notice that the KNN were a lot better against the Linear Regression. That 
said, it has been tested a myriad of test sizes from 20 up to 80 percent,
and the results remained the same. It was done the same for testing the 
accuracy of the K value ranging up to 70 that weren't surprising when K 
reaches 35.

