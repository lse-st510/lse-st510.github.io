---
papersize: a4
documentclass: scrartcl
classoption: DIV=14
colorlinks: true
---

![LSE](images/lse-logo.jpg)
# ST445 Managing and Visualizing Data

### Michaelmas Term 2020

### Instructors

* Chengchun Shi, [c.shi7@lse.ac.uk](mailto:c.shi7@lse.ac.uk), Department of Statistics.  *Office hours*: Tuesday 9-10:00 am, Friday 9-10:00 am, Zoom ID: 929 1256 8563, Passcode: ST445, [Zoom Link](https://lse.zoom.us/j/92912568563?pwd=Z1U3RzJTek0wZ1daV210ZUlWbVZsUT09)

### Teachers/GTAs

* Christine Yuen, [l.t.yuen@lse.ac.uk](mailto:l.t.yuen@lse.ac.uk), Department of Statistics. *Office hours*: Tuesday 15-16:00 pm, Zoom ID: 967 2156 4066, Passcode: 044826, [Zoom Link](https://lse.zoom.us/j/96721564066?pwd=cW1PSVlZWDM4YlRIV3U2UXQ5N1ZCdz09)
* Siliang Zhang, [s.zhang68@lse.ac.uk](mailto:s.zhang68@lse.ac.uk), Department of Statistics. *Office hours*: Wednesday 15-16:00 pm, Zoom ID: 942 362 0935, [Zoom Link](https://lse.zoom.us/j/9423620935)

* Please use **LSE Student Hub** to book slots for office hours.

### Course Information

* **Please send an email** to Christine, Siliang or me with your **GitHub account name or email address (please use a name that does not reveal your identity)** if you would like to have access to the lecture/class materials. For auditing students, you are **not** allowed to attend online or on-campus seminars/classes.
* Lectures will be pre-recorded and uploaded to Moodle, **not** GitHub. 
* Classes will be given via a combination of online and on-campus sessions. Wednesday 11:00am-12:30pm (ONLINE), 13-14:30 pm (ONLINE), 13-14:30pm (PAN.G.01).
**Please DO NOT go to an in-person class at 1pm in PAN.G.01 if you have not registered to that class**

No lectures or classes will take place during School Reading Week 6.

### Assessment

Project assignment (60%) and continuous assessment in weeks 3, 5, 8, 10 (10% each). Students will be expected to produce 10 problem sets in the MT.

We will use GitHub classroom to manage our homeworks. Please find the instruction [here](https://github.com/lse-st445/lectures2020/blob/master/Week1/User-Friendly%20Guide%20for%20Submitting%20HW1.pdf). 

### Some Code/Suggestions for On-Campus Seminar/Class 

[LSE's Student Code for a COVID-Secure Campus](https://www.lse.ac.uk/international-relations/assets/documents/pdfs/Student-Code-for-a-COVID-Secure-Campus.pdf)

* **Wear masks** (except if medically exempted) and keep two meters social distancing. We will ask anyone without face coverings to leave the classroom. 
* Get **tested** on LSE campus. Book a test [here](https://info.lse.ac.uk/coronavirus-response/get-tested-at-lse). 
* Please do **not** attend on-campus seminar if tested positive or experiencing COVID symptoms. 
* It is totally fine to **not** attend on-campus seminars if you are worried about COVID. We have online seminar sessions and OHs. Seminar exercises, solutions and recordings will be uploaded online. 

| **Week** | **Topic**                            | **Week** | **Topic**      |
|:----------:|:--------------------------------------|:----:|:----|
| 1        | [Introduction to Data](#week-1-introduction-to-data) | 7        | [Exploratory data analysis](#week-7-exploratory-data-analysis)          |
| 2        | [Python and NumPy Data Structures](#week-2-python-and-numpy-data-structures)                  | 8        | [Matrix data visualization](#week-8-matrix-data-visualization) |
| 3        | [Wrangling Data with Pandas](#week-3-wrangling-data-with-pandas)    | 9        | [Model evaluation](#week-9-model-evaluation)                   |
| 4        | [Creating and Managing Databases](#week-4-creating-and-managing-databases)       | 10       | [Dimensionality reduction](#week-10-dimensionality-reduction)           |
| 5        | [Collecting Data from the Internet](#week-5-collecting-data-from-the-internet)                  |  11       | [Graph data visualization](#week11-graph-data-visualization)           |
| 6        | _Reading Week_                       |



### Course Description

In this course you will learn to carry out a full data science project cycle going from data acquisition to reporting with conclusions and visualisations. To do this you will use a range of Python modules: NumPy and Pandas for data cleaning and wrangling, matplotlib and Seaborn for visualisation. We shall emphasise the use of visualisation to check for the integrity of data. Techniques for data acquisition using web-scraping and Application Programming Interfaces (APIs) will be introduced as will the principles behind the use of databases. After Reading Week we will cover some more advanced topics concerned with visualising higher dimensional data sets. These techniques include the use of clustering and dimension reduction algorithms from the scikit-learn module. Finally we will look at using graphs so as to visualise relationships. Throughout, we shall be using Jupyter and Anaconda Python.   

For the final project, we will provide you with a dataset, which you will be expected to transform in order to produce a report with visualizations and conclusions. The report should demonstrate the use of the techniques taught in the course.

### Organization

This course is an introduction to the fundamental concepts of data and data visualization for students and assumes no prior knowledge of these concepts.

The course will involve 20 hours of lectures and 15 hours of computer workshops in the MT.

### Prerequisites

No prior experience with programming is required.

### Software

We will use some tools, notably SQLite and Python, but these will be used in coordination with MY470 (Computer Programming) where their use will be covered more formally.  Lectures and assignments will be posted on Github, Students are expected to use Github also to submit problem sets.

Where appropriate, we will use Jupyter notebooks for lab assignments, demonstrations, and the course notes themselves.

### Schedule

------
#### Week 1. [Introduction to Data](https://github.com/lse-st445/lectures2020/blob/master/Week1/ST445_wk1_lecture.ipynb)

In the first week, we will introduce the basic concepts of the course, including how data is recorded, stored, and shared.  Because the course relies fundamentally on GitHub, a collaborative code and data sharing platform, we will introduce the use of git and GitHub, using the lab session to guide students through in setting up an account and subscribing to the course organisation and assignments.

This week will also introduce basic data types from the perspective of machine implementations through to high-level programming languages. A short historical perspective to data science will be given. Issues concerning data integrity will be discussed. The process flow of capturing, wrangling, exploring and visualising data will be emphasised. We will introduce the notion of databases and database managers.

*Lecture Notes*:
* [Lecture, Week 1](https://github.com/lse-st445/lectures2020/blob/master/Week1/ST445_wk1_lecture.ipynb)
* [Python example to fix](https://github.com/lse-st445/lectures2020/blob/master/Week1/DebugExercise.ipynb)


*Readings*:
* Lake, P. and Crowther, P. 2013. _Concise guide to databases: A Practical Introduction_.  London: Springer-Verlag.  Chapter 1, [Data, an Organizational Asset](https://books.google.co.uk/books?id=SuK2BAAAQBAJ&pg=PA301&lpg=PA301&dq=Concise+Guide+to+Databases+pdf&source=bl&ots=pEJj8miMrf&sig=3nrRgpk3kF7fXzcWUWpJ_uzpfl0&hl=en&sa=X&ved=0ahUKEwiAkM3JrbHWAhXE7xQKHWseCZAQ6AEISzAH#v=onepage&q=Concise%20Guide%20to%20Databases%20pdf&f=false)
* Goodrich, M.T., Tamassia, R. and Goldwasser, M.H. 2013. _Data structures and algorithms in Python_. John Wiley & Sons Ltd.  Ch. 1, through section 1.3.
* Wickham, Hadley.  Nd.  _Advanced R_, 2nd ed.  Ch 1, [Introduction](https://adv-r.hadley.nz/introduction), and Chapter 2, [Names and Values](https://adv-r.hadley.nz/names-values.html#function-calls).
* [GitHub Guides](https://guides.github.com), especially: "Understanding the GitHub Flow", "Hello World", and "Getting Started with GitHub Pages".

*Further Readings*:
* "[Understanding Big and Little Endian Byte Order](https://betterexplained.com/articles/understanding-big-and-little-endian-byte-order/)".  _Better Explained_ website.
* Nelson, Meghan.  2015.  "[An Intro to Git and GitHub for Beginners (Tutorial).](http://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)"
* GitHub.  "[Markdown Syntax](https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf)" (a cheatsheet).
* Chacon, Scott and Ben Straub. [_Pro Git_](https://git-scm.com/book/en/v2). 2nd ed. Apress.  Chapters 1-2.
* Jim McGlone, "[Creating and Hosting a Personal Site on GitHub
A step-by-step beginner's guide to creating a personal website and blog using Jekyll and hosting it for free using GitHub Pages.](http://jmcglone.com/guides/github-pages/)".

*Lab*: [**Working with Ipython notebook and GitHub**](https://github.com/lse-st445/lectures2020/blob/master/Week1/ST445_wk1_class.ipynb).

------
#### Week 2. [Python and NumPy Data Structures](https://github.com/lse-st445/lectures2020/blob/master/Week2/ST445_Week2_Lecture.ipynb)

Firtly, we shall review fundamental Python datatypes such as lists and dicts. Then we shall introduce Numerical Python or NumPy which is the module on which Pandas is built. NumPy permits fast array based computation and is the basis for efficient pre-processing and visualisation of data. Many of the built-in NumPy methods can be used in Exploratory Data Analysis (EDA). We will also cover ways to restructure data from "wide" to "long" format, within strictly rectangular data structures.  Additional topics concerning text encoding, date formats, and sparse matrix formats are also covered.

*Readings*:
* [NumPy Documentation](https://www.numpy.org)
* Wickham, Hadley and Garett Grolemund.  2017.  _R for Data Science: Import, Tidy, Transform, Visualize, and Model Data_.  Sebastopol, CA: O'Reilly.  [Part II Wrangle](http://r4ds.had.co.nz/wrangle-intro.html), [Tibbles](http://r4ds.had.co.nz/tibbles.html), [Data Import](http://r4ds.had.co.nz/data-import.html), [Tidy Data](http://r4ds.had.co.nz/tidy-data.html) (Ch. 7-9 of the print edition).
* McKinney, Wes. 2017 (2nd Edition) _Python for Data Analysis_. Sebastopol, CA: O'Reilly

*Further Resources*:
* Reshaping data in Python: "[Reshaping and Pivot Tables](https://pandas.pydata.org/pandas-docs/stable/reshaping.html)".
* Robin Linderborg, "[Reshaping Data in Python](https://hackernoon.com/reshaping-data-in-python-fa27dda2ff77)".

*Lab*: [**Control flow in Python**](https://github.com/lse-st445/lectures2020/blob/master/Week2/ST445_wk2_class.ipynb).

------

#### Week 3. [Wrangling Data with Pandas](https://github.com/lse-st445/lectures2020/blob/master/Week3/ST445_week3_Lecture.ipynb)

This week we shall be exploring Pandas which is one of Python's main tools. It gives Python a DataFrame similar to the other main data science language R. Pandas can handle heterogeneous data and so it extends the capability of NumPy, which is mostly suited to homogeneous numerical data. Pandas works well with other key Python modules such as scikit-learn (machine learning) and matplotlib. We will also cover common data formats such as JSON (Javascript Object Notation).  


*Readings*:
* [10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
* McKinney, Wes. 2017 (2nd Edition) _Python for Data Analysis_. Sebastopol, CA: O'Reilly
* Galea, Alex 2018 _Beginning Data Science with Python and Jupyter_ Packt

*Lab*: [**More on pandas**](https://github.com/lse-st445/lectures2020/blob/master/Week3/ST445_week3_class_presentation.ipynb)

------
#### Week 4. [Creating and Managing Databases](https://github.com/lse-st445/lectures2020/blob/master/Week4/ST445_week4_Lecture.ipynb)

We will return to database normalization, and how to implement this using good practice in a relational database manager, SQLite.  We will cover how to structure data, verify data types, set conditions for data integrity, and perform complex queries to extract data from the database.    

*Readings*:
*	Lake, Peter.  _Concise Guide to Databases: A Practical Introduction_. Springer, 2013.  Chapters 4-5, Relational Databases and NoSQL databases.
*	Nield, Thomas. _Getting Started with SQL: A hands-on approach for beginners_. O’Reilly, 2016.  Entire text.
* [SQLite documentation](https://www.sqlite.org/docs.html).
*   Bassett, L. 2015.  [_Introduction to JavaScript Object Notation: A to-the-point Guide to JSON_](http://shop.oreilly.com/product/0636920041597.do).  O'Reilly Media, Inc.
* Shay Howe. 2015.  [_Learn to Code HTML and CSS: Develop and Style Websites_](http://learn.shayhowe.com/html-css/).  New Riders.  Chs 1-8.

*Lab*: [**Classes in Python**](https://github.com/lse-st445/lectures2020/blob/master/Week4/ST445_week4_class_presentation.ipynb)

------

#### Week 5. [Collecting Data from the Internet](https://github.com/lse-st445/lectures2020/blob/master/Week5/ST445_Week5_Lecture.ipynb)

Publicly accessible _application programming interfaces_ (APIs) provide a common source of "big" data available from a variety of sources, such as social media data.  This data consists of a variety of data types, but is usually transmitted in JSON format.  In this session, we will cover the basics of APIs, including authentication and the use of protocols for interacting with APIs, and in processing the data that is obtained using these methods.  We will also discuss common problems in using text, including character encodings, working with Unicode, transforming text into numeric data, and cleaning textual data for analysis.We will cover basic web scraping, to turn web data into text or numbers.

*Readings*:
*   Cooksey, Brian.  [_An Introduction to APIs_](https://zapier.com/learn/apis/#download).  Zapier, 2014.
*   [**python-twitter** documentation](https://python-twitter.readthedocs.io/en/latest/)
* Vik Paruchuri, "[Python Web Scraping Tutorial using BeautifulSoup](https://www.dataquest.io/blog/web-scraping-tutorial-python/)", 17 November 2016.
* [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* Justin Yek, "[How to scrape websites with Python and BeautifulSoup](https://medium.freecodecamp.org/how-to-scrape-websites-with-python-and-beautifulsoup-5946935d93fe)", 10 June 2017.

*Further Resources*:
* [Documentation on the Twitter REST API](https://dev.twitter.com/rest/public)
* Richard Ishida. 2015.  "[Character encodings for beginners](https://www.w3.org/International/questions/qa-what-is-encoding)".  W3C.

*Lab*: [**More on web scraping and API**](https://github.com/lse-st445/lectures2020/blob/master/Week5/ST445_week5_class_presentation.ipynb)

------
#### Week 6. Reading Week


------
#### Week 7. Exploratory data analysis

We will introduce the basic statistical plots that are commonly used in exploratory data analysis. We will first consider standard plots for univariate data analysis, including histograms, empirical distribution functions, as well as plots of summary statistics such as boxplots and violinplots. We will then consider different variants of bar plots, which are commonly used for comparison of parallel batches of data, as well as scatter plots for exploration of correlation patterns in data.

*Readings*:
* M. Friendly, [A Brief History of Data Visualization](http://www.datavis.ca/papers/hbook.pdf), Handbook of Computational Statistics: Data Visualization (Editors C. Chen, W. Hardle and A. Unwin), Vol III, Springer-Verlag, 2006
* E. R. Tufte, The Visual Display of Quantitative Information, Second Edition, Graphics Press, 2001
* J. W. Tukey, Exploratory Data Analysis, Pearson, 1977
* [Matplotlib](https://matplotlib.org)
* [Seaborn: statistical data visualization](https://seaborn.pydata.org)

#### *Lab*: Matplotlib primer and basic statistical plots

------
#### Week 8. Matrix data visualization

We will consider how to visualize matrix data such as covariance and other similarity matrices and adjacency matrices of graphs such as those representing social networks. The key here is to use a suitable ordering of matrix rows and columns to visualize any possibly existing clustering structure. We will explain the underlying methods based on spectral theory of matrices, using the concepts of matrix eigenvectors and clustering based on matrix eigenvectors. In particular, we will explain the method based on *seriation* using the so-called Fiedler eigenvector and *spectral co-clustering* based on using eigenvectors in combination with k-means clustering method.

*Readings*:
* L. Wilkinson and M. Friendly, [History Corner: The History of the Cluster Heat Map](https://www.cs.uic.edu/~wilkinson/Publications/heatmap.pdf), The American Statistician, Vol 63, No 2, May 2009
* I. S. Dhilon, [Co-clustering documents and words using bipartite spectral graph partitioning](http://www.cs.utexas.edu/users/inderjit/public_papers/kdd_bipartite.pdf), Proc. of ACM KDD, 2001
* Scikit-learn documentation, [Section 2.4: Biclustering](http://scikit-learn.org/stable/modules/biclustering.html)

#### *Lab*: Statistical plots using Matplotlib and Seaborn
* Synthetic matrix data visualization using seriation method
* Visualization of adjacency matrices derived from GitHub archive dataset
* Using sklearn.cluster.bicluster

------
#### Week 9. Model evaluation

In this week, we will introduce standard statistical plots for the performance evaluation of statistical models and machine learning algorithms for classification. We will introduce standard statistical plots for assessing the performance of binary classifiers, such as *receiver operating characteristic* (ROC) and *precision-recall* (PR) curves. We will learn how to interpret these plots and discuss their advantages and limitations.

We will also discuss various standard metrics used for assessing the performance of binary classifiers, such as *accuracy*, *area under the curve* (AUC) and *Gini coefficient*, discuss their relation to the ROC curve, as well as their advantages and limitations.


*Readings*:
* J. A. Sweets, R. M. Dawes and J. Monahan, [Better Decisions through Science](http://ist-socrates.berkeley.edu/~maccoun/LP_SwetsDawesMonahan2000.pdf), Scientific American, October 2000, pp 82-87
* T. Fawcet, An Introduction to ROC Analysis, Pattern recognition letters, Vol 27, pp 861-874, 2006
* N. Japkowicz and M. Shah, Evaluating Learning Algorithms: A Classification Perspective, Cambridge University Press, 2011
* API reference: [sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)


#### *Lab*: Evaluating classifiers using sklearn.metrics
* Comparing binary classifiers in ROC and PR space
* Comparison of ROC and PR curves
* Accuracy, AUC and other metrics

------
#### Week 10. Dimensionality reduction

We will consider how to visualize hidden structures in high-dimensional data, such as hidden clusters or embedded low-dimensional manifolds, by using dimensionality reduction methods. We will explain the underlying principles of dimensionality reduction methods such as multidimensional scaling, locally linear embedding, isomap, spectral embedding, and stochastic neighbor embedding. We will see how the geometry, linear algebra and optimisation methods give raise to different dimensionality reduction methods.

Our focus will be on the dimensionality methods that are commonly used in practice and widely available through software libraries such as sklearn.manifold. We will also consider modern tools for visualizing different dimensionality reductions such as Google embedding projector.


*Readings*:
* T. F. Cox and M. A. A. Cox, Multidimensional Scaling, Second Edition, Chapman & Hall / CRC, 2000
* I. Borg and P. J. F. Groenen, Modern Multidimensional Scaling: Theory and Applications, Second Edition, Springer, 2005
* A. Geron, Hands-on Machine Learning with Scikit-Learn & TensorFlow, O’Reilly, 2017, Chapter 8, Dimensionality Reduction
* Google's [embedding projector](http://projector.tensorflow.org)
* API reference, [scikit-learn, Section 2.2: manifold learning](http://scikit-learn.org/stable/modules/manifold.html)

#### *Lab*: Dimensionality reduction using sklearn.manifold
* Dimensionality reduction plots using different methods
* Understanding the meaning of various input parameters
* Understanding the sensitivity to the input parameter values

------
#### Week 11. Graph data visualization

In the last week, we will consider basic methods for visualization of graph data such as visualizing social network relationships. We will consider different graph layouts and the principles of how they are computed. This will involve methods based on simple principles for drawing graphs that have a tree structure as well as more sophisticated methods based on spectral theory of linear algebra and dynamical systems for general graphs.

*Readings*:
* A. Hagberg, D. Schult and P. Swart, [NetworkX Reference]( https://networkx.github.io/documentation/latest/_downloads/networkx_reference.pdf)
* NetworkX: Software for complex networks, [https://networkx.github.io/](https://networkx.github.io/)
* [Graphviz – Graph Visualisation Software](http://graphviz.org/), especially manual pages, layout commands


#### *Lab*: Graph drawing using NetworkX
* Loading and manipulating graphs using NetworkX
* Changing basic properties of graph visualization such as node or edge colors
* Drawing graphs using different layouts
* Using graphviz graph layouts
