![LSE](images/lse-logo.jpg)
# ST510 Foundations of Machine Learning

### Lent Term 2021

### Instructors

* Yining Chen, [y.chen101@lse.ac.uk](mailto:Y.Chen101@lse.ac.uk), Department of Statistics. *Office hours*: by appointment
* Kostas Kalogeropoulos, [k.kalogeropoulos@lse.ac.uk](mailto:k.kalogeropoulos@lse.ac.uk), Department of Statistics.  *Office hours*: by appointment
* Joshua Loftus, [J.R.Loftus@lse.ac.uk](mailto:J.R.Loftus@lse.ac.uk), Department of Statistics. *Office hours*: by appointment
* Xinghao Qiao, [x.qiao@lse.ac.uk](mailto:x.qiao@lse.ac.uk), Department of Statistics. *Office hours*: by appointment
* Chengchun Shi, [c.shi7@lse.ac.uk](mailto:c.shi7@lse.ac.uk), Department of Statistics.  *Office hours*: by appointment
* Milan Vojnovic, [m.vojnovic@lse.ac.uk](mailto:M.Vojnovic@lse.ac.uk), Department of Statistics.  *Office hours*: by appointment

### Teaching assistant

* Bento Natura, [b.natura@lse.ac.uk](mailto:b.natura@lse.ac.uk), Department of Mathematics. *Office hours*: TBD

Please use **LSE Student Hub** to book slots for seminar office hours.


| **Week** | **Topic**                            | **Week** | **Topic**      |
|:----------:|:--------------------------------------|:----:|:----|
| 1        | [Statistical learning theory](#week-1-statistical-learning-theory) | 7        | [Neural networks](#week-7-neural-networks)          |
| 2        | [Convex optimisation](#week-2-convex-optimisation)                  | 8        | [Unsupervised learning - clustering](#week-8-unsupervised-learning-clustering) |
| 3        | [Non-convex optimisation](#week-3-nonconvex-optimisation)    | 9        | [Unsupervised learning - dimension reduction](#week-9-unsupervised-learning-dimension-reduction)                   |
| 4        | [Support vector machines](#week-4-support-vector-machines)       | 10       | [Online learning and optimisation](#week-10-online-learning-and-optimisation)           |
| 5        | [Decision trees and random forests](#week-5-decision-trees-and-random-forests)                  |  11       | [Reinforcement learning](#week11-reinforcement-learning)           |
| 6        | _Reading Week_                       |



### Course Description

The goal of this course is to provide students with a training in foundations of machine learning with a focus on statistical and algorithmic aspects. Students will learn fundamental statistical principles, algorithms, and how to implement and apply machine learning algorithms using the state-of-the-art Python packages such as scikit-learn, TensorFlow, and OpenAI Gym.

### Organization

The course will involve 20 hours of lectures and 10 hours of computer workshops in the LT.

### Prerequisites

A knowledge of probability and statistical theory to the level of ST102 and ST206 and some parts of ST505 (e.g. linear models and generalized linear models). Some experience with computer programming will be assumed (e.g., Python, R).

### Availability

This course is available on the MPhil/PhD in Statistics. This course is available with permission as an outside option to students on other programmes where regulations permit.

The availability as an outside option requires a demonstration of sufficient background in mathematics and statistics and is at the discretion of the instructor.

### Schedule

------
#### Week 1. Statistical learning theory

In this lecture we cover basic concepts and some of the key results of statistical learning theory. We start with an introduction to basic assumptions of the statistical learning framework and the key concept of *probably almost correct* (PAC) learning, and discuss the concept of the *bias-complexity trade-off*. We then discuss two facts for learning infinite hypothesis classes: first, that some infinite hypothesis classes are learnable, and second, that a universal learner that has no prior knowledge cannot be successful in learning any given task. We then introduce and discuss different concepts for learning infinite hypothesis classes, starting with  *uniform convergence* and showing that it is a sufficient condition for learning. We then introduce the concepts of *Rademacher complexity* and *growth function*, and fundamental bounds for learning using these concepts. Finally, we introduce the concept of *VC-dimension*, its relation to the growth function, and the *fundamental theorem of PAC learning*. 

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week1/lecture1-slides.pdf)
* Lecture [notes](https://github.com/lse-st510/Lectures2021/blob/main/Week1/lecture1.pdf)

*Readings*:
* Shai Shalev-Shwartz and Shai Ben-David, Understanding Machine Learning: from Theory to Algorithms, Cambridge University Press, 2014; [online](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)
* Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar, Foundations of Machine Learning, 2nd Edition, The MIT Press, 2018
* Martin J. Wainwright, High-Dimensional Statistics: A Non-Asymptotic Viewpoint, Cambridge University Press, 2019 (Chapter 2, Basic tail and concentration inequalities)

*Lab*: 
* PAC learning
* Bias-complexity trade-off: polynomical regression example
* Rademacher complexity and growth function example
* VC dimension example

------
#### Week 2. Convex optimisation

In this lecture we cover basic concepts and algorithms of convex optimisation. We start with the definition of convexity and the implications of having a convex objective function to minimise. We then look into the unvariate case, focusing on bisection method, gradient descent and Newton-Raphson. We discuss the convergence rates, as well as some theory for the aforementioned algorithms. These algorithms are illustrated again in the multivariate setting. Other topics such as coordinate descent, stochastic gradient descent (SGD), and acceleration by momentum are also briefly mentioned.

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week2/Lecture2.pdf)

*Readings*:
* Stephen Boyd and Lieven Vandenberghe, Convex optimization, Cambridge University Press, 2004; book [here](http://web.stanford.edu/~boyd/cvxbook)
* Sebastien Bubeck, Convex optimization: algorithms and complexity, Now Publishers Inc. 2016; book [here](http://sbubeck.com/Bubeck15.pdf)
* Jorge Nocedal and Stephen Wright,  Numerical optimization, Springer, 2006; book [here](https://www.springer.com/gp/book/9780387303031) (for free access - via LSE library)
* Shai Ben-David and Shai Shalev-Shwartz, Understanding machine learning: from theory to algorithms, Cambridge University Press, 2014; book [here](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/index.html)

*Lab*:
* Convex analysis - some theory
* Exercise on convergence rates
* Implementing bi-section/Newton/Stochastic gradient in Python
* Basic stochastic gradient descent in Python for logistic regression

------

#### Week 3. Non-convex optimisation

In this week we will introduce and describe the problem of non-convex optimisation. This is problem is particularly challenging, known as a *NP-Hard* problem. We will focus on techniques such as Random Gradient Descent methods, Bayesian optimisation, Markov Chain Monte Carlo and we will briefly touch Genetic Algorithms. The focus will be on the principles while keeping close tabs with applications. You can find the lecture slides below as well as some further reading which you may explore further if want to do a project on non-convex optimisation.

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week3/SlidesWeek03.pdf)

*Readings*:
* [Tutorial](https://arxiv.org/abs/1807.02811) on Bayesian Optimisation by Peter Frazier
* [Blog](http://krasserm.github.io/2018/03/21/bayesian-optimization/) on Bayesian Optimisation by Martin Krasser
* [Textbook](https://www.mcmchandbook.net/) on MCMC
* [Introduction](https://arxiv.org/abs/1701.02434) to Hamiltonian MCMC
* [Paper](http://www-stat.wharton.upenn.edu/~edgeorge/Research_papers/GeorgeMcCulloch97.pdf) on Bayesian variable selection
* [Paper](https://www.cs.ubc.ca/~arnaud/delmoral_doucet_jasra_smcsamplers_jrssb.pdf) on Sequential Monte Carlo

*Lab*:
* Sampling from the posterior using the Gibbs Sampler in Python
* Presenting the output of a Markov Chain Monte Carlo (MCMC) ouput 
* Sparse Linear Regression
* Spike and Slab Priors in Linear Regression

------
#### Week 4. Support vector machines

This week we focus on linear methods for high-dimensional supervised learning, with particular attention to classification with SVMs, kernel methods, and the "kernel trick." We begin with a brief review of logistic regression and classification with linear decision boundaries, then discuss maximum margin classification with SVMs, introduce kernels as a method for relatively automated feature transformation or embedding, discuss mathematical results that tell us kernel optimization problems reduce from high- and potentially infinite-dimensional to problems with dimension controlled by the sample size, and conclude by connecting these methods to regression via ridge (or L2) penalization.

* Lecture [slides](slides_svm_kernels_lasso_part1.pdf)

*Readings*:
* [ESL](https://web.stanford.edu/~hastie/ElemStatLearn/) Chapter 12 for SVMs, 3.4 for ridge. Supplementary: 5.8 for RKHS in the splines context, and 6 on kernel regression
* [AoS paper](https://projecteuclid.org/euclid.aos/1211819561) with solid definition-theorem presentation for statisticians ([arxiv version](https://arxiv.org/abs/math/0701907))
* R package vignettes for [`kernlab`](https://cran.r-project.org/web/packages/kernlab/vignettes/kernlab.pdf) and/or [`e1071`](https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf)
* Python scikit learn documentation for [kernel ridge](https://scikit-learn.org/stable/modules/kernel_ridge.html) and [svm](https://scikit-learn.org/stable/modules/svm.html)

*Lab*:
* R markdown notebooks and some useful libraries
* Classification with non-linear decision boundaries
* Support vector machines and kernel SVM

------

#### Week 5. Decision trees and random forests

In this lecture we cover tree-based methods and some ensemble methods. We start with regression and classification trees, discussing the estimation procedure, cost complexity pruning, different node purity measures and illustrative examples. We then discuss bagging and random forests, including e.g. the variance reduction, Out-of-Bag errors and kernel-based random forests. Some recent developments of random forests will also be briefly mentioned.

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week5/ST510_Lecture%205%20slides.pdf)

*Readings*:
* [Textbook](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) on regression/classification trees, bagging and random forests.
* [Paper](https://projecteuclid.org/journals/annals-of-statistics/volume-30/issue-4/Analyzing-bagging/10.1214/aos/1031689014.full) on theoretical analysis of bagging.
* [Paper 1](https://www.jmlr.org/papers/volume13/biau12a/biau12a.pdf), [Paper 2](https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-4/Consistency-of-random-forests/10.1214/15-AOS1321.full) on theoretical analysis of random forests.
* [Paper](https://erwanscornet.github.io/pdf/articlekernel.pdf) on kernel-based random forests.
* [Paper](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839?journalCode=uasa20) on nonparametric causal forest.

*Lab*: 
* Constructing classification trees using different node purity measures. [Exercise sheet](https://github.com/lse-st510/Lectures2021/blob/main/Week5/Seminar%205%20exercise.pdf) and [solution](https://github.com/lse-st510/Lectures2021/blob/main/Week5/Solution%20to%20Seminar%205%20exercise.pdf).
* Implementing regression/classification trees, tree pruning, bagging and random forests. [Python code](https://github.com/lse-st510/Lectures2021/blob/main/Week5/Seminar5.ipynb).
------
#### Week 6. Reading Week


------
#### Week 7. Neural networks

In part 1 this week we focus on penalized regression methods for high-dimensional supervised learning, with particular attention to lasso. In part 2 we introduce deep learning with feed-forward neural networks.

Part 1 begins with a review of the Stein paradox and bias in estimation through the James-Stein estimator and ridge regression. The rest of the lecture then focuses on the lasso for sparse, interpretable regression in high-dimensional problems. We discuss the constrained and lagrangian forms of the lasso optimization problem, give some intuition about why solutions are sparse, interpret the Karush-Kuhn-Tucker conditions and compare them to ridge and OLS, investigate the degrees of freedom, optimism gap, and choosing the penalty parameter with cross-validation, and conclude by illustrating how model selection bias can invalidate classical linear regression inferences if computed on the same data used to select the model.

In part 2 we introduce the multilayer perceptron (MLP) or feed-forward neural network architecture for building complex non-linear functions by compositions of simple non-linear functions. We discuss some of the terminology used to describe network architecture, give some intuition about why the depth of the network improves on using single hidden-layer networks, briefly focus on some of the specific issues that arise in optimization and fitting of these models, try to understand why such highly overparametrized models can have good generalization error, and briefly conclude with words of caution on out-of-distribution generalization and potential ethical issues with common deep learning applications like facial recognition.

- [Slides](https://joshualoftus.com/ml4ds/06-regularization/slides/06-1-regularization-phd.html) for part 1 on lasso
- [Slides](https://joshualoftus.com/ml4ds/07-nonlinear/slides/07-1-nonlinearity-phd.html) for part 2 on neural nets

*Readings*:
#### For part 1 on the lasso
* Optimism / generalization gap (ESL 7.4-6)
* Covariance penalty and degrees of freedom (CASI 12.3)
* Cross-validation (ESL 7.10)
* SLS 2 for lasso, cross-validation, degrees of freedom
* CASI 16 for lasso
* CASI 20 / SLS 6 for inference after model selection
* SLS 5.2 for KKT conditions
* SLS 11 for theoretical resuls about lasso

#### For part 2 on neural networks

* ESL Chapter 11 (pre-deep, solid foundation)
* CASI Chapter 18
* [MLstory](https://mlstory.org/index.html) chapters on optimization, generalization, deep learning, and datasets
* [DLbook](https://www.deeplearningbook.org/), especially part II and chapters 6-9
* Jared Tanner's [course](https://courses.maths.ox.ac.uk/node/37111) on theories of DL
* [Paper](https://arxiv.org/abs/2102.11107) surveying applications to causality
* Beyond feed-forward networks: architectures like RNN, GAN, LSTM, transformer (lecture 10)

*Lab*: TBD
* Simulation study of lasso/ridge for high-dimensional regression 
* Inference for models selected by lasso
* The keras neural network API and R package

------
#### Week 8. Unsupervised learning - clustering

In this lecture we cover some commonly adopted clustering mehtods to find subgroups and clusters in a dataset. We start with k-means clustering, discussing k-means alogirthm, "overfit then merge" strategy, k-means++ algortihm, the selection of k, theoretical guarantees and high-dimensional clusterings. We then briefly discuss hierarchical clustering including the algorithm, types of linkage and similarity measures. Finally, we discuss spectral clustering using ideas related to eigenanalysis, including different types Laplacians, supporting theorems, the algorithm and illustrative examples.

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week8/ST510_Lecture%207%20slides.pdf).

*Readings*:
* [Textbook](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) on k-means and hierarchical clustering.
* [Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) on k-means++ algortihm.
* [Paper](https://www.tandfonline.com/doi/abs/10.1198/016214508000000454) on selecting the number of clusters.
* [Paper](https://mast.queensu.ca/~linder/pdf/LiLuZe94.pdf) on theoretical analysis of k-means clustering.
* [Paper](https://www.tandfonline.com/doi/abs/10.1198/jasa.2010.tm09415) on sparse k-means.
* [Tutorial](https://arxiv.org/pdf/0711.0189.pdf) on spectral clustering.

*Lab*:
 * Theoretical analysis of k-means clustering. [Exercise sheet](https://github.com/lse-st510/Lectures2021/blob/main/Week8/Seminar%207%20exercise.pdf) and [solution](https://github.com/lse-st510/Lectures2021/blob/main/Week8/Solution%20to%20Seminar%207%20exercise.pdf).
 * Implementing k-means clustering, hierarchical clustering and spectral clustering. [Python code](https://github.com/lse-st510/Lectures2021/blob/main/Week8/Seminar%207.ipynb).

------
#### Week 9. Unsupervised learning - dimension reduction

In this lecture we cover some popular strategies and algorithms for dimension reduction. We start with principle component analysis (PCA), discussing various ways of performing and interpreting PCA, and showing their equivalence. We then move on to show how the PCA can be extended in different directions, either by having more constraints, or by performing non-linear (instead of linear) transformation. In particular, we briefly discuss the key ideas behind sparse PCA, non-negative matrix factorization (NMF), multidimensional scaling (MDS), and autoencoder.

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week9/Lecture_slides.pdf)

*Readings*:
* Trevor Hastie, Robert Tibshirani, Jerome Friedman, The elements of statistical learning, Springer, 2009; book [here](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
* Youwei Zhang, Alexandre d'Aspremont and Laurent El Ghaoui, Sparse PCA: convex relaxations, algorithms and applications, 2010; paper [here](https://arxiv.org/abs/1011.3781) 
* Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray and Mark Crowley, Multidimensional scaling, sammon mapping, and isomap: tutorial and survey, 2020; paper [here](https://arxiv.org/abs/2009.08136) 
* Ian Goodfellow, Yoshua Bengio, and Aaron Courville, Deep learning, The MIT Press, 2016; book [here](https://www.deeplearningbook.org)
* James Gentle, Matrix algebra: theory, computations, and applications in statistics, Springer, 2007; book [here](https://www.springer.com/gp/book/9780387708720) (for free access - via LSE library)

*Lab*: 
* Using PCA, NMF, MDS and autoencoder in Python
* A discussion on Johnson-Lindenstrauss Lemma, see also a paper [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.6717&rep=rep1&type=pdf) 

------
#### Week 10. Online learning

In this lecture we discuss some fundamental results of online learning. We start with online classification where we introduce the basic concept of online learnability and the notion of Littlestone's dimension, prediction mistake bounds for various algorithms, and classic weighted majority algorithm and regret bounds derived by using this algorithm. We then consider online convex optimization and establish a regret bound of the online gradient descent algorithm. We finally discuss the online perceptron algorithm, and establish a bound for the prediction mistakes for this algorithm by showing that the perceptron algorithm is an online gradient descent algorithm defined for certain surrogate convex loss function. 

*Readings*:
* Shai Shalev-Shwartz and Shai Ben-David, Understanding Machine Learning: from Theory to Algorithms, Cambridge University Press, 2014; [online](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)
* Elad Hazan, Introduction to Online Convex Optimization, manuscript, 2019; [online](https://sites.google.com/view/intro-oco/). 
* Alexandrs Slivkins, Introduction to Multi-Armed Bandits, Now Publishers, 2019 (Chapter 5); [online](https://arxiv.org/pdf/1904.07272.pdf).

*Lab*:
* Weighted majority algorithm
* Adam optimizer
* Online perceptron algorithm

------
#### Week 11. Reinforcement learning

In this lecture we cover some popular reinforcement learning algorithms. We start with discussing applications that could benefit from applying reinforcement learning algorithms. We then introduce various basic concepts as well as the mathematical foundations of reinforcement learning. We next focus on one of the most popular class of reinforcement learning algorithms: Q-learning, and introduce some detailed algorithms such as tabular Q-learning, tabular SARSA, fitted Q-iteration and deep Q-network. Finally, we briefly talk about policy-based learning and highlight their difference from Q-learning.

* Lecture [slides](https://github.com/lse-st510/Lectures2021/blob/main/Week11/slides.pdf)

*Readings*: 
* Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, Second Edition, MIT Press, Cambridge, MA, 2018; text [here](http://www.incompleteideas.net/book/the-book-2nd.html)
* Martin L. Puterman, Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.


*Lab*: 
* An introduction of OpenAI Gym
* Q-learning
* Policy-based learning 
