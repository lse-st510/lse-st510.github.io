---
papersize: a4
documentclass: scrartcl
classoption: DIV=14
colorlinks: true
---

![LSE](images/lse-logo.jpg)
# ST510 Foundations of Machine Learning

### Lent Term 2020

### Instructors

* Milan Vojnovic, [M.Vojnovic@lse.ac.uk](mailto:M.Vojnovic@lse.ac.uk), Department of Statistics.  *Office hours*: by appointment
* Kostas Kalogeropoulos, [k.kalogeropoulos@lse.ac.uk](mailto:k.kalogeropoulos@lse.ac.uk), Department of Statistics.  *Office hours*: TBD
* Xinghao Qiao, [x.qiao@lse.ac.uk](mailto:x.qiao@lse.ac.uk), Department of Statistics. *Office hours*: TBD
* Yining Chen, [Y.Chen101@lse.ac.uk](mailto:Y.Chen101@lse.ac.uk), Department of Statistics. *Office hours*: TBD
* Joshua Loftus, Department of Statistics. *Office hours*: TBD
* Chengchun Shi, [c.shi7@lse.ac.uk](mailto:c.shi7@lse.ac.uk), Department of Statistics.  *Office hours*: TBD

### GTA

* Bento Natura, [b.natura@lse.ac.uk](mailto:b.natura@lse.ac.uk), Department of Mathematics. *Office hours*: TBD

* Please use **LSE Student Hub** to book slots for office hours.

### Course Information

* **Please send an email** to us with your **GitHub account name or email address (please use a name that does not reveal your identity)** if you would like to have access to the lecture/class materials. For auditing students, you are **not** allowed to attend online or on-campus seminars/classes.
* Lectures will be pre-recorded and uploaded to Moodle, **not** GitHub. 
* Classes will be given via a combination of online and on-campus sessions. 

No lectures or classes will take place during School Reading Week 6.

### Assessment

* Exam (40%, duration: 2 hours, reading time: 10 minutes) in the summer exam period.
* Project (40%, 3000 words) and take-home assessment (20%) in the LT.
* For the take-home assessments, students will be given homework problem sets and computer programming exercises in weeks 2, 4, 7, and 9.
* The project assesment will be in April. The project report should be no fewer than 3000 words and students will be asked to submit ther project reports within one week.
* Students are expected to produce 9 problem sets in the LT, including 4 take-home assessments and 5 formative assignments.

### Some Code/Suggestions for On-Campus Seminar/Class 

[LSE's Student Code for a COVID-Secure Campus](https://www.lse.ac.uk/international-relations/assets/documents/pdfs/Student-Code-for-a-COVID-Secure-Campus.pdf)

* **Wear masks** (except if medically exempted) and keep two meters social distancing. We will ask anyone without face coverings to leave the classroom. 
* Get **tested** on LSE campus. Book a test [here](https://info.lse.ac.uk/coronavirus-response/get-tested-at-lse). 
* Please do **not** attend on-campus seminar if tested positive or experiencing COVID symptoms. 
* It is totally fine to **not** attend on-campus seminars if you are worried about COVID. We have online seminar sessions and OHs. Seminar exercises, solutions and recordings will be uploaded online. 

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

*Readings*:
* Shai Shalev-Shwartz and Shai Ben-David, Understanding Machine Learning: from Theory to Algorithms, Cambridge University Press, 2014; text [here](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)
* Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar, Foundations of Machine Learning, 2nd Edition, The MIT Press, 2018
* Martin J. Wanwright, High-Dimensional Statistics: A Non-Asymptotic Viewpoint, Cambridge University Press, 2019 (Chapter 2, Basic tail and concentration inequalities)

*Lab*: 
* PAC learning
* Bias-complexity trade-off: polynomical regression example
* Rademacher complexity and growth function example
* VC dimension example

------
#### Week 2. Convex optimisation

In this week...

*Readings*:
* Stephen Boyd and Lieven Vandenberghe, Convex Optimization, Cambridge University Press, 2004; text [here](http://web.stanford.edu/~boyd/cvxbook)
* Sebastien Bubeck, Convex optimization: algorithms and complexity, Now Publishers Inc. 2016; text [here](http://sbubeck.com/Bubeck15.pdf)

*Lab*: TBD

------

#### Week 3. Non-convex optimisation

This week ...

*Readings*:

*Lab*: TBD

------
#### Week 4. Support vector machines

This week ...

*Readings*:

*Lab*: TBD

------

#### Week 5. Decision trees and random forests

This week ...

*Readings*:

*Lab*: TBD

------
#### Week 6. Reading Week


------
#### Week 7. Neural networks

This week ...

*Readings*: 
* Ian Goodfellow, Yoshua Bengio, and Aaron Courville, Deep Learning, The MIT Press, 2016
* Aston Zhang, Zack C. Lipton, Mu Li, and Alex J. Smola, Deep Dive into Deep Learning, 2020; text [here](https://d2l.ai/)

*Lab*: TBD

------
#### Week 8. Unsupervised learning - clustering

This week ...

*Readings*:

*Lab*: TBD

------
#### Week 9. Unsupervised learning - dimension reduction

This week ...

*Readings*:

*Lab*: TBD

------
#### Week 10. Online learning and optimisation

This week ...

*Readings*:

*Lab*: TBD

------
#### Week 11. Reinforcement learning

This week ...

*Readings*: Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, Second Edition, MIT Press, Cambridge, MA, 2018; text [here](http://www.incompleteideas.net/book/the-book-2nd.html)

*Lab*: TBD
