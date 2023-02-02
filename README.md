# Refactor Predict Customer Churn Model


## Project Description
In this project, I will walk you through the refactoring of a jupyter notebook for a customer churn prediction model. The scenario is as follows: a bank manager grows increasingly perturbed by the number of customers cancelling their credit cards and so has consulted us to build a model that will predict the likelihood of a customer churning so that management can make a data-informed decision on how to intervene.

In this project, I will refactor the notebook by creating a modular script where each step of the machine learning process is encapsulated in a function that can be easily reused. I will also perform tests and logging in order to ensure that the code is up to PEP-8 standards.

The dataset for this project was downloaded from Kaggle, and can be found <a href="https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers">here</a>.

An example of the structure of the data can be seen in the image below:

<img src='sample-data.png'>

## Files and data description
Overview of the files and data present in the root directory. 
Here is the file structure of the project:

<pre><code class="lang-bash">.
├── Guide.ipynb          <span class="hljs-comment"># Given: Getting started and troubleshooting tips</span>
├── churn_notebook.ipynb <span class="hljs-comment"># Given: Contains the code to be refactored</span>
├── churn_library.py     <span class="hljs-comment"># <span class="hljs-doctag">ToDo:</span> Define the functions</span>
├── churn_script_logging_and_tests.py <span class="hljs-comment"># <span class="hljs-doctag">ToDo:</span> Finish tests and logs</span>
├── README.md            <span class="hljs-comment"># <span class="hljs-doctag">ToDo:</span> Provides project overview, and instructions to use the code</span>
├── data                 <span class="hljs-comment"># Read this data</span>
│   └── bank_data.csv
├── images               <span class="hljs-comment"># Store EDA results </span>
│   ├── eda
│   └── results
├── logs                 <span class="hljs-comment"># Store logs</span>
└── models               <span class="hljs-comment"># Store models</span>
</code></pre>

## Running Files
<!--How do you run your files? What should happen when you run your files?-->
If you would like to reproduce this project and try it out on your own, please follow these steps:

## Step 1:
Clone this repo on to your local machine using the following command in your terminal or command line prompt: <code> git clone </code>. Make sure you're in the folder in which you want to clone the repo. For example, I have mine in a subdirectory under Documents entitled Projects. Hit me up if you would like guidance on this or any other aspect of the project.



