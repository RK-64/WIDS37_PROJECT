# WIDS37_PROJECT
Week 1 – Data Loading and Initial Exploration
The first week is focused on getting familiar with the dataset and building a base pipeline.
Main steps covered:
Importing required Python libraries such as pandas, numpy, and matplotlib.
Uploading and loading the dataset directly into the notebook.
Inspecting the dataset structure (columns, data types, shape).
Checking for missing values and obvious inconsistencies.
Performing basic exploratory data analysis (EDA).
Visualizing simple patterns and distributions where required.
This notebook mainly helps in understanding what the data looks like and what kind of preprocessing will be needed later.

Week 2 – Data Cleaning and Preprocessing
The second week builds on the first by refining the dataset and preparing it for further analytical or modeling tasks.
Main steps covered:
Reloading the dataset in a structured way.
Handling missing or inconsistent values.
Cleaning and transforming relevant columns.
Feature-level inspection to understand usefulness and impact.
Removing unnecessary or redundant data where applicable.
Preparing a cleaner version of the dataset for downstream tasks.
By the end of Week 2, the data is in a much more usable and reliable form compared to the raw input.

Week 3 – Feature Engineering and Text Processing
Week 3 focuses on transforming raw textual and categorical information into meaningful features that can be used for building a recommendation system.
Main steps covered:
Loading the cleaned master dataset generated in Week 2.
Converting stringified JSON-like columns (cast, crew, keywords, genres) into Python objects.
Extracting important information such as the director name from the crew data.
Selecting the top cast members for each movie to reduce noise.
Extracting and cleaning keyword and genre names.
Normalizing text by converting to lowercase and removing spaces for consistency.
Applying stemming to keywords using NLP techniques.
Creating a combined textual feature (often called a “soup”) by merging keywords, cast, director, and genres.
This week plays a crucial role in preparing meaningful features that capture the content of each movie, which are essential for building a content-based recommendation system.

Week 4 – Dataset Refinement and Optimization
Week 4 is focused on refining the dataset further to make it efficient, lightweight, and suitable for similarity-based modeling.
Main steps covered:
Loading the feature-engineered dataset from Week 3.
Dropping unnecessary, redundant, or heavy metadata columns that are not required for recommendations.
Cleaning and validating numerical columns such as popularity.
Sorting movies based on popularity to maintain relevance.
Removing rows with missing or invalid critical information.
Resetting indices to maintain clean and consistent referencing.
Saving the final optimized dataset for model building.
By the end of Week 4, the dataset is streamlined, optimized, and ready to be directly used for implementing the recommendation system.

Week 5 – Building the Movie Recommendation System
Week 5 is the final and most important phase of the project, where the actual movie recommendation system is implemented.
Main steps covered:
Loading the finalized dataset prepared in Week 4.
Cleaning movie titles to ensure accurate matching during recommendations.
Converting textual features into numerical vectors using Count Vectorization.
Computing similarity between movies using cosine similarity.
Creating an index mapping between movie titles and dataset rows.
Implementing a function to recommend top N similar movies based on a given input movie.
Testing the recommendation system with example inputs and verifying the output.
This week brings together all the concepts learned throughout the project and demonstrates a complete, working content-based movie recommendation system using data science techniques.
