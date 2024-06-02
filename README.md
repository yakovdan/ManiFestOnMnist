
# Few Sample Feature Selection via Feature Manifold Learning

This repository contains an implementation of ManiFest algorithm, executed on the MNIST dataset. 

## Prerequisites
 - Octave.
To run the code, GNU Octave is required. GNU Octave is a free and open source syntax compatible alternative to Matlab.
To install GNU Octave in a Linux environment, run: *sudo apt install octave*
The code depends on **manopt**, a manifold optimization package. Is is available for download at: [manopt](https://www.manopt.org/download.html)
Once downloaded, unzip the file wherever is convenient and run the script importmanopt from Octave. 
 - Python environment.
 To run the  code, a python environment is recommended. A python environment can be created via **venv** or **conda**
	 - **venv**
	 Create an environment by calling *python -m venv ManiFest* in the git clone directory. Then, requirements can be installed using *pip install -r requirements.txt*
	 - **conda**
Create a conda environment: *conda env create -f environment.yaml*

## Running the code
If not already activated, activate the environment. Change directory to the git clone directory.
Run: *python MultiClassMNIST.py*

if VISUALIZE is true, illustrative figures will be saved to disk. score_{percentile}.png contains the ManiFest score vector for the particular chosen percentile as scale factor
{class_id}_{percentile}.png contains the eigenvector with the largest eigenvalue for each class, for the particular chosen percentile 


