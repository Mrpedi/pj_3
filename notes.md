## Project Submission

For this project, a reviewer will be testing the model that you generated on the first test track (the one to the left in the track selection options).

Whether you decide to zip up your submission or submit a GitHub repo, please follow the naming conventions below to make it easy for reviewers to find the right files:

<b>model.py</b> - The script used to create and train the model.

<b>drive.py</b> - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version.
<b>model.json</b> - The model architecture.

<b>model.h5</b> - The model weights.

README.md - explains the structure of your network and training approach. While we recommend using English for good practice, writing in any language is acceptable (reviewers will translate). There is no minimum word count so long as there are complete descriptions of the problems and the strategies. See the rubric for more details about the expectations.
You can review the rubric for the project here.


### Quality of Code

<b>CRITERIA</b>

MEETS SPECIFICATIONS

1) Is the code functional?

2) The model provided can be used to successfully operate the simulation.

3) Is the code usable and readable?

4) The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

### Model Architecture and Training Strategy

<b>CRITERIA</b>

MEETS SPECIFICATIONS
Has an appropriate model architecture been employed for the task?

The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

Has an attempt been made to reduce overfitting of the model?

Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.

Have the model parameters been tuned appropriately?

Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

Is the training data chosen appropriately?

Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

### Architecture and Training Documentation

<b>CRITERIA</b>

MEETS SPECIFICATIONS

Is the solution design documented?

The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

Is the model architecture documented?

The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Is the creation of the training dataset and training process documented?

The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.

### Simulation

<b>CRITERIA</b>

MEETS SPECIFICATIONS
Is the car able to navigate correctly on test data?

No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).