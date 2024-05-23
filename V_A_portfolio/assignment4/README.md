# Assignment 4 - Detecting faces in historical newspapers

## IMPORTANT
Before running any of the assignments Please run the following sh scripts:
../setup/create_venv.sh to ensure the virtual environment is created
then type " source ../Visual_venv/bin/activate " into the terminal to activate it
then run ../setup/setup.sh to install necessary libraries

in case ../setup/setup.sh is not working, all the packages used in making the assignments can be found in requirements.txt

## Description

If you have ever looked at old newspapers, you might notice that they are very verbose. There is a lot of text, perhaps some illustrations - but, ultimately, they are very text heavy data sources. However, this begins to change in the 19th century with the advent of new technology and with the adoption of personal cameras in the 20th century, images become increasingly dominant.

In this assignment, we're going to build on this idea to look for changing patterns in print media. Specifically, we are going to look at the presence in historical newspapers of *pictures of human faces*. This is a culturally meaningful question - how has the prevelance of images of human faces changed in print media over the last roughly 200 years? Are there any significant differences and what might this mean?

We're going to work with a corpus of historic Swiss newspapers: the *Journal de Genève* (JDG, 1826-1994); the *Gazette de Lausanne* (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). You can read more about this corpus in the associated reserch article (linked to below).

You should write code which does the following:

- For each of the three newspapers
    - Go through each page and find how many faces are present
    - Group these results together by *decade* and then save the following:
        - A CSV showing the total number of faces per decade and the percentage of pages for that decade which have faces on them
        - A plot which shows the latter information - i.e. percentage of pages with faces per decade over all of the decades avaiable for that newspaper
- Repeat for the other newspapers

Finally, remember your repository should include a writtens summary and interpretation of what you think this analysis might being showing. You do not need to be an expert in the history of printed Swiss media - just describe what you see and what that might mean in this context. Make sure also to mention any possible limitations of your approach!


## Starter code

For this task, we are going to use a pretrained CNN model which has been finteuned for face detection. You can see documentation of this model and some starter code about how to get it running at [this website](https://medium.com/@danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144). In particular, you'll want to use the first code block down to the line which detects faces in images:

```python
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Load an image containing faces
img = Image.open('path_to_image.jpg')

# Detect faces in the image
boxes, _ = mtcnn.detect(img)
```

The shape of the variable ```boxes``` can then be used to tell you how many faces are on the page.

## Data access

The data for the assignment is available in the shared drive on UCloud. For the purposes of this assignment, you can link to [this version](https://zenodo.org/records/3706863) in your README files.

## Tips

- Notice that the filenames contain the name of the newspaper, and the year-month-date of publication. This will be useful for you!

## Purpose

- To demonstrate that you can pretrained CNNs to extract meaningful information from image data
- To convey the kinds of datasets and problems encountered doing image processing for cultural analytics
- To show understanding of how to interpret machine learning outputs

###################################################################################################

### Structure
- data : the data folders should have the newspapers orrganized in folders similarly to this structure: data/IMP, data/GDL, data/JDG. These folders shpuld have the newspaper images similarly to this structure: data/IMP/IMP-1882-05-04-a-p0001.jpg, etc.
- src: the python file used for generating results
- out: the output .csv files and plots for the face data

### Code
The scriptis based on a few functions.
After the face recognition CNN is setup,

count_faces() loads a single image, runs the CNN on the image, the takes the number of blocks as the number of faces, since the CNN works by drawing boxes around faces. So an output shaped like (2,4) means the page contains 2 faces with 4 reference points for the 4 corners of the box. Additional relevant image information, like decade and newspaper title is derived from the image filename. output is saved as a single row of pandas

count_faces_folder() iterates through all images in a folder, it is used to iterate through all 3 subfolders. 

count_faces_all() iterates through the data folder and runs count_faces_folder() on the respective repositories. Data is the processed and dataframes are concatinated.

make_df() Results are saved to pandas dataframes which are then organized by decade and page %s are added based on task description.

Outputs are then visualized and saved.

### Results
####  individual Papers
![Alt text](out/ "n_faces ")
![Alt text](out/ "n_faces ")
![Alt text](out/ "n_faces IMP")


![Alt text](out/ "Percentage")
![Alt text](out/ "Percentage")
![Alt text](out/ "Percentage IMP")

As time increases, so does the number of faces present in printed media.

The initial jump in valami could be explained by the small number of available pages from that decade, which by chance could have had some illustrations.

Percentage of pages with pictures also seems to increase, so it seems pictures


### Concerns
#### Data
The datasets are not entirely balanced, as mentioned in the task description
- (JDG, 1826-1994),
- (GDL, 1804-1991),
- (IMP, 1881-2017)
The overlapping period, where we have published versions of all papers is between 1881-1991.
It would be maybe wise therefore to only make comparisons among the papers in that period if the research question lies in that domain. For example, just to demonstrate my point without explicit knowledge of Swiss printed media, if I were to investigate whether more conservative newspapers are less "human" oriented and therefore have less printed faces, I would only examine the period, where comparison is possible. It would be awkward to find later, that I spent my academic career pursuing the notion, that in 1850 no faces were found in conservative papers, just because my data had no conservative papers from 1850.

Also, there are filenames, which dates from th 1700s, so something might be off about when these papers were available.

Another point is, that results might not generalize well. The collection of papers varies by number of pages by decade, and there are only 3 newspapers in the corpus. So results might be descriptive of these 3 papers, but might not mean this trend is present in all printed Swiss media, let alone all printed media from around the globe. I think it is also safe to assume, that the sheer number of available printed papers increased with time, so maybe the dataset should be controlled for that as well.
#### Method
This is an unsupervised task. It is unknown which pages do contain faces, and how many do they contain. We have to trust in the performance of the pre-trained model. When evaulating findings, we can not rely on a classification report and a training set.

A task like this highlights the importance of supervised training on models and making sure these models are efficient enough before utilizing them in unsupervised tasks. For example, based on the classification report of assignment 3, I would not recommend using that model on unsupervised Tobacco Company document classification.

Additional steps could be taken to ensure model validity, for example a smaller chunk of the data could be manually noted for faces then the model could be tested on that chunk, but since the whole point of this approach is to automate data collection without using unnecessary manpower, that might not be the best tactique.

The documentation also mentions confidence scores as possible outputs. Using these scores, dubious cases of face recognition could be manually checked to spot possible mistakes. 