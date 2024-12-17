

## SEP 769 Part 2 of Project - Generate Realistic Human Face using GAN  <br>

Authors			    :	Kyaw Kyaw Oo, Hariharan Mohan, Faizan Parabtani<br>
Student Numbers	    : 	400551761, 400608376, 400489257<br>
Group No			:   5	<br>
Date of Submission	: 	Nov 28, 2024<br>
Lecturer			:	Dr. Alireza Fatehi <br>
                        Dr. Amir Pourhajiaghashafti <br>

CopyRight@2024 by Group 5 of SEP 769 Part 2<br>

# Generate Realistic Human Face using GAN 

## Overview
The goal of this project is to build a generative model, in particular a GAN to create images of faces using the CelebA dataset that contains more than twenty thousand face images of celebrities with their attributes. To achieve this, the primary objective is to train the GAN to generate realistic-style images which are then realistically photorealistic with a focus on the realistic style portrayal of human faces in any conceivable emotion. GAN mainly includes a generator that generates synthetic image set and a discriminator that evaluates their authenticity. The generator enhances image production while the discriminator enhances the identification ability between the real image and the synthetic images during the Adversarial training. To evaluate the qualities of generated images, two indices, Inception Score (IS) and Fréchet Inception Distance (FID) will be used. Further experiments will focus on other attributes and features of the face including additional possible expressions to further demonstrate the versatility of the GAN synthesizing capability. This project facilitates the development of the existing field of generative modeling as it proves that GANs are efficient in the generation of high fidelity faces and assesses some of the capabilities of the network for mimicking intricate human features. 

## Prerequisites
- Python 3.9.6+
- Jupyter Notebook
- Celebe Dataset - https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- pip (Python package installer)

## Installation

1. # Clearing the unnecessary packages due to the dependency issues
%pip uninstall keras tensorflow tensorflow-addons -y

%pip install tensorflow scipy numpy pillow
%pip install pydot graphviz
%pip install --upgrade pydot

2. Create local directories of real_images where to save and selected images only due to the constraint of the computation and generated_images

# Global Variables
# Constants
Real_Images_DIR = current_directory + '/Data/Images'
IMAGES_COUNT = 100   # Adjust based on available data due to the computational constraints in our local machine
WIDTH = 64           # Image width after resizing
HEIGHT = 64          # Image height after resizing
LATENT_DIM = 32      # Latent dimension for GAN
CHANNELS = 3         # Number of channels (RGB)
CONTROL_SIZE_SQRT = 6
RES_DIR = 'res2'
FILE_PATH = '%s/generated_%d.png'

# Predefined Modules
oad_images_from_folder
preprocess_image
extract_features
get_real_features
generate_images
get_generated_features
display_images
create_generator
create_discriminator
create_gan
train_gan_classification
save_images_to_directory

## Visualization
plot_losses

## Metrics
inception_score
calculate_fid

## Testing 
real_images
generator
discriminator
gan
generated_features
d_losses, a_losses, classification_df
generated_fake_images
inceptfid_score
ion_score_value
precision, recall
