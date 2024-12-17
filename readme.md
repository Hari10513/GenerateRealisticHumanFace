# **Generate Realistic Human Face using GAN**

## **Project Information**
- **Authors**: Hariharan Mohan, Kyaw Kyaw Oo, Faizan Parabtani  
- **Group No**: 5  
- **Date of Submission**: Nov 28, 2024  
- **Lecturers**:  
  - Dr. Alireza Fatehi  
  - Dr. Amir Pourhajiaghashafti  

**CopyRight © 2024 by Group 5 of SEP 769 Part 2**

---

## **Overview**
The purpose of this project is to build a **Generative Adversarial Network (GAN)** that generates realistic human face images using the **CelebA dataset**. The GAN model comprises:  
- **Generator**: Produces synthetic face images.  
- **Discriminator**: Differentiates between real and synthetic images.

    <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/Architecture.jpg" width="500">
    </p>

By training the GAN on thousands of celebrity face images, the generator learns to produce photorealistic faces. The quality of generated images is evaluated using **Inception Score (IS)** and **Fréchet Inception Distance (FID)**.

---

## **Prerequisites**

Ensure you have the following installed on your system:  
- **Python 3.9.6+**  
- **Jupyter Notebook**  
- **TensorFlow** and associated libraries  
- **CelebA Dataset**: [Download here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
- **pip** (Python package installer)  

---

## **Installation**

Follow these steps to set up and run the project:  

1. **Uninstall Conflicting Packages**  

   ```bash
   %pip uninstall keras tensorflow tensorflow-addons -y
  
3. **Install Required Libraries**

   ```bash
    %pip install tensorflow scipy numpy pillow
    %pip install pydot graphviz --upgrade

4. **Create Project Directories**

    ```python
    import os
    
    current_directory = os.getcwd()
    Real_Images_DIR = current_directory + '/Data/Images'
    Generated_Images_DIR = current_directory + '/generated_images'
    os.makedirs(Real_Images_DIR, exist_ok=True)
    os.makedirs(Generated_Images_DIR, exist_ok=True)

5. Global Configuration
    
    ```python
    # Configuration Parameters
    IMAGES_COUNT = 100      # Adjust dataset size based on resources
    WIDTH = 64              # Image width (resized)
    HEIGHT = 64             # Image height (resized)
    LATENT_DIM = 32         # Latent dimension for the generator input
    CHANNELS = 3            # RGB color channels
    CONTROL_SIZE_SQRT = 6   # Grid size for visualization
    RES_DIR = 'res2'        # Directory to save results
    FILE_PATH = '%s/generated_%d.png'

---

## **Modules and Functions**

1. Dataset and Preprocessing
    - load_images_from_folder: Load images from the dataset directory.
    - preprocess_image: Resize and normalize images for training.
    - extract_features and get_real_features: Feature extraction functions for real images.

2. GAN Components
    - create_generator: Builds the generator network.
    - create_discriminator: Defines the discriminator network.
    - create_gan: Combines generator and discriminator models for adversarial training.
    - train_gan_classification: Trains the GAN on the dataset.
  
   <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/FlowImage.png" width="500">
    </p>

3. Visualization
    - display_images: Display generated and real images side-by-side.
    - plot_losses: Visualize generator and discriminator losses over training epochs.

4. Evaluation Metrics
    -inception_score: Measures image diversity and realism.
    - calculate_fid: Computes Fréchet Inception Distance (FID) to assess image quality.

---

## **How to Run the Project**

1. Train the GAN Model
  
    ```python
    d_losses, g_losses = train_gan_classification(generator, discriminator, epochs=100)

  <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/GeneratorParameters.png" width="500">
  </p>

2. Generate and Display Images using the generator to create synthetic faces:

    ```python
    generated_images = generate_images(generator, LATENT_DIM)
    display_images(real_images, generated_images)

  <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/DiscriminatorParameters.png" width="500">
  </p>

3. Evaluate the Results
  Compute IS and FID scores for generated images:

    ```python
    IS = inception_score(generated_images)
    FID = calculate_fid(real_images, generated_images)
    print(f"Inception Score: {IS}, FID: {FID}")

---

## **Visualization**
1. Example Outputs

   <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/SampleImages.png" width="800">
    </p>
3. Generated Faces

   <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/anim.gif" width="500">
    </p>

5. Training Loss Plot

    <p align="center">
      <img src="https://github.com/Hari10513/GenerateRealisticHumanFace/blob/main/Images/Results.png" width="500">
    </p>

---
## **Evaluation Results**

After training the GAN model and generating synthetic faces, the following evaluation metrics were computed:

| **Metric**                         | **Value**  |
|------------------------------------|------------|
| **Inception Score (IS)**           | 1.2379     |
| **Fréchet Inception Distance (FID)** | 3.3769     |
| **Precision**                      | 0.0496     |
| **Recall**                         | 0.0471     |
| **Kernel Inception Distance (KID)** | 0.0125     |

---

## **Metrics Interpretation:**

- **Inception Score (IS)**: A value of **1.2379** indicates moderate diversity and realism, but there’s still room for improvement.
- **Fréchet Inception Distance (FID)**: The value of **3.3769** suggests that the generated images are relatively close to the real images, but fine-tuning is necessary for better performance.
- **Precision**: A precision value of **0.0496** indicates that a small portion of the generated data aligns with the real data distribution.
- **Recall**: The recall value of **0.0471** shows that the generator has limited coverage of the real data.
- **Kernel Inception Distance (KID)**: A **0.0125** KID score is relatively low, indicating good alignment between the real and generated data distributions.

---

## **Future Enhancements**
- Train the model for higher-resolution image generation (e.g., 256x256).
- Integrate Conditional GANs (cGANs) to generate faces with specific attributes.
- Optimize training performance for larger datasets.
- Explore additional evaluation metrics like LPIPS for perceptual similarity.

---

## **Contributors**
- Hariharan Mohan
- Kyaw Kyaw Oo
- Faizan Parabtani

---

## **License**

**© 2024 Group 5 of SEP 769 Part 2. All rights reserved.**

---

## **Contact**
For inquiries, please reach out via:

- Email: hariharan10513@gmail.com
- GitHub Repository: https://github.com/Hari10513

---
