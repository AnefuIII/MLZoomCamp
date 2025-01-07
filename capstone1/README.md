# CAPSTONE 1 PROJECT; FAKE (AI) VS REAL IMAGE DETECTOR
This project is done as a capstone project for machine learning zoomcamp by 
https://datatalks.club/. The course is hosted for free on <a href ="https://www.youtube.com/watch?v=8wuR_Oz-to0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=1">youtube.</a>

## Demo Video

<a href = "https://youtu.be/cS6oa99W7rg" target = "_blank">Demo Video (Youtube)</a>

## CIFAKE Data source 
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download

## Dataset description

The dataset contains two classes; REAL and FAKE. This project used a subset of the dataset.

We used 1,400 images for training (700 per class) and 600 for testing (300 per class)

** Dataset References **

Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.

Bird, J.J. and Lotfi, A., 2024. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.

Real images are from Krizhevsky & Hinton (2009), fake images are from Bird & Lotfi (2024).

for more info about data check <a href = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download"> here.</a>

## INSTRUCTION TO USE MODEL
** REQUIREMENTS **
- Python 3.9
- Docker

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AnefuIII/MLZoomCamp.git
   cd MLZoomCamp/capstone1/
   ```
2. **EDA & ModeL Training in google colaboratory**
    ```bash
    jupyter notebook capstone1/CapstoneCNN.ipynb
    jupyter notebook capstone1/Capstone1_tflite.ipynb
    ```

3. **Set Up a Virtual Environment (powershell recommended)**
    ```bash
    cd capstone1
    pip install pipenv
    pipenv install --python C:\Users\HP\AppData\Local\Programs\Python\Python39\python.exe # Python 3.9 location
    pipenv shell`
    pipenv install numpy tensorflow keras-image-helper
    ```

4. **Model Reproduce & Deployment**
    ```bash
    pipenv shell 
    python # to enter and run python 3.9
    import lambda_function
    lambda_function.predict('url_of_image')
    lambda_function.lambda_handler(event = {'url':'url/path/to/image', None)
    ```

5. **Containerization**
    ```bash
    # docker was installed and leave pipenv i.e powershell not activated 
    cd capstone1
    docker build -t aivsrealimagemodel .
    docker run -it --rm -p 8080:8080 aivsrealimagemodel:latest
    python test.py # test model pipeline in another terminal such as VS Code
    ```



