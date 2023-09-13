Skin Types Classification
======================

The Skin Type Classification project leverages advanced machine learning techniques to accurately classify skin types according to the renowned Baumann Skin Type Classification System. This system, developed by dermatologist Dr. Leslie Baumann, is widely recognized for its comprehensive approach to categorizing skin into distinct types based on various attributes such as sensitivity, moisture retention, pigmentation and wrinkles.

This project aims to provide an efficient tool for both skincare professionals and enthusiasts to determine their Baumann skin type quickly and accurately. By utilizing a diverse dataset of facial images, along with carefully curated feature extraction methods, this model can effectively differentiate between the various Baumann skin types, enabling personalized skincare recommendations and routines.

Technological Stack
-------------------
- Tensorflow
- Keras
- FastAPI
- DVC (Data Version Control)
- MLflow

How to use?
-----------

### Try ready model:

1. Download this project
2. Create venv
3. Install all dependencies
> pip install -r requirements.txt
4. Run uvicorn
> uvicorn app.app:app --host 0.0.0.0 --port 80
<p><small>where 80 - is parameter, specifying the port on which the server will be started</small></p>
<p>Then make a POST request to localhost:80/macro with an image file of facial skin texture in its body</p>
<p>Example of skin image:</p>

![SkinImage](image.jpg)

<p>Response body example</p>

>{
> 
> "skin_type": "OSPT",
>
> "short_info": "While the majority of OSPT skin types struggle with acne and related pigmentation problems, following a consistent skincare regimen aimed at preventing acne rather than treating lesions and scars will leave this skin type with a healthy, clear complexion and even skin tone."
>
>}

### Reproduce learning process:

1. Download this project
2. Create venv
3. Install all dependencies
> pip install -r requirements-dev.txt
4. **[Download dataset](https://drive.google.com/file/d/1UmGOhudwfi0SKUs9X74ubDdFFh8VZ0dJ/view?usp=sharing)** and put all 4 directories (`oily_dry`, `pigmented_skin`, `sensitive_skin`, `wrinkled_smooth`) into `data/raw` project's directory
5. Add your project to git by running `git init` in root directory
6. Run
> dvc repro
7. To see experiment metrics, run
> mlflow ui

and move to browser

**NB!** Make sure that the data you track in `dvc` you do not version in `git`

Where to see?
-----------
You can see the realisation in **[Telegram bot](https://t.me/BeautyScieneFaceAnalysisBot)**

Docker
------

also, you can try this project in a **[docker container](https://hub.docker.com/repository/docker/kudrikmed/skin-type/general)**

Metrics
--------

![MetricsImage](metrics.png)

Author
------

**Dzmitry Kudrytski**, PhD, MD, Plastic Surgeon, and Data Scientist

For inquiries, please contact me via **[email](kudrikmed@gmail.com)**