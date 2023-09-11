Skin Type Classification documentation
==============================================
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

- 1. Download this project
- 2. Create venv
- 3. Install all dependencies with `pip install -r requirements.txt`
- 4. Run uvicorn `uvicorn app.app:app --host 0.0.0.0 --port XXXX`
- 5. Then make a POST request to localhost/macro with an image file of facial skin texture in its body

Where to see?
-----------
You can see the realisation in **Telegram bot** https://t.me/BeautyScieneFaceAnalysisBot

Docker
------

also, you can try this project in a **docker container** https://hub.docker.com/repository/docker/kudrikmed/skin-type/general


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   app
   src

