SkinTypeClassification
======================

The Skin Type Classification project leverages advanced machine learning techniques to accurately classify skin types according to the renowned Baumann Skin Type Classification System. This system, developed by dermatologist Dr. Leslie Baumann, is widely recognized for its comprehensive approach to categorizing skin into distinct types based on various attributes such as sensitivity, moisture retention, pigmentation and wrinkls.

This project aims to provide an efficient tool for both skincare professionals and enthusiasts to determine their Baumann skin type quickly and accurately. By utilizing a diverse dataset of facial images, along with carefully curated feature extraction methods, this model can effectively differentiate between the various Baumann skin types, enabling personalized skincare recommendations and routines.

How to use?
-----------
To run the app locally use command:
> uvicorn app.app:app --host 0.0.0.0 --port XXXX
<p><small>where XXXX - is parameter, specifying the port on which the server will be started (usually 80)</small></p>
<p>Then make a POST request to localhost/macro with an image file of facial skin texture in its body</p>

Where to see?
-----------
You can see the realisation in **[Telegram bot](https://t.me/BeautyScienceFaceAnalysisBot)**

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
