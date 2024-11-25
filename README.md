# Recommendation System Final Project

## Overview
This project is the final implementation of a recommendation system. It aims to provide personalized recommendations to users based on their preferences and past interactions.

## Installation
(**Must use Python 3.11**)

Clone this repo:
```
git clone https://github.com/FireFly177/recommendation_system_final.git
cd recommendation_system_final
```

Create python virtual environment:
```
python -m venv .venv  
```

Activate python virtual environment:

On Windows: `.\.venv\Scripts\activate`
On Linux/MacOS: `source .venv/Scripts/activate`

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
### Use the recommendation page only:
To run the page and get recommendation, execute:
```bash
python app.py
```

and access `localhost:5000` on browser


### Train the model to get new weights:
Download and process data:
```
python data_processing.py
```

Train the model:
```
python train.py
```
After training, the weight of the model will be save in the `weights.keras` file


