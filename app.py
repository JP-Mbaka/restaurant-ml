"""
 @author: Mbaka JohnPaul

 """

from http.client import HTTPException
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import asyncio
from dict import Recommendation, Recommendation1
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
df = pd.read_excel('data.xlsx')
sdf= pd.DataFrame()

# Rename Columns for ease
df.columns = ['time','age','gender','class','breakfast', 'lunch', 'dinner', 'dislikes','allergies','comments']
# df.head()

# Step 1: Extract all unique food items
all_foods = set()
def setFood(col):
  for item in col:
    foods = [food.strip() for food in item.split(',')]
    for e in foods:
        if e != "":
            all_foods.add(e.lower())
            
setFood(df['breakfast'])
setFood(df['lunch'])
setFood(df['dinner'])
setFood(df['dislikes'])
setFood(df['allergies'])

sdf['meal_time'] = ['breakfast', 'lunch', 'dinner']

# Step 3: Create new columns for each food item
for food in all_foods:
    for meal_time in ['breakfast', 'lunch', 'dinner']:
        col_name = f"{food}"
        sdf[col_name] = 0

# Iterate over meals and update sdf
for i, meal in enumerate(['breakfast', 'lunch', 'dinner']):
    for meal_entry in df[meal]:
        items = [i.strip().lower() for i in meal_entry.split(',')]
        for food in items:
            if food in sdf.columns:
                # sdf.at[i, food] += 1
                try:
                    sdf.loc[i, food] += 1
                except KeyError:
                    print(f"Row {i} or column '{food}' not found in sdf")
                
# Content-based filtering Method
def get_food_freq_sorted(sdf, meal_time):
    # Select the row for the given meal_time
    row = sdf[sdf['meal_time'] == meal_time]
    if row.empty:
        return f"No data for meal_time: {meal_time}"

    # Drop the meal_time column to keep only food columns
    food_counts = row.drop(columns=['meal_time']).iloc[0]

    # Sort ascending by frequency
    sorted_foods = food_counts.sort_values(ascending=False)

    return sorted_foods

# Collaborative-based filtering Method
# Prepare the food matrix: rows=meal_times, columns=foods, values=counts
food_matrix = sdf.set_index('meal_time')
# food_matrix = sdf['meal_time']


# Compute item-item similarity matrix (foods similarity)
item_sim = cosine_similarity(food_matrix.T)
item_sim_df = pd.DataFrame(item_sim, index=food_matrix.columns, columns=food_matrix.columns)

def recommend_by_collab(meal_time, food_name, top_n=15):
    if food_name not in item_sim_df.columns:
        return f"Food '{food_name}' not found."

    # Get similar foods to the input food
    sim_scores = item_sim_df[food_name].sort_values(ascending=False)

    # Exclude the input food itself
    sim_scores = sim_scores.drop(food_name)

    # Top N similar foods
    top_foods = sim_scores.head(top_n)
    return top_foods

# Hybrid-based filtering method
def recommend_by_content(sdf, meal_time, top_n=15):
    row = sdf[sdf['meal_time'] == meal_time]
    if row.empty:
        return f"No data for meal_time: {meal_time}"

    food_counts = row.drop(columns=['meal_time']).iloc[0]

    # Sort foods by descending frequency (most frequent foods first)
    recommended = food_counts.sort_values(ascending=False)

    return recommended.head(top_n)

def hybrid_recommendation(sdf, meal_time, liked_food, top_n=15):
    # Get top foods in meal_time (should be Series)
    top_foods = recommend_by_content(sdf, meal_time, top_n=20)
    if isinstance(top_foods, str):
        return top_foods  # No data message from recommend_by_content

    # Make sure top_foods is a Series with index = food names
    if not isinstance(top_foods, pd.Series):
        top_foods = pd.Series(top_foods)

    # Get foods similar to liked_food (should be Series)
    similar_foods = recommend_by_collab(meal_time, liked_food, top_n=top_n*2)
    if isinstance(similar_foods, str):
        return similar_foods  # No data message from recommend_by_collab

    # Make sure similar_foods is a Series with index = food names
    if not isinstance(similar_foods, pd.Series):
        similar_foods = pd.Series(similar_foods)

    # Debug prints to verify types and indices
    # print(f"Top foods index: {top_foods.index}")
    # print(f"Similar foods index: {similar_foods.index}")

    # Filter similar foods to those popular in meal_time
    filtered = similar_foods[similar_foods.index.isin(top_foods.index)].head(top_n)

    # Convert to JSON-friendly format (list of dicts with food and score)
    recommendations = [{"food": food, "score": float(score)} for food, score in filtered.items()]

    response = {
        "Top foods index": top_foods.index,
        "Similar foods index": similar_foods.index,
        "meal_time": meal_time,
        "liked_food": liked_food,
        "recommendations": recommendations
    }

    return response

@app.get('/')
def index():
    return {'message': 'Hello, welcome to Student-Performance-ML'}

@app.post('/content')
def predict_performance(data: Recommendation1):
    data = data.dict()
    mealTime = data['mealTime']

    res = get_food_freq_sorted(sdf, mealTime)  # this returns a pandas Series

    # Convert pandas Series to dict with native Python types
    res_dict = res.to_dict()
    clean_res = {k: int(v) for k, v in res_dict.items()}  # ensure all values are int

    result = {'result': clean_res}
    json_compatible = jsonable_encoder(result)

    return JSONResponse(content=json_compatible)

@app.post('/collaborate')
def predict_performance(data: Recommendation):
        data = data.dict()
        mealTime = data['mealTime']
        item = data['item']
            
        res = recommend_by_collab(mealTime, item)
        
            # Convert pandas Series to dict with native Python types
        res_dict = res.to_dict()
        clean_res = {k: int(v) for k, v in res_dict.items()}  # ensure all values are int
       
        result = {'result': clean_res}
        json_compatible = jsonable_encoder(result)

        return JSONResponse(content=json_compatible)
    
@app.post('/hybrid')
def predict_performance(data: Recommendation):
    data = data.dict()
    mealTime = data['mealTime']
    item = data['item']

    res = hybrid_recommendation(sdf, mealTime, item)

    # If response is an error message (str), return it directly
    if isinstance(res, str):
        return JSONResponse(content={"message": res}, status_code=404)

    # Optional: convert any non-serialisable values (e.g., pandas Index) in the dict
    # e.g., convert Index to list
    res["Top foods index"] = list(res["Top foods index"])
    res["Similar foods index"] = list(res["Similar foods index"])

    json_compatible = jsonable_encoder(res)
    return JSONResponse(content=json_compatible)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")

