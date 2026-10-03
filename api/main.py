from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

app = FastAPI(title="xG Shot Prediction API")


class ShotData(BaseModel):
    distance_to_goal: float
    shot_angle: float
    defenders_in_cone: int
    distance_to_keeper: float
    under_pressure: int
    shot_first_time: int
    body_part_name_Left_Foot: bool
    body_part_name_Other: bool
    body_part_name_Right_Foot: bool
    technique_name_Diving_Header: bool
    technique_name_Half_Volley: bool
    technique_name_Lob: bool
    technique_name_Normal: bool
    technique_name_Overhead_Kick: bool
    technique_name_Volley: bool
    play_pattern_name_From_Counter: bool
    play_pattern_name_From_Free_Kick: bool
    play_pattern_name_From_Goal_Kick: bool
    play_pattern_name_From_Keeper: bool
    play_pattern_name_From_Kick_Off: bool
    play_pattern_name_From_Throw_In: bool
    play_pattern_name_Other: bool
    play_pattern_name_Regular_Play: bool


@app.post("/predict")
def predict(data: ShotData):
    input_df = pd.DataFrame([{
        "distance_to_goal": data.distance_to_goal,
        "shot_angle": data.shot_angle,
        "defenders_in_cone": data.defenders_in_cone,
        "distance_to_keeper": data.distance_to_keeper,
        "under_pressure": data.under_pressure,
        "shot_first_time": data.shot_first_time,
        "body_part_name_Left Foot": data.body_part_name_Left_Foot,
        "body_part_name_Other": data.body_part_name_Other,
        "body_part_name_Right Foot": data.body_part_name_Right_Foot,
        "technique_name_Diving Header": data.technique_name_Diving_Header,
        "technique_name_Half Volley": data.technique_name_Half_Volley,
        "technique_name_Lob": data.technique_name_Lob,
        "technique_name_Normal": data.technique_name_Normal,
        "technique_name_Overhead Kick": data.technique_name_Overhead_Kick,
        "technique_name_Volley": data.technique_name_Volley,
        "play_pattern_name_From Counter": data.play_pattern_name_From_Counter,
        "play_pattern_name_From Free Kick": data.play_pattern_name_From_Free_Kick,
        "play_pattern_name_From Goal Kick": data.play_pattern_name_From_Goal_Kick,
        "play_pattern_name_From Keeper": data.play_pattern_name_From_Keeper,
        "play_pattern_name_From Kick Off": data.play_pattern_name_From_Kick_Off,
        "play_pattern_name_From Throw In": data.play_pattern_name_From_Throw_In,
        "play_pattern_name_Other": data.play_pattern_name_Other,
        "play_pattern_name_Regular Play": data.play_pattern_name_Regular_Play,
    }])

    probability = model.predict_proba(input_df)[0][1]

    return {
        "probabilite_de_but": round(float(probability), 4)
    }