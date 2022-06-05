import os
import uvicorn
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException
from typing import List, Union, Optional
from pydantic import BaseModel, conlist


class PredictResponse(BaseModel):
    id: int
    target: int


class InputDataRequest(BaseModel):
    data: List[conlist(Union[float, str, None])]
    features: List[str]


app = FastAPI()
model: Optional[Pipeline] = None


@app.on_event('startup')
async def load_model():
    global model
    model_path = os.getenv('PATH_TO_MODEL', default='model/svm_model.pkl')
    with open(model_path, 'rb') as model:
        model = pickle.load(model)


@app.get('/health')
def health() -> bool:
    if not model:
        return False
    return hasattr(model, 'classes_')


def make_predict(data: List, features: List[str]) -> List[PredictResponse]:
    df = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in df["id"]]
    pred = model.predict(df)
    return [
        PredictResponse(id=index, target=target) for index, target in zip(ids, pred)
    ]


@app.get("/predict/", response_model=List[PredictResponse])
def predict(request: InputDataRequest):
    if health():
        return make_predict(request.data, request.features)
    else:
        raise HTTPException(status_code=404, detail='not found or not fitted model')


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=os.getenv("PORT", 8000))

