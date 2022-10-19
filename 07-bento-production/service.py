import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    country: str
    rating: float

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
model_runner = model_ref.to_runner()
svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data):
    return model_runner.run(input_data)

# model_ref = bentoml.xgboost.get("credit_risk_model:latest")
# dv = model_ref.custom_objects["dictVectorizer"]

# model_runner = model_ref.to_runner()

# svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

# @svc.api(input=JSON(pydantic_model=UserProfile), output=JSON())
# def classify(credit_application):
#     application_data = credit_application.dict()
#     vector = dv.transform(application_data)
#     prediction = model_runner.predict.run(vector)
#     print(prediction)
    
#     result = prediction[0]
    
#     if result > 0.5:
#         return {"status": "DECLINED"}
#     elif result > 0.25:
#         return {"status": "MAYBE"}
#     else:
#         return {"status": "APPROVED"}