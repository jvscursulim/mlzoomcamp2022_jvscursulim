import pickle

from flask import Flask, jsonify, request

model_file_path = "/home/jvictor/vs_code/mlzoomcamp2022_jvscursulim/05-deployment/model/model1.bin"
dv_file_path = "/home/jvictor/vs_code/mlzoomcamp2022_jvscursulim/05-deployment/model/dv.bin"

with open(file=model_file_path, mode="rb") as model_file:
    model = pickle.load(model_file)
    
with open(file=dv_file_path, mode="rb") as dv_file:
    dv = pickle.load(dv_file)

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    
    client = request.get_json()
    
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    
    result = {"credit_card_probability": y_pred}
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4242)