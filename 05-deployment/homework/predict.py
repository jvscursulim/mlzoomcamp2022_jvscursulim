import pickle

from flask import Flask, jsonify, request

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    
    with open(file="model1.bin", mode="rb") as model_file:
        model = pickle.load(model_file)
        
    with open(file="dv.bin", mode="rb") as dv_file:
        dv = pickle.load(dv_file)
        
    client = request.get_json()
    
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    
    result = {"credit_card_probability": y_pred}
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4242)