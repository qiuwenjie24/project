# ---δ���-----
from flask import Flask, request, jsonify
import torch

# ����FlaskӦ��ʵ��
app = Flask(__name__) 

# ����ģ�ͽṹ
def get_net(): 
    net = nn.Sequential(nn.Linear(in_features,1))
    return net 
model = get_net() 

# ����ѵ���õĲ���
model.load_state_dict(torch.load("model_house_preds.pth"))  
model.eval()  # �л�ΪԤ��ģʽ


# ����·����Ԥ�⺯��
@app.route('/predict', methods=['POST'])
def predict():
    # 1. ��������
    data = request.json  # �û����͵�JSON���ݣ����� {'YearBuilt': 2020, 'HouseStyle': '1Story', ...}
    
    # 2. ��������Ƿ�Ϸ�
    if 'x' not in data:
        return jsonify({"error": "ȱ�ٲ���x"}), 400
    
    try:
        x = float(data['x'])
    except ValueError:
        return jsonify({"error": "x����������"}), 400
    
    # 3. ת��ΪTensor��Ԥ��
    with torch.no_grad():  # �ر��ݶȼ��㣬����Ԥ��
        tensor_x = torch.tensor([[x]], dtype=torch.float32)
        prediction = model(tensor_x).item()  # ִ��Ԥ��
    
    # 4. ���ؽ��
    return jsonify({"input_x": x, "predicted_y": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)