# ---未完成-----
from flask import Flask, request, jsonify
import torch

# 创建Flask应用实例
app = Flask(__name__) 

# 定义模型结构
def get_net(): 
    net = nn.Sequential(nn.Linear(in_features,1))
    return net 
model = get_net() 

# 加载训练好的参数
model.load_state_dict(torch.load("model_house_preds.pth"))  
model.eval()  # 切换为预测模式


# 定义路由与预测函数
@app.route('/predict', methods=['POST'])
def predict():
    # 1. 接收数据
    data = request.json  # 用户发送的JSON数据，例如 {'YearBuilt': 2020, 'HouseStyle': '1Story', ...}
    
    # 2. 检查输入是否合法
    if 'x' not in data:
        return jsonify({"error": "缺少参数x"}), 400
    
    try:
        x = float(data['x'])
    except ValueError:
        return jsonify({"error": "x必须是数字"}), 400
    
    # 3. 转换为Tensor并预测
    with torch.no_grad():  # 关闭梯度计算，加速预测
        tensor_x = torch.tensor([[x]], dtype=torch.float32)
        prediction = model(tensor_x).item()  # 执行预测
    
    # 4. 返回结果
    return jsonify({"input_x": x, "predicted_y": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)