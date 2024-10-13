from transformers import VitsModel, AutoTokenizer
import torch
import os

# Đường dẫn tương đối tới thư mục chứa mô hình
current_dir = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục hiện tại của file Python
model_dir = os.path.join(current_dir, "models/facebook/mms-tts-kor")  # Đường dẫn tương đối tới thư mục "models"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(model_dir, exist_ok=True)

model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

text = "농부와 관한 특별한 기억이 있습니다. 제 할아버지는 농부셨고, 저는 그와 함께한 많은 아름다운 기억이 있습니다. 여름마다 저는 고향에 가서 할아버지와 함께 밭에서 일하며, 채소를 심고 수확하는 일을 도왔습니다. 할아버지는 항상 노동의 가치와 땅에 대한 사랑을 가르쳐 주셨습니다. 이러한 경험은 저에게 자연에 대한 깊은 감정을 느끼게 해주었고, 농업의 중요성을 깨닫게 해주었습니다. 제가 할아버지와 함께 처음으로 채소를 수확했을 때의 특별한 기억이 있습니다. 그때의 기분은 정말 즐겁고 자랑스러웠습니다."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

import scipy
# Chuyển đổi output từ torch.Tensor sang numpy array
output_numpy = output.squeeze().cpu().numpy()
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output_numpy)


# from transformers import AutoProcessor, VitsForConditionalGeneration
# import torch
# import soundfile as sf
# import os

# # Đường dẫn tương đối tới thư mục chứa mô hình
# current_dir = os.path.dirname(os.path.abspath(__file__))  # Lấy thư mục hiện tại của file Python
# model_dir = os.path.join(current_dir, "models/facebook/mms-tts-kor")  # Đường dẫn tương đối tới thư mục "models"

# # Tạo thư mục nếu chưa tồn tại
# os.makedirs(model_dir, exist_ok=True)

# # Load processor và model từ thư mục chỉ định
# processor = AutoProcessor.from_pretrained("facebook/mms-tts-kor", cache_dir=model_dir)
# model = VitsForConditionalGeneration.from_pretrained("facebook/mms-tts-kor", cache_dir=model_dir)

# # Văn bản tiếng Hàn cần chuyển thành giọng nói
# text = "안녕하세요, 오늘 기분이 어떠세요?"

# # Tokenize input text
# inputs = processor(text, return_tensors="pt")

# # Generate speech (waveform)
# with torch.no_grad():
#     speech = model.generate(**inputs)

# # Lưu kết quả thành file âm thanh
# waveform = speech.squeeze().cpu().numpy()
# sf.write("output.wav", waveform, 16000)

# print(f"Đã tải mô hình từ {model_dir} và lưu tệp âm thanh dưới dạng output.wav")
