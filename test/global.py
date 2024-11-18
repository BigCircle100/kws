# 假设这是 SoundClassification 类的定义
class SoundClassification:
    def __init__(self):
        self.model_loaded = False

    def load_model(self):
        self.model_loaded = True
        print("Model loaded.")

    def classify(self, sound):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded.")
        # 这里可以添加实际的分类逻辑
        print(f"Classifying sound: {sound}")
        return "classified_result"

# 声明全局变量 client
client: SoundClassification

def classify_sound(sound):
    """使用 client 对象进行声音分类"""
    global client  # 声明使用全局变量
    result = client.classify(sound)
    return result

if __name__ == "__main__":
    global client  # 声明使用全局变量

    # 手动初始化 client
    client = SoundClassification()
    client.load_model()  # 加载模型

    # 使用 client 对象进行声音分类
    sound_input = "example_sound.wav"
    classification_result = classify_sound(sound_input)
    print("Classification result:", classification_result)
