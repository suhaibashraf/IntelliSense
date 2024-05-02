from speech_emotion_model import SpeechEmotionModel
import sounddevice as sd
from scipy.io.wavfile import write
from device_control import LightControlV2
import time
import os


def main(training=False):
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    model = SpeechEmotionModel()
    light = LightControlV2()
    if training:
        path = 'ravdess-data/'
        model.load_data(path)
        model.preprocessing()
        model.generate_model()
        model.start_training(32, 1000)
        model.save_model('saved_models/speech_model')
    else:
        model.load_model('saved_models/speech_model')
        emotion_list = sorted(list(set(model.labels)))
        colors = [[0, 255, 0], [255, 255, 0], [0, 0, 255], [100, 136, 234], [255, 255, 255], [253, 251, 211], [255, 244, 229], [225, 165, 0], [0, 255, 0], [255, 255, 0], [0, 0, 255], [100, 136, 234], [255, 255, 255], [253, 251, 211], [255, 244, 229], [225, 165, 0]]
        light_colors = dict(zip(emotion_list, colors))
        try:
            while True:
                myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()  # Wait until recording is finished
                write('output.wav', fs, myrecording)
                emotion = model.predict_emotions('output.wav')[0]
                light.set_light(light_colors[emotion])
                time.sleep(5)
        except KeyboardInterrupt:
            os.remove("output.wav")
            pass


if __name__ == '__main__':
    main(False)
