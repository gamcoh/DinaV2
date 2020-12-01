from collections import Counter
import alsaaudio

class ActionsParser:
    def __init__(self, predictions: list):
        self.prediction = self._most_predict(predictions)
        if hasattr(self, self.prediction):
            getattr(self, self.prediction)()

    def _most_predict(self, predictions: list):
        counter = Counter(predictions)
        return counter.most_common(1)

    def sliding_two_fingers_down(self):
        audio = alsaaudio.Mixer()
        vol = int(audio.getvolume()[0])
        new_vol = vol - 10
        audio.setvolume(new_vol)

    def sliding_two_fingers_up(self):
        audio = alsaaudio.Mixer()
        vol = int(audio.getvolume()[0])
        new_vol = vol + 10
        audio.setvolume(new_vol)
