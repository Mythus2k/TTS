import re, pyaudio, wave, os, argparse
from time import sleep
from TTS.api import TTS
from torch import cuda

voices = {"default":r"C:\Users\Mythus2k\Documents\Code\yuko_hub\tts\transformers\default_7.wav", 
          'yuko': r"C:\Users\Mythus2k\Documents\Code\yuko_hub\tts\transformers\yuko_1.wav",
          'yuko_alt': r"C:\Users\Mythus2k\Documents\Code\yuko_hub\tts\transformers\yuko_4.wav"}

parser = argparse.ArgumentParser(description="Runs wav files for blocks of text from Yuko's responses")
parser.add_argument("response", help="string - pass the completed response from yuko to be read aloud")
parser.add_argument("voice", help="string (optional) - which voice tranformer to use", nargs='?')

args = parser.parse_args()

class running_tts():
    def __init__(self, voice='default') -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.voice = voice
        self.completed_wavs = list()
    
    def stream_sentences(self, script):
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_endings, script)

        for i, sentence in enumerate(sentences):
            path = f"C:/Users/Mythus2k/Documents/Code/yuko_hub/tts/temp/{i}_temp.wav"
            yield sentence, path

    def play_wav(self, filepath):
        """
        Plays a WAV file using PyAudio to the default speaker device.

        Args:
            filepath (str): Path to the WAV file.
        """
        try:
            wf = wave.open(filepath, 'rb')
        except FileNotFoundError:
            print(f"Error: WAV file not found at {filepath}")
            return

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(1024)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf.close()

        print(f"Finished playing: {filepath}")

    def stream_response(self, response):
        for sentence, path in self.stream_sentences(response):
            self.tts.tts_to_file(
                text=sentence, 
                speaker_wav=voices[self.voice], 
                language="en",
                file_path=path)
            self.play_wav(path)
            self.completed_wavs.append(path)

        sleep(1)
        self.clear_temp()

    def clear_temp(self):
        for path in self.completed_wavs:
            os.remove(path)

        self.completed_wavs = list()
    

if __name__ == '__main__':
    if args.voice == None: args.voice = 'default'

    tts_module = running_tts(args.voice)
    tts_module.stream_response(args.response)
