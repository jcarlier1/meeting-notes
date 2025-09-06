import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")

# Configure the decoding to use greedy decoding which is more stable
asr_model.cfg.decoding.strategy = "greedy"

transcript = asr_model.transcribe(["data/samples/data/samples/Lecture 1 EvoPsy_converted_60_120.mp3"])[0].text
print(transcript)