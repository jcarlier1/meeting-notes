import nemo.collections.asr as nemo_asr

# Use a smaller, more compatible model
asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_conformer_ctc_small")

transcript = asr_model.transcribe(["Lecture 1 EvoPsy_converted_60_120.mp3"])[0].text
print(transcript)
