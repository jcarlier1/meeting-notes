import nemo.collections.asr as nemo_asr

# Load the ASR model with CUDA graphs disabled
asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")

# Disable CUDA graphs to avoid compilation errors
asr_model.cfg.decoding.use_cuda_graph_decoder = False

transcript = asr_model.transcribe(["data/samples/Lecture 1 EvoPsy_converted_60_120.mp3"])[0].text
print(transcript)
