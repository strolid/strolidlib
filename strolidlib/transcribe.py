"""
Transcription code from ~/alpha-weird/transcribe.py

Copy the following functions from alpha-weird/transcribe.py:
- load_parakeet_110m
- load_en
- load_canary_180m
- load_non_en
- transcribe_batch
- transcribe (main worker function)
"""

# TODO: Copy functions from ~/alpha-weird/transcribe.py

# From transcribe.py lines 33-37:
# def load_parakeet_110m():
#     model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-110m", refresh_cache=False)
#     model.eval()
#     model = move_to_gpu_maybe(model)
#     return model

# From transcribe.py lines 40-41:
# def load_en():
#     return load_parakeet_110m()

# From transcribe.py lines 44-48:
# def load_canary_180m():
#     model = ASRModel.from_pretrained(model_name="nvidia/canary-180m-flash", refresh_cache=False)
#     model.eval()
#     model = move_to_gpu_maybe(model)
#     return model

# From transcribe.py lines 51-52:
# def load_non_en():
#     return load_canary_180m()

# From transcribe.py lines 55-113:
# def transcribe_batch(model, batch, language):
#     """Transcribe all tensors in a batch at once"""
#     
#     valid_tensors = []
#     valid_indices = []
#     
#     for i, tensor in enumerate(batch['tensors']):
#         if tensor.numel() == 0:
#             batch['tensor_info'][i]['transcription'] = ""
#             batch['tensor_info'][i]['confidence'] = 0.0
#         else:
#             if len(tensor.shape) > 1:
#                 tensor = tensor.squeeze()
#             valid_tensors.append(tensor)
#             valid_indices.append(i)
#     
#     if not valid_tensors:
#         return batch
#     
#     if hasattr(model, 'device') and hasattr(model, '_model') and hasattr(model._model, 'device'):
#         device = model._model.device
#         valid_tensors = [tensor.to(device, non_blocking=True) for tensor in valid_tensors]
#     
#     config = {}
#     if language == "en":
#         pass
#     else:
#         config.update({
#             "task": "asr",
#             "pnc": "yes"
#         })
#     
#     if config:
#         transcriptions = model.transcribe(valid_tensors, **config)
#     else:
#         transcriptions = model.transcribe(valid_tensors)
#     
#     for tensor_idx, batch_idx in enumerate(valid_indices):
#         tensor_info = batch['tensor_info'][batch_idx]
#         
#         if tensor_idx < len(transcriptions):
#             transcription = transcriptions[tensor_idx]
#             
#             if hasattr(transcription, 'text'):
#                 text = transcription.text
#             elif isinstance(transcription, str):
#                 text = transcription
#             else:
#                 text = str(transcription)
#             
#             confidence = getattr(transcription, 'confidence', 1.0)
#             
#             tensor_info['transcription'] = text.strip()
#             tensor_info['confidence'] = confidence
#         else:
#             tensor_info['transcription'] = ""
#             tensor_info['confidence'] = 0.0
#     
#     return batch

# From transcribe.py lines 116-153:
# def transcribe(must_transcribe_queue, must_stitch_queue, language, ready_event=None):
#     
#     if language == "en":
#         model = load_en()
#     else:
#         model = load_non_en()
#     if debug:
#         print(cuda_context_id())   
#     try:
#         if ready_event:
#             ready_event.set()
#         while True:
#             try:
#                 batch = must_transcribe_queue.get(timeout=queue_get_timeout)
#                 if batch is None:
#                     # Re-enqueue sentinel for other workers and exit loop
#                     must_transcribe_queue.put(None)
#                     break
#                     
#                 if batch:
#                     transcribed_batch = transcribe_batch(model, batch, language)
#                     must_stitch_queue.put(transcribed_batch)
#             except:
#                 continue
#                 
#     except KeyboardInterrupt:
#         if debug:
#             print(f"Transcribe ({language}) shutting down")
#     
#     # Signal downstream that this worker is done
#     try:
#         must_stitch_queue.put(None)
#     except Exception:
#         pass
#     
#     if debug:
#         print(f"Transcribe ({language}) done")

