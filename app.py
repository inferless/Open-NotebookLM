from transformers import AutoModelForCausalLM, AutoTokenizer
from kokoro import KModel, KPipeline
from utils import convert_script_format, create_summarization_messages, create_podcast_conversion_messages, clean_podcast_script, clean_utterance_for_tts, extract_pdf_content
import time
import numpy as np
import soundfile as sf
import io
import base64
import inferless
from pydantic import BaseModel, Field

@inferless.request
class RequestObjects(BaseModel):
    pdf_url: str = Field(default="https://arxiv.org/pdf/2502.01068")

@inferless.response
class ResponseObjects(BaseModel):
    generated_podcast_base64: str = Field(default='Test output')

class InferlessPythonModel:
    def initialize(self):
        model_name = "Qwen/Qwen3-32B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="cuda")
        self.kmodel = KModel(repo_id='hexgrad/Kokoro-82M').to("cuda").eval()
        self.kpipeline = KPipeline(lang_code="a")
        self.MALE_VOICE = "am_adam"
        self.FEMALE_VOICE = "af_heart"

    def infer(self,request: RequestObjects) -> ResponseObjects:
        messages_content = extract_pdf_content(request.pdf_url)
        summary_content = self.generate_text(self.tokenizer, self.model,create_summarization_messages(messages_content))
        tts_content = self.generate_text(self.tokenizer, self.model,create_podcast_conversion_messages(summary_content))
        converted_script = convert_script_format(tts_content)
        all_audio = []
        for sr, audio_segment in self.generate_podcast_audio(converted_script, self.kmodel, self.kpipeline, self.MALE_VOICE, self.FEMALE_VOICE):
            all_audio.append(audio_segment)
            pause = np.zeros(int(sr * 0.5))
            all_audio.append(pause)

        if all_audio:
            final_audio = np.concatenate(all_audio)
            buf = io.BytesIO()
            sf.write(buf, final_audio, sr, format='WAV')
            buf.seek(0)
            base64_audio = base64.b64encode(buf.read()).decode('utf-8')
            
            generateObject = ResponseObjects(generated_podcast_base64=base64_audio)
            return generateObject
            

    def generate_text(self,tokenizer, model,text_content):
        tokenized_text = tokenizer.apply_chat_template(text_content,
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False
                                                )
        model_inputs = tokenizer([tokenized_text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content


    def generate_podcast_audio(self,podcast_script: str, kmodel, kpipeline, male_voice: str, female_voice: str):    
        pipeline_voice_male = kpipeline.load_voice(male_voice)
        pipeline_voice_female = kpipeline.load_voice(female_voice)
        
        speed = 1.0
        sr = 24000

        lines = clean_podcast_script(podcast_script)

        for i, line in enumerate(lines):            
            if line.startswith("[Alex]"):
                pipeline_voice = pipeline_voice_male
                voice = male_voice
                utterance = line[len("[Alex]"):].strip()
            elif line.startswith("[Romen]"):
                pipeline_voice = pipeline_voice_female
                voice = female_voice
                utterance = line[len("[Romen]"):].strip()
            else:
                continue
            
            if not utterance.strip():
                continue

            utterance = clean_utterance_for_tts(utterance)
            
            try:
                for _, ps, _ in kpipeline(utterance, voice, speed):
                    ref_s = pipeline_voice[len(ps) - 1]
                    audio_numpy = kmodel(ps, ref_s, speed).numpy()
                    yield (sr, audio_numpy)
            except Exception as e:
                continue
