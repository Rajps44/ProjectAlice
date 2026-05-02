#  Copyright (c) 2026
#
#  This file, OllamaNlu.py, is part of Project Alice (Zoya Integration).
#
#  Project Alice is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

import json
import requests
from core.nlu.model.NluEngine import NluEngine
from core.util.Logger import Logger

class OllamaNlu(NluEngine):
    """
    Zoya's Brain Bridge: Connects Project Alice to Ollama API.
    Uses DeepSeek-R1 for reasoning and intent extraction.
    """

    def __init__(self):
        super().__init__()
        # Ollama local API endpoint
        self._url = "http://localhost:11434/api/generate"
        # Aapka model name (Aap isse DeepSeek-R1 ya Llama-3 par change kar sakte hain)
        self._model = "deepseek-r1:14b"
        self.logInfo(f"Zoya Engine initialized using Ollama model: {self._model}")

    def parse(self, text: str):
        """
        Input text ko Ollama ke paas bhejta hai aur Intent extract karta hai.
        """
        self.logInfo(f"Zoya is thinking about: {text}")

        # System Prompt: Jo DeepSeek ko 'Zoya' banata hai
        system_prompt = (
            "You are Zoya, the NLU engine of Project Alice. "
            "Your job is to extract the intent from the user's speech. "
            "Respond ONLY in valid JSON format like this: "
            "{'intent': 'IntentName', 'probability': 1.0, 'slots': {}}"
        )

        payload = {
            "model": self._model,
            "prompt": f"{system_prompt}\n\nUser said: {text}",
            "stream": False,
            "format": "json" # Ollama ko force karega JSON dene ke liye
        }

        try:
            response = requests.post(self._url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                raw_answer = result.get('response', '{}')
                
                # JSON parse karke Alice ke format mein return karna
                parsed_data = json.loads(raw_answer)
                return parsed_data
            else:
                self.logError(f"Ollama connection failed with status: {response.status_code}")
                return None
        except Exception as e:
            self.logError(f"Zoya Brain Error: {str(e)}")
            return None

    def train(self, forceLocalTraining: bool = False):
        # DeepSeek ko training ki zarurat nahi hai, wo pehle se smart hai!
        self.logInfo("Zoya doesn't need training. She is born smart!")
        return True

    def stop(self):
        self.logInfo("Zoya engine stopping...")
        super().stop()
      
