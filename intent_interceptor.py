import json
import queue
import re
import threading
import sys
from datetime import datetime
import ollama

class IntentInterceptor:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self.history = []
        self.history_lock = threading.Lock()
        
        # Trigger keywords (case-insensitive checks)
        self.keywords = [
            "bug", "issue", "action item", "action-item", 
            "todo", "to-do", "ticket", "track a", "create a", 
            "fix a", "log a", "task"
        ]

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print("Intent interceptor thread spawned.")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.queue.put(None)  # Sentinel to unblock and exit
        if self.thread:
            self.thread.join(timeout=3.0)
            self.thread = None
        print("Intent interceptor thread joined.")

    def process_text(self, text, notes_manager):
        # Clean segment and split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Perform keyword pre-filtering to avoid unnecessary LLM invocations
            matched = False
            lower_sentence = sentence.lower()
            for kw in self.keywords:
                if kw in lower_sentence:
                    matched = True
                    break
            
            with self.history_lock:
                # Capture preceding context block (last 2 sentences)
                context = " ".join(self.history[-2:])
                self.history.append(sentence)
                if len(self.history) > 15:
                    self.history.pop(0)
            
            if matched:
                print(f"[IntentInterceptor] Keyword match in: '{sentence}'. Queueing evaluation...")
                self.queue.put({
                    "sentence": sentence,
                    "context": context,
                    "notes_manager": notes_manager
                })

    def _worker_loop(self):
        while self.running:
            task_data = self.queue.get()
            if task_data is None:  # Stop signal
                break
                
            sentence = task_data["sentence"]
            context = task_data["context"]
            notes_manager = task_data["notes_manager"]
            
            self._evaluate_intent(sentence, context, notes_manager)
            self.queue.task_done()

    def _evaluate_intent(self, sentence, context, notes_manager):
        system_prompt = (
            "You are a meeting assistant. You are analyzing text to detect if the speaker is requesting to "
            "track a bug, log a task/action item, or create an issue. Analyze the context and the sentence.\n"
            "Respond ONLY with a JSON object. If a task/bug needs to be tracked, set 'is_task' to true. "
            "Otherwise, set 'is_task' to false.\n\n"
            "JSON Format Schema:\n"
            "{\n"
            "  \"is_task\": boolean,\n"
            "  \"task\": \"Clean description of the action item in third person (omit conversational fluff)\",\n"
            "  \"type\": \"bug\" | \"change\" | \"action\",\n"
            "  \"priority\": \"High\" | \"Medium\" | \"Low\"\n"
            "}\n"
            "Do not output markdown codeblocks, prefix text, or conversational lines. Output ONLY raw valid JSON."
        )
        
        user_prompt = f"Context: {context}\nSentence to evaluate: {sentence}"
        
        try:
            # Query local Ollama instance with format constraint
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json"
            )
            
            content = response.get("message", {}).get("content", "").strip()
            if not content:
                return
                
            # Parse payload
            data = json.loads(content)
            
            if data.get("is_task"):
                task_desc = data.get("task", "").strip()
                task_type = data.get("type", "action").strip()
                task_priority = data.get("priority", "Medium").strip()
                
                if task_desc:
                    print(f"\n[Agent Action] 🎯 Intent Intercepted:\n"
                          f"  - Task: {task_desc}\n"
                          f"  - Type: {task_type}\n"
                          f"  - Priority: {task_priority}\n")
                    
                    # Prepend task to top section of markdown notes
                    notes_manager.add_action_item(task_desc, task_type, task_priority)
                    
                    # Dispatch to external API mock webhook
                    self._dispatch_webhook(task_desc, task_type, task_priority)
                    
        except json.JSONDecodeError as jde:
            print(f"[IntentInterceptor] JSON parsing error from Ollama response: {jde}. Raw content was: '{content}'", file=sys.stderr)
        except Exception as e:
            print(f"[IntentInterceptor] Ollama evaluation error: {e}", file=sys.stderr)

    def _dispatch_webhook(self, task, type_, priority):
        try:
            # Webhook mock simulation
            payload = {
                "event": "localecho_task_triggered",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": {
                    "task": task,
                    "type": type_,
                    "priority": priority
                }
            }
            print(f"--- WEBHOOK TRIGGERED ---")
            print(json.dumps(payload, indent=2))
            print(f"-------------------------")
        except Exception as e:
            print(f"[IntentInterceptor] Webhook error: {e}", file=sys.stderr)
