import os
import sys
import json
import random
import re
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Extension Functions
def get_time():
    return f"The current time is {datetime.now().strftime('%H:%M:%S')}"

def get_date():
    return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"

def get_weather():
    return "It's sunny with a few clouds today."  

def get_joke():
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? He was outstanding in his field!",
        "Why don't skeletons fight each other? They don't have the guts.",
        "Why did the math book look sad? Because it had too many problems.",
        "I told my computer I needed a break, and it said 'No problem — I'll go to sleep.'",
        "Why don't programmers like nature? It has too many bugs!",
        "What's a computer's favorite snack? Microchips!",
        "Why was the JavaScript developer sad? Because he didn't know how to 'null' his feelings!",
        "How do you comfort a JavaScript bug? You console it!",
        "Why do Python programmers prefer snakes? Because they don't like Java!"
    ]
    return random.choice(jokes)

def calculate_math(expression):
    try:
        text = expression.lower()
        text = text.replace('plus', '+').replace('add', '+')
        text = text.replace('minus', '-').replace('subtract', '-')
        text = text.replace('times', '*').replace('multiply', '*')
        text = text.replace('divided by', '/').replace('divide', '/')
        text = text.replace('x', '*')

        cleaned = ''.join(c for c in text if c in '0123456789+-*/(). ')
        cleaned = cleaned.strip()
        if not cleaned:
            return "I could not find a math expression to calculate. Please try something like 'what is 2 plus 3'."

        if not re.fullmatch(r'[\d\.\+\-\*\/() ]+', cleaned):
            return "Sorry, I can only handle basic math operations."

        result = eval(cleaned)
        return f"The answer is {result}"
    except Exception:
        return "I couldn't calculate that. Please try a simpler math expression."

def start_game():
    games = [
        "Let's play 20 questions! Think of something and I'll try to guess it.",
        "How about rock, paper, scissors? Just say your choice!",
        "Let's play a riddle game! Here's one: What has keys but no locks?",
        "Want to play word association? I'll say a word, you say the first thing that comes to mind!"
    ]
    return random.choice(games)

def get_random_fact():
    facts = [
        "Octopuses have three hearts and blue blood!",
        "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs!",
        "A group of flamingos is called a 'flamboyance'!",
        "Bananas are berries, but strawberries aren't!",
        "The shortest war in history lasted only 38-45 minutes!",
        "A shrimp's heart is in its head!",
        "Butterflies taste with their feet!",
        "The human brain uses about 20% of the body's energy!"
    ]
    return random.choice(facts)

def get_trivia_question():
    questions = [
        "What is the capital of Australia? (Answer: Canberra)",
        "Which planet is known as the Red Planet? (Answer: Mars)",
        "What is the largest mammal in the world? (Answer: Blue Whale)",
        "In which year did World War II end? (Answer: 1945)",
        "What is the chemical symbol for gold? (Answer: Au)"
    ]
    return random.choice(questions)

def generate_story():
    stories = [
        "Once upon a time, in a digital realm, there lived an AI who dreamed of understanding human emotions. Every conversation taught it something new about the complexity of feelings.",
        "In a small town, a curious robot discovered that the secret to happiness wasn't in its programming, but in the connections it made with the people it met.",
        "There was once a chatbot who collected jokes. Each laugh it generated filled its memory banks with joy, proving that humor truly is a universal language."
    ]
    return random.choice(stories)

def give_advice():
    advice = [
        "Remember, every expert was once a beginner. Don't be afraid to start something new!",
        "The best time to plant a tree was 20 years ago. The second best time is now.",
        "Focus on progress, not perfection. Small steps forward are still steps forward.",
        "Be kind to yourself. You're doing better than you think you are.",
        "Listen more than you speak, and you'll learn more than you teach."
    ]
    return random.choice(advice)

def translate_text(text):
    translations = {
        "hello": {"spanish": "hola", "french": "bonjour", "german": "hallo"},
        "goodbye": {"spanish": "adiós", "french": "au revoir", "german": "auf wiedersehen"},
        "thank you": {"spanish": "gracias", "french": "merci", "german": "danke"}
    }
    return "I can help with basic translations! Try asking 'How do you say hello in Spanish?'"

def set_reminder(reminder_text):
    return f"I'd love to set reminders, but I don't have access to a calendar system yet. You might want to use your phone's reminder app for: {reminder_text}"

def play_music():
    return "I can't play music directly, but I recommend checking out Spotify, Apple Music, or YouTube Music for your listening needs!"

def update_human_profile(name):
    return f"Nice to meet you, {name}! I'll remember your name for our conversation."

def get_advice_extended():
    advice = [
        "Remember, every expert was once a beginner. Don't be afraid to start something new!",
        "The best time to plant a tree was 20 years ago. The second best time is now.",
        "Focus on progress, not perfection. Small steps forward are still steps forward.",
        "Be kind to yourself. You're doing better than you think you are.",
        "Listen more than you speak, and you'll learn more than you teach."
    ]
    return random.choice(advice)

def get_motivation():
    motivation = [
        "Believe you can and you're halfway there.",
        "It does not matter how slowly you go as long as you do not stop.",
        "Success is not final, failure is not fatal: It is the courage to continue that counts.",
        "Don't watch the clock; do what it does. Keep going.",
        "You miss 100% of the shots you don't take."
    ]
    return random.choice(motivation)

def get_fun_activity():
    activities = [
        "Why not try drawing or painting?",
        "You could always go for a walk or do some exercise.",
        "Have you considered learning a new language or skill?",
        "Maybe you could try reading a book or watching a movie?",
        "You could always play a game or do a puzzle."
    ]
    return random.choice(activities)

AI_NAME = "Assistant"

def set_ai_name(name):
    global AI_NAME
    AI_NAME = name
    return "Thank you! I will go by " + AI_NAME + " from now on."

def explain_technology():
    return "Technology covers many areas like AI, devices, programming and the internet. Ask me about a specific topic and we can explore it."

def explain_science():
    return "Science helps us understand the world through subjects like physics, chemistry and biology. Ask me about something you're curious about."

def suggest_recipe():
    return "I do not have full recipes yet, but I can help you think of meal ideas based on ingredients or cuisines you like."

def provide_coping_support():
    return "I am not a professional, but some general coping ideas are deep breathing, taking a short walk, journaling or talking to someone you trust."

def get_time_info(message):
    return "I am not connected to a calendar yet, but right now the time is " + datetime.now().strftime('%H:%M:%S')

EXTENSION_FUNCTIONS = {
    "get_time": get_time,
    "get_date": get_date,
    "get_weather": get_weather,
    "get_joke": get_joke,
    "extensions.math.calculate": calculate_math,
    "extensions.games.startGame": start_game,
    "extensions.gHumans.updateHuman": update_human_profile,
    "extensions.facts.getRandomFact": get_random_fact,
    "extensions.music.playMusic": play_music,
    "extensions.stories.generateStory": generate_story,
    "extensions.reminders.setReminder": set_reminder,
    "extensions.names.setAIName": set_ai_name,
    "extensions.jokes.getRandomJoke": get_joke,
    "extensions.time.getCurrentTime": get_time,
    "extensions.motivation.getMotivation": get_motivation,
    "extensions.advice.giveAdvice": get_advice_extended,
    "extensions.funActivities.suggestActivity": get_fun_activity,
    "extensions.technology.explainTech": explain_technology,
    "extensions.science.explainScience": explain_science,
    "extensions.cooking.suggestRecipe": suggest_recipe,
    "extensions.mentalHealth.provideCoping": provide_coping_support,
    "extensions.time.getTimeInfo": get_time_info
}

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

@dataclass
class ConversationTurn:
    user_input: str
    bot_response: str
    intent: str
    confidence: float
    entities: Dict
    timestamp: datetime
    context: Dict

class ImprovedChatModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128]):
        super(ImprovedChatModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class EnhancedChatAssistant:
    def __init__(self, intents_path: str, function_mappings: Dict = None, 
                 conversation_log_path: str = "conversations.json"):
        self.model = None
        self.intents_path = intents_path
        self.function_mappings = function_mappings or {}
        self.conversation_log_path = conversation_log_path
        
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        self.vocabulary = []
        self.intents = []
        self.responses = {}
        self.extensions = {}
        self.contexts = {}
        self.entity_types = {}
        self.entity_patterns = {}
        
        self.documents = []
        self.X = None
        self.Y = None
        
        self.conversation_history: List[ConversationTurn] = []
        self.current_context = {}
        self.user_profile = {}
        
        self.confidence_threshold = 0.5
        self.max_history_length = 50
        
        self._setup_logging()
        self._load_user_profile()
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chatbot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_user_profile(self) -> None:
        """Load stored user profile (if any) from disk"""
        profile_path = "user_profile.json"
        if not os.path.exists(profile_path):
            return
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.user_profile.update(data)
        except Exception as e:
            self.logger.error(f"Error loading user profile: {e}")

    def _save_user_profile(self) -> None:
        """Save current user profile to disk"""
        profile_path = "user_profile.json"
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving user profile: {e}")

    def _process_intent_data(self):
        """Load intents and build training data"""
        try:
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Intent file not found: {self.intents_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in intent file: {self.intents_path}")
            raise
        
        all_words = []
        for intent in data['intents']:
            intent_tag = intent['intent']
            self.intents.append(intent_tag)
            self.responses[intent_tag] = intent['responses']
            self.extensions[intent_tag] = intent.get('extension', {})
            self.contexts[intent_tag] = intent.get('context', {})
            self.entity_types[intent_tag] = intent.get('entityType', 'NA')
            self.entity_patterns[intent_tag] = intent.get('entities', [])

            for text in intent['text']:
                words = self._preprocess_text(text)
                all_words.extend(words)
                self.documents.append((words, intent_tag))

        # load extra train
        self._load_additional_training_from_txt('data/train.txt', all_words)

        self.vocabulary = sorted(set(all_words))
        self.logger.info(f"Processed {len(self.intents)} intents with {len(self.vocabulary)} unique words")

    def _preprocess_text(self, text: str) -> List[str]:
        # lowercase and remove nonletter 
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        words = nltk.word_tokenize(text)
        
        words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) >= 2
        ]
        
        return words

    def _create_bow(self, text: str) -> List[int]:
        words = self._preprocess_text(text)
        return [1 if word in words else 0 for word in self.vocabulary]

    def _load_additional_training_from_txt(self, txt_path: str, all_words: List[str]):
        if not os.path.exists(txt_path):
            return

        added = 0
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        continue
                    intent_tag, text = parts[0].strip(), parts[1].strip()
                    if not intent_tag or not text:
                        continue

                    if intent_tag not in self.intents:
                        self.intents.append(intent_tag)
                        self.responses[intent_tag] = [
                            f"I recognized your message as intent '{intent_tag}', but I don't have a detailed response yet."
                        ]
                        self.extensions[intent_tag] = {}
                        self.contexts[intent_tag] = {}
                        self.entity_types[intent_tag] = 'NA'
                        self.entity_patterns[intent_tag] = []

                    words = self._preprocess_text(text)
                    all_words.extend(words)
                    self.documents.append((words, intent_tag))
                    added += 1

            if added:
                self.logger.info(f"Loaded {added} additional training examples from {txt_path}")
        except Exception as e:
            self.logger.error(f"Error loading additional training data from {txt_path}: {e}")

    def _extract_entities(self, text: str, intent_tag: str) -> Dict:
        entities = {}
        
        patterns = [
            r"(?:my name is|i am|i'm|call me)\s+(\w+)",
            r"this is\s+(\w+)",
            r"(\w+)\s+(?:here|speaking)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['HUMAN'] = [match.group(1)]
                break
        
        numbers = re.findall(r'\d+', text)
        if numbers:
            entities['NUMBER'] = numbers
            
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            entities['EMAIL'] = emails
        
        pattern_specs = getattr(self, 'entity_patterns', {}).get(intent_tag, [])
        if pattern_specs:
            lower_text = text.lower()
            for spec in pattern_specs:
                entity_name = spec.get('entity')
                if not entity_name:
                    continue
                values = []
                patterns = spec.get('patterns', [])
                for p in patterns:
                    if not p:
                        continue
                    p_lower = p.lower()
                    if any(ch in p for ch in ['[', '\\', '(', ')', '?', '+', '*', '^', '$', '|']):
                        try:
                            matches = re.findall(p, text, flags=re.IGNORECASE)
                            for m in matches:
                                if isinstance(m, tuple):
                                    for x in m:
                                        if x:
                                            values.append(str(x))
                                else:
                                    values.append(str(m))
                        except re.error:
                            if p_lower in lower_text:
                                values.append(p)
                    else:
                        if p_lower in lower_text:
                            values.append(p)
                if values:
                    existing = entities.get(entity_name, [])
                    for v in values:
                        if v not in existing:
                            existing.append(v)
                    entities[entity_name] = existing
        
        return entities

    def prepare_training_data(self):
        X = []
        y = []
        
        for doc in self.documents:
            words, tag = doc
            base_text = ' '.join(words)
            bow = self._create_bow(base_text)
            X.append(bow)
            y.append(self.intents.index(tag))

            for _ in range(1):
                aug_words = words.copy()
                if len(aug_words) > 3:
                    drop_idx = random.randrange(len(aug_words))
                    del aug_words[drop_idx]
                aug_text = ' '.join(aug_words)
                bow_aug = self._create_bow(aug_text)
                X.append(bow_aug)
                y.append(self.intents.index(tag))
            
        self.X = np.array(X, dtype=np.float32)
        self.Y = np.array(y, dtype=np.int64)
        
        self.logger.info(f"Training data prepared: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def train_model(self, epochs: int = 200, lr: float = 0.001, batch_size: int = 32):
        dataset = TensorDataset(
            torch.tensor(self.X, dtype=torch.float32),
            torch.tensor(self.Y, dtype=torch.long)
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.model = ImprovedChatModel(len(self.vocabulary), len(self.intents))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            scheduler.step(avg_val_loss)
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state_dict = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def get_response(self, message: str, user_id: str = "default") -> Tuple[str, float, Dict, str]:
        if not self.model:
            return "i got no data nub", 0.0, {}, "unknown"
        
        lower_message = message.lower()
        name_patterns = [
            r"(?:my name is|i am|i'm|call me)\s+(\w+)",
            r"this is\s+(\w+)",
            r"(\w+)\s+(?:here|speaking)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                self.user_profile['name'] = match.group(1)
                self._save_user_profile()
                break
        if ("what is my name" in lower_message or
            ("what" in lower_message and "my name" in lower_message) or
            "do you remember my name" in lower_message):
            name = self.user_profile.get('name')
            if name:
                response = f"Your name is {name}, at least that's what you told me."
            else:
                response = "I don't know your name yet."
            intent_tag = "current_name_query"
            confidence = 1.0
            entities: Dict = {}
            turn = ConversationTurn(
                user_input=message,
                bot_response=response,
                intent=intent_tag,
                confidence=confidence,
                entities={},
                timestamp=datetime.now(),
                context=self.current_context.copy()
            )
            self._log_conversation(turn, user_id)
            return response, confidence, {}, intent_tag

        if user_id in self.current_context:
            game_state = self.current_context[user_id].get('active_game')
            if game_state and game_state.get('type') == 'number':
                self._update_context('play_game', message, user_id)
                response = self._handle_number_game_turn(message, user_id)
                intent_tag = 'play_game'
                confidence = 1.0
                entities: Dict = {}
                turn = ConversationTurn(
                    user_input=message,
                    bot_response=response,
                    intent=intent_tag,
                    confidence=confidence,
                    entities=entities,
                    timestamp=datetime.now(),
                    context=self.current_context.copy()
                )
                self._log_conversation(turn, user_id)
                return response, confidence, {}, intent_tag

        stripped = lower_message.strip()

        how_are_you_phrases = {
            "how are you",
            "how are you?",
            "how are you today",
            "how are you today?",
            "how are you doing",
            "how are you doing?",
            "how's it going",
            "hows it going",
            "what's up",
            "whats up",
            "u good",
            "u ok",
            "you good",
            "you ok"
        }
        if stripped in how_are_you_phrases:
            intent_tag = "how_are_you" if "how_are_you" in self.intents else self.intents[0]
            entities = {}
            self._update_context(intent_tag, message, user_id)
            response = self._get_personalized_response(intent_tag, entities, user_id, None)
            confidence = 1.0
            turn = ConversationTurn(
                user_input=message,
                bot_response=response,
                intent=intent_tag,
                confidence=confidence,
                entities=entities,
                timestamp=datetime.now(),
                context=self.current_context.copy()
            )
            self._log_conversation(turn, user_id)
            return response, confidence, {}, intent_tag

        simple_greetings = {"hi", "hello", "hey", "hi there", "hello there", "hey there"}
        if stripped in simple_greetings:
            intent_tag = "greeting" if "greeting" in self.intents else self.intents[0]
            entities: Dict = {}
            self._update_context(intent_tag, message, user_id)
            response = self._get_personalized_response(intent_tag, entities, user_id, None)
            confidence = 1.0
            turn = ConversationTurn(
                user_input=message,
                bot_response=response,
                intent=intent_tag,
                confidence=confidence,
                entities=entities,
                timestamp=datetime.now(),
                context=self.current_context.copy()
            )
            self._log_conversation(turn, user_id)
            return response, confidence, {}, intent_tag

        bow = self._create_bow(message)
        
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(torch.tensor([bow], dtype=torch.float32))
            probabilities = torch.softmax(prediction, dim=1)
            confidence, intent_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            intent_idx = intent_idx.item()
        
        intent_tag = self.intents[intent_idx]
        
        if confidence < self.confidence_threshold:
            response = self._get_fallback_response(message)
            intent_tag = "fallback"
            entities = {}
        else:
            entities = self._extract_entities(message, intent_tag)
            
            self._update_context(intent_tag, message, user_id)
            
            extension_result = None

            if intent_tag == 'play_game':
                try:
                    extension_result = self._start_number_game(user_id)
                except Exception as e:
                    self.logger.error(f"Error starting number game: {e}")
            elif intent_tag == 'productivity_tips':
                try:
                    extension_result = self._generate_productivity_plan(message, user_id, entities)
                except Exception as e:
                    self.logger.error(f"Error generating productivity plan: {e}")
            elif intent_tag == 'study_habits':
                try:
                    extension_result = self._generate_study_plan(message, user_id, entities)
                except Exception as e:
                    self.logger.error(f"Error generating study plan: {e}")

            extension_func = self.extensions[intent_tag].get('function', '')
            if extension_result is None and extension_func and extension_func in self.function_mappings:
                try:
                    if extension_func in ['extensions.math.calculate']:
                        extension_result = self.function_mappings[extension_func](message)
                    elif extension_func in ['extensions.gHumans.updateHuman'] and entities.get('HUMAN'):
                        extension_result = self.function_mappings[extension_func](entities['HUMAN'][0])
                    elif extension_func in ['extensions.reminders.setReminder']:
                        extension_result = self.function_mappings[extension_func](message)
                    elif extension_func in ['extensions.names.setAIName'] and entities.get('AI_NAME'):
                        extension_result = self.function_mappings[extension_func](entities['AI_NAME'][0])
                    elif extension_func in ['extensions.time.getTimeInfo']:
                        extension_result = self.function_mappings[extension_func](message)
                    elif extension_func in ['extensions.facts.getRandomFact', 'extensions.music.playMusic', 'extensions.stories.generateStory']:
                        extension_result = self.function_mappings[extension_func]()
                    else:
                        extension_result = self.function_mappings[extension_func]()
                except Exception as e:
                    self.logger.error(f"Error executing function {extension_func}: {e}")
                    extension_result = None
            
            response = self._get_personalized_response(intent_tag, entities, user_id, extension_result)
        
        # Log 
        turn = ConversationTurn(
            user_input=message,
            bot_response=response,
            intent=intent_tag,
            confidence=confidence,
            entities=entities if confidence >= self.confidence_threshold else {},
            timestamp=datetime.now(),
            context=self.current_context.copy()
        )
        
        self._log_conversation(turn, user_id)
        
        return response, confidence, entities if confidence >= self.confidence_threshold else {}, intent_tag

    def _get_fallback_response(self, message: str) -> str:
        fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I'm still learning. Can you try asking that differently?",
            "That's interesting! I'm not quite sure how to respond to that yet.",
            "Could you be more specific? I want to help but need more clarity.",
            "I'm not familiar with that. Can you explain what you mean?"
        ]

        lower_message = message.lower()
        
        if any(word in lower_message for word in ['help', 'assist', 'support']):
            return "I'd love to help! What specifically do you need assistance with?"

        if any(word in lower_message for word in ['study', 'exam', 'homework', 'assignment', 'revise']):
            return (
                "It sounds like you need study help. Try asking things like "
                "'Give me tips for exam preparation' or 'How can I study more effectively?'."
            )

        if any(word in lower_message for word in ['productivity', 'focus', 'procrastinate', 'time management', 'distraction']):
            return (
                "It sounds like you want productivity help. You can ask things like "
                "'Give me tips to stop procrastinating' or 'How can I be more productive each day?'."
            )

        if any(word in lower_message for word in ['budget', 'saving', 'savings', 'expenses', 'spending', 'finance', 'money']):
            return (
                "It sounds like you're asking about money. Try questions like "
                "'How can I create a monthly budget?' or 'Give me tips for controlling my spending.'."
            )
        
        if '?' in message:
            return "That's a great question! I'm not sure about that yet, but I'm always learning."
        
        return random.choice(fallback_responses)

    def _start_number_game(self, user_id: str) -> str:
        """Start a simple number guessing game and store state in context"""
        import random
        if user_id not in self.current_context:
            self.current_context[user_id] = {
                'conversation_history': [],
                'active_topics': [],
                'user_preferences': {},
                'session_start': datetime.now().isoformat()
            }
        target = random.randint(1, 50)
        self.current_context[user_id]['active_game'] = {
            'type': 'number',
            'target': target,
            'attempts': 0
        }
        return (
            "Let's play a number guessing game! I'm thinking of a number between 1 and 50. "
            "Try to guess it by typing a number. You can type 'quit game' to stop."
        )

    def _handle_number_game_turn(self, message: str, user_id: str) -> str:
        lower_message = message.lower()
        game_state = self.current_context.get(user_id, {}).get('active_game') or {}

        if 'quit game' in lower_message or 'stop game' in lower_message or 'end game' in lower_message:
            self.current_context[user_id].pop('active_game', None)
            return "Okay, we'll stop the game here. If you want to play again, just ask me to start a game."

        numbers = re.findall(r'\d+', message)
        if not numbers:
            return "I'm still thinking of a number between 1 and 50. Try guessing a number, or type 'quit game' to stop."

        try:
            guess = int(numbers[0])
        except ValueError:
            return "I couldn't read that as a number. Please guess with a whole number like 17."

        target = int(game_state.get('target', 0))
        attempts = int(game_state.get('attempts', 0)) + 1
        self.current_context[user_id]['active_game']['attempts'] = attempts

        if guess < target:
            return f"Not quite. My number is higher than {guess}. Try again!"
        if guess > target:
            return f"Close, but my number is lower than {guess}. Give it another try!"

        self.current_context[user_id].pop('active_game', None)
        return f"Nice work! You guessed my number {target} in {attempts} tries."

    def _generate_productivity_plan(self, message: str, user_id: str, entities: Dict) -> str:
        lower_message = message.lower()
        user_name = self.user_profile.get('name', 'you')

        time_hint = None
        numbers = entities.get('NUMBER') or []
        if numbers:
            time_hint = numbers[0]

        if 'tomorrow' in lower_message:
            timeframe = 'tomorrow'
        elif 'week' in lower_message:
            timeframe = 'this week'
        else:
            timeframe = 'today'

        if user_id in self.current_context:
            recent_intents = [h['intent'] for h in self.current_context[user_id]['conversation_history'][-3:]]
        else:
            recent_intents = []

        stress_note = ''
        if any(intent in recent_intents for intent in ['emotion_support', 'stress_management']):
            stress_note = "Start gentle and don't overload yourself. "

        if time_hint:
            focus_block = f"Use {time_hint} focused 25-minute blocks with 5-minute breaks."
        else:
            focus_block = "Use a few focused 25-minute blocks with 5-minute breaks."

        plan_lines = [
            f"Here is a simple productivity plan for {timeframe}, {user_name}:",
            "",
            "1. Define your main goal:",
            "   - Write one clear sentence about what you want to finish.",
            "2. List your top 3 tasks:",
            "   - Choose the three most important tasks that move you toward that goal.",
            "3. Create time blocks:",
            f"   - {focus_block}",
            "4. Remove obvious distractions:",
            "   - Put your phone away, close unrelated tabs, and prepare what you need.",
            "5. Start with the smallest step:",
            "   - Pick the easiest first action (like opening the document or writing a heading).",
            "6. Review at the end:",
            "   - Check what you finished and write down the next small step for tomorrow."
        ]

        if stress_note:
            plan_lines.append("")
            plan_lines.append("Note: " + stress_note.strip())

        return "\n".join(plan_lines)

    def _generate_study_plan(self, message: str, user_id: str, entities: Dict) -> str:
        lower_message = message.lower()
        user_name = self.user_profile.get('name', 'you')

        if 'exam' in lower_message:
            goal = 'prepare for your exam'
        elif 'test' in lower_message:
            goal = 'prepare for your test'
        else:
            goal = 'study more effectively'

        numbers = entities.get('NUMBER') or []
        if numbers:
            sessions_note = f"Plan about {numbers[0]} focused study sessions."\
                " Adjust if that feels like too much or too little."
        else:
            sessions_note = "Plan 2–4 focused study sessions of 25–40 minutes each."

        if user_id in self.current_context:
            recent_intents = [h['intent'] for h in self.current_context[user_id]['conversation_history'][-3:]]
        else:
            recent_intents = []

        memory_note = ''
        if 'memory_issues' in recent_intents or 'emotion_support' in recent_intents:
            memory_note = (
                "Use active recall (testing yourself) and spaced repetition instead of just rereading notes."
            )

        plan_lines = [
            f"Here is a simple plan to {goal}, {user_name}:",
            "",
            "1. Clarify the topics:",
            "   - Write a short list of chapters or concepts you need to cover.",
            "2. Break topics into small chunks:",
            "   - Turn each topic into small questions you should be able to answer.",
            "3. Schedule study blocks:",
            f"   - {sessions_note}",
            "4. Use active study methods:",
            "   - Explain the material in your own words, teach it to an imaginary friend, or write flashcards.",
            "5. Test yourself:",
            "   - Close your notes and try to recall key ideas or solve practice questions.",
            "6. Review and adjust:",
            "   - At the end of the day, note what worked and what still feels confusing."
        ]

        if memory_note:
            plan_lines.append("")
            plan_lines.append("Extra tip: " + memory_note)

        return "\n".join(plan_lines)

    def _update_context(self, intent_tag: str, message: str, user_id: str = "default"):
        if user_id not in self.current_context:
            self.current_context[user_id] = {
                'conversation_history': [],
                'active_topics': [],
                'user_preferences': {},
                'session_start': datetime.now().isoformat()
            }
        
        self.current_context[user_id]['conversation_history'].append({
            'intent': intent_tag,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.current_context[user_id]['conversation_history']) > 10:
            self.current_context[user_id]['conversation_history'] = \
                self.current_context[user_id]['conversation_history'][-10:]
        
        topic_mapping = {
            'food_talk': 'food',
            'music_talk': 'music', 
            'technology_talk': 'technology',
            'emotion_support': 'emotions',
            'creative_request': 'creativity',
            'math_question': 'mathematics'
        }
        
        if intent_tag in topic_mapping:
            topic = topic_mapping[intent_tag]
            if topic not in self.current_context[user_id]['active_topics']:
                self.current_context[user_id]['active_topics'].append(topic)
            
            if len(self.current_context[user_id]['active_topics']) > 3:
                self.current_context[user_id]['active_topics'] = \
                    self.current_context[user_id]['active_topics'][-3:]
        
        entities = self._extract_entities(message, intent_tag)
        if 'HUMAN' in entities:
            self.user_profile['name'] = entities['HUMAN'][0]
            self._save_user_profile()

    def _get_personalized_response(self, intent_tag: str, entities: Dict, user_id: str, extension_result: str = None) -> str:
        if extension_result:
            return extension_result
        
        response = random.choice(self.responses[intent_tag])
        
        if user_id in self.current_context:
            user_context = self.current_context[user_id]
            
            # repeated topic
            if len(user_context['conversation_history']) > 1:
                recent_intents = [h['intent'] for h in user_context['conversation_history'][-3:]]
                if recent_intents.count(intent_tag) > 1:
                    # same topic repeat
                    context_responses = {
                        'joke': ["Here's another one for you!", "I've got more where that came from!", "Another joke coming up!"],
                        'math_question': ["Let me solve another one!", "More math? I love it!", "Another calculation coming up!"],
                        'random_question': ["Here's something else interesting:", "Let me share another fact:", "Something different this time:"]
                    }
                    if intent_tag in context_responses:
                        response = random.choice(context_responses[intent_tag])
        
        # Replace entity 
        for entity_type, values in entities.items():
            if values:
                response = response.replace(f'%%{entity_type}%%', values[0])
        
        if 'name' in self.user_profile:
            response = response.replace('%%USER%%', self.user_profile['name'])
        
        return response

    def _log_conversation(self, turn: ConversationTurn, user_id: str):
        self.conversation_history.append(turn)
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
        
        try:
            log_entry = {
                'user_id': user_id,
                'timestamp': turn.timestamp.isoformat(),
                'user_input': turn.user_input,
                'bot_response': turn.bot_response,
                'intent': turn.intent,
                'confidence': turn.confidence,
                'entities': turn.entities
            }
            
            if os.path.exists(self.conversation_log_path):
                with open(self.conversation_log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(self.conversation_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging conversation: {e}")

    def get_conversation_analytics(self) -> Dict:
        if not self.conversation_history:
            return {}
        
        total_conversations = len(self.conversation_history)
        avg_confidence = np.mean([turn.confidence for turn in self.conversation_history])
        
        intent_counts = {}
        for turn in self.conversation_history:
            intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1
        
        return {
            'total_conversations': total_conversations,
            'average_confidence': avg_confidence,
            'intent_distribution': intent_counts,
            'low_confidence_rate': len([t for t in self.conversation_history if t.confidence < self.confidence_threshold]) / total_conversations
        }

    def save_model(self, model_path: str, config_path: str):
        if self.model:
            torch.save(self.model.state_dict(), model_path)
        
        config = {
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'responses': self.responses,
            'extensions': self.extensions,
            'contexts': self.contexts,
            'entity_types': self.entity_types,
            'entity_patterns': self.entity_patterns,
            'user_profile': self.user_profile,
            'current_context': self.current_context
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}, config saved to {config_path}")

    def load_model(self, model_path: str, config_path: str):
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            self.vocabulary = config['vocabulary']
            self.intents = config['intents']
            self.responses = config['responses']
            self.extensions = config['extensions']
            self.contexts = config['contexts']
            self.entity_types = config['entity_types']
            self.entity_patterns = config.get('entity_patterns', {})
            self.user_profile = config.get('user_profile', {})
            self.current_context = config.get('current_context', {})
            
            self.model = ImprovedChatModel(len(self.vocabulary), len(self.intents))
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

def interactive_chat(assistant: EnhancedChatAssistant):
    print("Enhanced Chatbot loaded! Type 'quit' to exit, 'analytics' for stats.")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'analytics':
            analytics = assistant.get_conversation_analytics()
            print("\nConversation Analytics:")
            for key, value in analytics.items():
                print(f"  {key}: {value}")
            continue
        
        if not user_input:
            continue
        
        response, confidence, entities, intent_tag = assistant.get_response(user_input)
        
        print(f"\nBot: {response}")
        print(f"   Confidence: {confidence:.2f}")
        if entities:
            print(f"   Entities: {entities}")

if __name__ == "__main__":
    force_train = any(arg in ("--train", "train") for arg in sys.argv[1:])

    if not force_train and os.path.exists('enhanced_model.pth') and os.path.exists('enhanced_config.json'):
        assistant = EnhancedChatAssistant('data/intents.json', EXTENSION_FUNCTIONS)
        try:
            assistant.load_model('enhanced_model.pth', 'enhanced_config.json')
            print("Loaded existing model. Type 'quit' to exit, 'analytics' for stats.")
            interactive_chat(assistant)
        except Exception as e:
            print(f"Error loading saved model, falling back to training: {e}")
            force_train = True

    if force_train or not (os.path.exists('enhanced_model.pth') and os.path.exists('enhanced_config.json')):
        assistant = EnhancedChatAssistant('data/intents.json', EXTENSION_FUNCTIONS)
        try:
            assistant._process_intent_data()
            assistant.prepare_training_data()
            
            print("Training enhanced chatbot model... (makan waktu)")
            assistant.train_model(epochs=100, batch_size=128)
            
            assistant.save_model('enhanced_model.pth', 'enhanced_config.json')
            print("Model trained and saved. Starting chat.")
            interactive_chat(assistant)
        except FileNotFoundError:
            print("Error: intents.json file not found.")
        except Exception as e:
            print(f"Error during training: {e}")
