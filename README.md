# Python-NLP
import spacy
from collections import Counter
from  transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

story="Once upon a time, there was a young girl named Emma. Emma loved to explore new places and learn about different cultures. One summer, she decided to visit a small village in the mountains of Peru. She arrived in the village early in the morning, just as the sun was rising. Emma was curious about the ancient traditions of the villagers and wanted to understand their way of life. She spent her days talking to the locals, participating in their daily activities, and learning their stories. One day, while Emma was exploring the outskirts of the village, she stumbled upon an old, hidden temple covered in vines. Intrigued, she entered the temple and discovered ancient artifacts and inscriptions on the walls. Emma couldn't decipher the inscriptions, so she returned to the village to ask for help. There, she met an old woman named Rosa, known in the village for her incredible storytelling and knowledge of the village's history. Rosa agreed to help Emma and together they returned to the temple. As they studied the inscriptions, Rosa began to tell Emma a fascinating tale about the village's history. The story took place hundreds of years ago, during a time of great upheaval and change. The temple was once a sacred place where villagers sought guidance and protection from the gods during difficult times. Rosa explained that the story was important because it taught the villagers about resilience and hope. Emma listened intently, asking questions to learn more and taking notes to remember every detail. Rosa also revealed that a significant event was approaching – the annual festival celebrating the village's history and culture. This year, the villagers would reenact the story Rosa had told, and Emma was invited to participate. Excited, Emma spent the next few weeks preparing for the festival with the villagers. She helped make costumes, learned traditional dances, and practiced the reenactment. The festival day arrived, and Emma played a key role in the performance, bringing the story to life for everyone to see. After spending a month in the village, Emma felt like she had made lifelong friends. She returned home with a heart full of stories and a newfound appreciation for different cultures. Back in her hometown, she shared her experiences with her friends and family. Emma gave a presentation at her school about her trip during a special assembly. She wanted to inspire others to explore the world and learn about the beauty of diversity. Emma's passionate storytelling captivated her audience, encouraging many of her peers to dream about their own adventures."
doc = nlp(story)
labels= [ent.label_ for ent in doc.ents]
print(labels)

persons= [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
persons = list(set(persons))
print(f"Who? --> {persons}")

date_time= [ent.text for ent in doc.ents if ent.label_ == 'DATE' or 'TIME']
date_time = list(set(date_time))
places= [ent.text for ent in doc.ents if ent.label_ == 'GPE']
places = list(set(places))
print(f"Where? --> {places}")

for persons in date_time and persons:
   date_time.remove(persons)
for places in date_time and places:
   date_time.remove(places)
print(f"When? --> {date_time}")

words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
word = Counter(words)
list_of_words=[]
for word, freq in word.items():
  if freq > 5:
    list_of_words.append(word)
print(f"What? --> {list_of_words}")

tokens=tokenizer.tokenize(story)
ing=[word for word in tokens if 'ing' in word]
ing = list(set(ing))
print(f"How? --> {ing}")

reasons = [sent.text for sent in doc.sents if "because" in sent.text]
reasons = list(set(reasons))
print(f"Why? --> {reasons}")
