# Keous
 
Keous uses AI to mitigate media bias. This is the code for the AI which runs Keous. This code is 100% opensource, and it used to power our website [keous.ai] (https://www.keous.ai). 

You can view our research paper at [keous.ai/paper] (https://www.keous.ai/paper). 

There are 4 main files in the root directory:

- scrape.py: this scrapes the day's stories from news sites and adds it to a collection
- cluster.py: create headline embeddings for all the stories
- predict.py: predict targeted sentiment for all the stories
- pair.py: create the knowledge base, create mention matrices, select stories
