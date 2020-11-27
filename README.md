# Keous

Keous uses AI to show articles on the same topic with different perspectives, to promote diversity of thought and reduce the effects of ideological echo chambers. We scan through hundreds of articles per week from most major media outlets, show them to our AI, and choose articles that are the mathematically most different in opinion while still being on the same topic. You can find the pairings for the week on our website [keous.ai](https://www.keous.ai). 

You can view our research paper at [keous.ai/paper](https://www.keous.ai/paper). 

There are 4 main files in the root directory:

- scrape.py: this scrapes the day's stories from news sites and adds it to a collection
- cluster.py: create headline embeddings for all the stories
- predict.py: predict targeted sentiment for all the stories
- pair.py: create the knowledge base, create mention matrices, select stories
