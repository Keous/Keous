# Keous

Keous uses AI to show articles on the same topic with different perspectives, to promote diversity of thought and reduce the effects of ideological echo chambers. We scan through hundreds of articles per week from most major media outlets, show them to our AI, and choose articles that are the mathematically most different in opinion while still being on the same topic. You can find the pairings for the week on our website [keous.ai](https://www.keous.ai). 

You can view our research paper at [keous.ai/paper](https://www.keous.ai/paper). 

There are 5 main commands, accessing keous as a module:

- ```python -m keous scrape_urls```: this scrapes the day's stories into a url file (urls.txt)
- ```python -m keous build_collection```: this takes the url file and scrapes all the links there into a collection, and clears the file.
- ```python -m keous cluster```: create headline embeddings for all the stories
- ```python -m keous predict```: predict targeted sentiment for all the stories
- ```python -m keous pair```: create the knowledge base, create mention matrices, select stories
