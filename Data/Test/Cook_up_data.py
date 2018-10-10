import spacy
from pprint import pprint
import json

MODEL = spacy.load('de_core_news_sm')

emails = ['Conor Anthony McGregor ist ein irischer professioneller Kampfsportler und Boxer. Er ist der ehemalige Ultimate Fighting Championship im Federgewicht und Leichtgewicht. Er hat auch als Weltergewicht in Mixed Martial Arts und Halbmittelgewicht im Boxen konkurriert', \
          'Hinton war einer der ersten Forscher, der die Verwendung eines generalisierten Backpropagation-Algorithmus zum Trainieren von mehrschichtigen neuronalen Netzen demonstrierte. Er ist eine führende Figur in der Deep Learning Community und wird von manchen als "Godfather of Deep Learning" bezeichnet.',\
          'Hinton moved from the U.S. to Canada in part due to disillusionment with Reagan-era politics and disapproval of military funding of artificial intelligence.', \
          'Hinton is the great-great-grandson both of logician George Boole whose work eventually became one of the foundations of modern computer science',\
          'Hinton lehrte 2012 einen kostenlosen Online-Kurs zu Neuronalen Netzen auf der Bildungsplattform Coursera']
res = []
for i, email in enumerate(emails):
    tmp = {}
    tmp['name'] = 'doc' + str(i)
    tmp['text'] = email
    tags = []
    doc = MODEL(email)
    for sent in doc.sents:
        for word in sent:
            if word.ent_type_ in {'PER'}:
                tags.append({'start_idx': word.idx , 'end_idx' : word.idx + len(word.text), \
                             'entity' : word.ent_type_ , 'phrase' : word.text})
    tmp['tags'] = tags
    res.append(tmp)

pprint (res)


json.dump(res, open('./data.json', 'w'))