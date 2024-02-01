# Lire les N premières lignes du fichier bz2, extraire seulement le texte du post + catégorie neutre et en faire un fichier csv
# Page consultée : https://www.pythontutorial.net/python-basics/python-write-csv-file/
# Le script de cette page a été repris et adapté à nos besoins

from itertools import islice
import json
import csv 
import ast

header = ["author", "subreddit", "post", "categorie"]
n = 500000
with open("C:/Users/p1115145/Documents/SCI6203/RC_2017-02/RC_2017-02", 'r') as sample:    
    # Définir le nombre de lignes par fichier (500k)
    head = list(islice(sample, n))
    lines = [json.loads(line) for line in head]
    test = lines[0]
    print(test)
    
    # now we will open a file for writing
    path_w = 'C:/Users/p1115145/Documents/SCI6203/'
    file_name = 'neutres_500k.csv'
    data_file = open(path_w+file_name, 'w', encoding="utf-8", newline='')
        
    # create the csv writer object
    csv_writer = csv.writer(data_file, delimiter=';')
    csv_writer.writerow(header)
    
    for line in lines:
        csv_writer.writerow([line['author'], line['subreddit'], line['body'], 'neutre'])     
    
data_file.close()
