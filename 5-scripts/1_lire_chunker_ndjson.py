# Morceler le fichier reddit.json en x fichiers .csv d'une longueur de n lignes chacun
# Page consultée : https://coderwall.com/p/5vi8ca/use-python-to-read-file-by-n-lines-each-time
# Le script de cette page a été repris et adapté à nos besoins

from itertools import islice
import json
import csv 


def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]


#Définir le nombre de fichiers voulus 

#Nous avons découpé le fichier reddit.ndjson (fourni par les auteurs de Riberio et al. 
#en 60 fichiers .csv de 500 000 lignes chacun (le fichier total faisant plus de 28.8M de lignes)
#Pour la remise, nous n'avons cependant joint à notre archive que 
#les 2M premières lignes (soit les 4 premiers fichiers produits par le script)

nb_fichiers = 60

#Changer le path au besoin 
with open("C:/Users/p1115145/Documents/SCI6203/corpus/ndjson/reddit.ndjson", 'r') as sample:
    for f in range(nb_fichiers) : 
     
# Définir le nombre de lignes par fichier (500k)
        fichier = next_n_lines(sample, 500000) 
        
        # now we will open a file for writing (Changer le path au besoin )
        path_w = 'C:/Users/p1115145/Documents/SCI6203/corpus/ndjson/'
        file_name = 'reddit_500k_'+str(f+1)+'.csv'
        data_file = open(path_w+file_name, 'w', encoding="utf-8", newline='')
        
        # create the csv writer object
        csv_writer = csv.writer(data_file)
        
        # Counter variable used for writing
        # headers to the CSV file
        count = 0
        
        for line in fichier:
            line = json.loads(line)
            
            # Writing headers of CSV file
            if count == 0:
                header = line
                csv_writer.writerow(header)
                count += 1
 
            # Writing data of CSV file
            csv_writer.writerow(line.values())
        
data_file.close()

