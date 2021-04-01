import json
import os
import pandas as pd

def loadPathJSONfiles(path):
    files_in_directory = os.listdir(path)
    filtered_files = [
        file for file in files_in_directory if file.endswith(".json")]

    return filtered_files

def ResultsCSV(paths,NewDirectory):
    df = pd.DataFrame({})
    for file in paths:
        json_file = open('{}/{}'.format('./Results/JSON/', file), "r")
        dictionary = json.load(json_file)
        json_file.close()

        df2 = pd.DataFrame(dictionary, index=[0])
        df = pd.concat([df, df2])

    
    if not os.path.exists('./Results/{}'.format(NewDirectory)):
        os.makedirs('./Results/{}'.format(NewDirectory))

    df.to_csv('./Results/{}/dataset.csv'.format(NewDirectory),index=False)


pathsJsons=loadPathJSONfiles('./Results/JSON/')

#bisogna decidere il nome da dare alla nuova cartella
ResultsCSV(pathsJsons,'30')