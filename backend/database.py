# imports
import numpy as np
import os
import pickle
import sqlite3
import re
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import savemat
from collections import Counter


# constants
#STORAGE_FOLDER = './backend/storage/'
STORAGE_FOLDER = '/app/storage/'  # production
CANTUS_DB_FILE = STORAGE_FOLDER + 'chant_info.db'


def find_chant_concordances(cantusid, excludesource):
    """Retrieve all concordances of a chant in sources
    """
    # connect to database
    conection = sqlite3.connect(CANTUS_DB_FILE)
    cursor = conection.cursor()

    # CD (Cantus Database) database
    exclude = excludesource[0] if excludesource[1] == 'CD' else ''
    cursor.execute('SELECT source, folio, sequence FROM chant_cantusdb WHERE cantusid=? AND source!=?',
                   (cantusid, exclude, ))
    result = cursor.fetchall()
    dataset = ['CD'] * len(result)

    conection.close()
    return result, dataset


def find_chant_neighbours(chantuple, dataset, numvecinos=1):
    """Retrieve neighbours of a chant
    """
    source = chantuple[0]
    folio = chantuple[1]
    sequence = int(chantuple[2])

    # connect to database
    conection = sqlite3.connect(CANTUS_DB_FILE)
    cursor = conection.cursor()

    vecinos = []

    if dataset == 'CD':
        # get list of folios
        cursor.execute('SELECT folios FROM source_cantusdb WHERE id=?',
                       (source,))
        result = cursor.fetchall()
        folios = result[0][0].split(';')[:-1]
        folioidx = folios.index(folio)

        # look forward
        currentfolio = folio
        currentfolioidx = folioidx
        currentseq = sequence+1
        for i in range(0,numvecinos):
            cursor.execute('SELECT cantusid FROM chant_cantusdb WHERE source=? AND folio=? AND sequence=?',
                           (source, currentfolio, str(currentseq)))
            result = cursor.fetchall()
            if len(result)==0 and currentfolioidx<len(folios):
                currentfolioidx = currentfolioidx+1
                currentfolio = folios[currentfolioidx]
                currentseq = 1
                cursor.execute('SELECT cantusid FROM chant_cantusdb WHERE source=? AND folio=? AND sequence=?',
                               (source, currentfolio, str(currentseq)))
                result = cursor.fetchall()
            currentseq = currentseq + 1
            if len(result)>0 and len(result[0][0]) > 0:
                vecinos.append(result[0][0])

        # look backward
        currentfolio = folio
        currentfolioidx = folioidx
        currentseq = sequence-1
        for i in range(0,numvecinos):
            cursor.execute('SELECT cantusid FROM chant_cantusdb WHERE source=? AND folio=? AND sequence=?',
                           (source, currentfolio, str(currentseq)))
            result = cursor.fetchall()
            if len(result)==0 and currentfolioidx>0:
                currentfolioidx = currentfolioidx-1
                currentfolio = folios[currentfolioidx]
                cursor.execute('SELECT sequence FROM chant_cantusdb WHERE source=? AND folio=?',
                               (source, currentfolio))
                result = cursor.fetchall()
                aux = np.array(list(int(item[0]) for item in result))
                currentseq = np.max(aux)
                cursor.execute('SELECT cantusid FROM chant_cantusdb WHERE source=? AND folio=? AND sequence=?',
                               (source, currentfolio, str(currentseq)))
                result = cursor.fetchall()
            currentseq = currentseq - 1
            if len(result)>0 and len(result[0][0]) > 0:
                vecinos.append(result[0][0])

    conection.close()
    return vecinos


def find_chant_suggested(cantusidlist, excludesource = ('', '')):
    """Retrieve suggested chants according to position in sources
    """
    suggested = []

    for chant in cantusidlist:
        results, dataset = find_chant_concordances(chant, excludesource)
        for result, set in zip(results, dataset):
            vecinos = find_chant_neighbours(result, set)
            suggested.extend(vecinos)
    
    contador = Counter(suggested)
    ordered = sorted(contador.items(), key=lambda x: x[1], reverse=True)
    return ordered


def get_genre_desc(name):
    # connect to database
    conection = sqlite3.connect(CANTUS_DB_FILE)
    cursor = conection.cursor()

    cursor.execute('SELECT desc FROM genre WHERE name=?',
                   (name, ))
    result = cursor.fetchall()
    conection.close()
    if len(result) == 0:
        return ''
    else:
        return result[0][0]


cantusid = ['001002', '001019']
excludesource = ('123592', 'CD')
ordered = find_chant_suggested(cantusid, excludesource)
# print('Done')
