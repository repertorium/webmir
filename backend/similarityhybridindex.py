# imports
import numpy as np
import os
import pickle
import sqlite3
import re
from tqdm import tqdm
import gc
import numpy as np
from Bio.Align import PairwiseAligner
from multiprocessing import Pool, cpu_count
from utils import nearest_index
import random
import string
import ast
import roman
import copy
from database import find_chant_suggested, get_genre_desc
from sporkapi import SporkWorkspaceSession


# constants
STORAGE_FOLDER = '/app/storage/'
CANTUS_DB_FILE = STORAGE_FOLDER + 'chant_info.db'
DICTIONARY_MELODY = STORAGE_FOLDER + 'dictionary_melody.pkl'
DICTIONARY_LYRICS = STORAGE_FOLDER + 'dictionary_lyrics.pkl'

RECOMMENDATIONS_FLAG = False

MUSIC_MIN_SCORE = 0.7
LYRICS_MIN_SCORE = 0.3

# constants
SPORK_URL = 'https://spork-repertorium-bd227a1477a1.herokuapp.com'
MANUS_URL = SPORK_URL + '/manuscript'
CHANT_URL = SPORK_URL + '/api/chant-record'
CANTU_API = SPORK_URL + '/api/cantus-index/lookup?cantusId='
INDEX_API = SPORK_URL + '/api/image'


# global variables
sequence_foralign = ''
max_alignments = 0

# local aligner
aligner = PairwiseAligner()
aligner.mode = 'local'
aligner.match_score = 1
aligner.mismatch_score = -0.5
aligner.open_gap_score = -1
aligner.extend_gap_score = -0.5



def read_omr(datapath):
    """Read result from OMR.
    """
    omrdata = []
    trans_folios = []

    badsubstrings = ["V.", ".", "*", "\n"]

    # prediction (coordinates) folder
    pred_path = os.path.join(datapath, 'predictions')

    # read data for each folio
    for file in os.listdir(pred_path):
        # folio/file name
        base_name = os.path.splitext(file)[0]

        # read coordinates table
        with open(os.path.join(pred_path, file), "r") as fileh:
            lines = fileh.readlines()
        
        # parse table and sort according to y_center
        table = [list(map(float, line.strip().split())) for line in lines]
        table = [entry for entry in table if int(entry[0]) == 12]
        table.sort(key=lambda x: x[2])

        # read transcription for each square
        trans_path = os.path.join(datapath, 'trans', base_name)

        if not os.path.exists(trans_path):
            omrdata.append({'name': base_name, 'table': table, 'trans': []})
            trans_folios.append('')
            continue

        trans_folio = [None] * len(os.listdir(trans_path))
        for transfile in os.listdir(trans_path):
            with open(os.path.join(trans_path, transfile), "r") as fileh:
                texto = fileh.read()

                # clean transcription
                for badsub in badsubstrings:
                    texto = texto.replace(badsub, "")
                texto = ' '.join(texto.split())

                num = int(re.search(r'\d+', transfile).group())
                trans_folio[num-1] = texto

        omrdata.append({'name': base_name, 'table': table, 'trans': trans_folio})
        trans_folios.append(' '.join(trans_folio))

    return omrdata, trans_folios


# def create_omr_sequence(omrdata, trans_folios):
#     """Create string sequence from OMR data.
#     """
#     seq_folio = []
#     seq_y = []
#     seq_box = []

#     nboxes = 0

#     for folio, omrfolio in enumerate(omrdata):
#         for square, omrsquare  in enumerate(omrfolio['table']):
#             aux = [omrsquare[2]] * (len(omrfolio['trans'][square]) + 1)
#             seq_y.extend(aux)
#             aux = [folio] * (len(omrfolio['trans'][square]) + 1)
#             seq_folio.extend(aux)
#             aux = [nboxes + square] * (len(omrfolio['trans'][square]) + 1)
#             seq_box.extend(aux)
#         nboxes = nboxes + square + 1

#     seq_folio = seq_folio[:-1]
#     seq_y = seq_y[:-1]
#     seq_box = seq_box[:-1]
#     sequence = ' '.join(trans_folios)

#     return sequence, seq_folio, seq_y, seq_box


def extract_lyrics_and_positions2(text):

    # get position of each syllable
    syllablepos = [0,0,0] # [start, "("-pos, ")"-pos]
    syllablelist = []
    for i, char in enumerate(text):
        if char == '(':
            syllablepos[1] = i
        elif char == ')':
            syllablepos[2] = i
            syllablelist.append(syllablepos)
            syllablepos = [i+1,0,0]

    # determine whether syllable has pitch
    syllableisempty = [False] * len(syllablelist)
    for i, syllable in enumerate(syllablelist):
        decision = True
        for j in range(syllable[1]+1, syllable[2]):
            if not text[j].isspace():
                decision = False
                break
        syllableisempty[i] = decision

    # filter syllables connected to other pitched syllables
    for i in range(len(syllablelist)):
        if syllableisempty[i]:
            if i > 0 and not syllableisempty[i-1] and not text[syllablelist[i][0]].isspace():
                syllableisempty[i] = False
            elif i < len(syllablelist)-1 and not syllableisempty[i+1] and not text[syllablelist[i+1][0]].isspace():
                syllableisempty[i] = False

    # create vector indicating whether each char is in empty syllable
    in_empty_syllable = [False] * len(text)
    for i, syllable in enumerate(syllablelist):
        if syllableisempty[i]:
            for j in range(syllable[0], syllable[2]+1):
                in_empty_syllable[j] = True

    in_note = False
    buffer = []
    buffer2 = []

    # Extract lyrics
    for i, char in enumerate(text):
        if char == '(':
            in_note = True
        elif char == ')':
            in_note = False
        elif not in_note:
            if char.isalpha() or char.isspace():
                if not in_empty_syllable[i]:
                    buffer.append((char, i))
                else:
                    buffer2.append((char, i))

    # Clean lyrics in buffer 1
    cleaned = []
    cleaned_pos = []
    last_was_space = False

    for idx, (char, pos) in enumerate(buffer):
        # Evitar espacios duplicados
        if char.isspace():
            if last_was_space or not cleaned:
                continue
            last_was_space = True
        else:
            last_was_space = False

        cleaned.append(char)
        cleaned_pos.append(pos)

    # Reconstruir texto limpio
    text_clean = ''.join(cleaned)

    # Filtrar secuencias prohibidas
    forbidden_patterns = ["V.", ".", "*", "\n"]
    for pattern in forbidden_patterns:
        while pattern in text_clean:
            start = text_clean.find(pattern)
            end = start + len(pattern)
            del cleaned[start:end]
            del cleaned_pos[start:end]
            text_clean = ''.join(cleaned)

    # Clean lyrics in buffer 2 (empty syllables)
    cleaned2 = []
    cleaned_pos2 = []

    for idx, (char, pos) in enumerate(buffer2):
        # Evitar espacios duplicados
        if char.isspace():
            if last_was_space or not cleaned2:
                continue
            last_was_space = True
        else:
            last_was_space = False

        cleaned2.append(char)
        cleaned_pos2.append(pos)

    # Reconstruir texto limpio
    text_clean2 = ''.join(cleaned2)

    # Filtrar secuencias prohibidas
    for pattern in forbidden_patterns:
        while pattern in text_clean2:
            start = text_clean2.find(pattern)
            end = start + len(pattern)
            del cleaned2[start:end]
            del cleaned_pos2[start:end]
            text_clean2 = ''.join(cleaned2)

    return ''.join(cleaned), cleaned_pos, ''.join(cleaned2), cleaned_pos2



def extract_lyrics_and_positions(text):

    in_note = False
    buffer = []

    # Extract lyrics
    for i, char in enumerate(text):
        if char == '(':
            in_note = True
        elif char == ')':
            in_note = False
        elif not in_note:
            if char.isalpha() or char.isspace():
                buffer.append((char, i))

    cleaned = []
    cleaned_pos = []
    last_was_space = False

    for idx, (char, pos) in enumerate(buffer):
        # Evitar espacios duplicados
        if char.isspace():
            if last_was_space or not cleaned:
                continue
            last_was_space = True
        else:
            last_was_space = False

        cleaned.append(char)
        cleaned_pos.append(pos)

    # Reconstruir texto limpio
    text_clean = ''.join(cleaned)

    # Filtrar secuencias prohibidas
    forbidden_patterns = ["V.", ".", "*", "\n"]
    for pattern in forbidden_patterns:
        while pattern in text_clean:
            start = text_clean.find(pattern)
            end = start + len(pattern)
            del cleaned[start:end]
            del cleaned_pos[start:end]
            text_clean = ''.join(cleaned)

    return ''.join(cleaned), cleaned_pos


def create_omr_sequence(omrdata):
    """Create string sequence from OMR data.
    """
    seq_folio = []
    seq_group = []
    seq_box = []
    seq_char = []
    seq_time = []
    sequence = ''

    nboxes = 0

    for folio, omrfolio in enumerate(omrdata):
        if len(omrfolio) == 0:
            continue
        for square, omrsquare in enumerate(omrfolio['annotationGroups']):
            notation = omrsquare['notation']
            sequence = sequence + notation + ' '
            if "notationtime" in omrsquare:
                notationtime = omrsquare['notationtime']
                seq_time.append(notationtime)
                seq_time.append([notationtime[-1]])
            else:
                seq_time.append([None] * (len(notation) + 1))
            seq_folio.extend([folio] * (len(notation) + 1))
            seq_group.extend([square] * (len(notation) + 1))
            seq_box.extend([nboxes] * (len(notation) + 1))
            nboxes = nboxes + 1
    
    sequenceclean, pos, _, _ = extract_lyrics_and_positions2(sequence)
    seq_folio = [seq_folio[i] for i in pos]
    seq_group = [seq_group[i] for i in pos]
    seq_box = [seq_box[i] for i in pos]
    if len(seq_time) > 0:
        seq_time = np.concatenate(seq_time)
        seq_time = np.array([seq_time[i] for i in pos])
    seq_char = pos

    return sequence, sequenceclean, seq_folio, seq_group, seq_box, seq_char, seq_time


def clean_gabc(text):
    """Clean gabc melody for chant detection.
    """

    text = text.lower()

    # delete clef
    text = re.sub(r"(c\d|f\d|cb\d|fb\d)", "", text)

    # delete v
    text = text.replace('v', '')

    # preserve only pitches
    text = re.sub(r"[^a-m\s]", " ", text)

    # single space
    text = ' '.join(text.split())

    return text



def read_omr_music(datapath):
    """Read result from OMR (music version).
    """
    omrdata = []
    trans_folios = []

    badsubstrings = ["V.", ".", "*", "\n"]

    # prediction (coordinates) folder
    pred_path = os.path.join(datapath, 'predictions')

    # read data for each folio
    for file in os.listdir(pred_path):
        # folio/file name
        base_name = os.path.splitext(file)[0]

        # read coordinates table
        with open(os.path.join(pred_path, file), "r") as fileh:
            lines = fileh.readlines()
        
        # parse table and sort according to y_center
        table = [list(map(float, line.strip().split())) for line in lines]
        table = [entry for entry in table if int(entry[0]) == 12]
        table.sort(key=lambda x: x[2])

        # read transcription for each square
        trans_path = os.path.join(datapath, 'trans_music', base_name)

        if not os.path.exists(trans_path):
            omrdata.append({'name': base_name, 'table': table, 'trans': []})
            trans_folios.append('')
            continue

        trans_folio = [None] * len(os.listdir(trans_path))
        for transfile in os.listdir(trans_path):
            with open(os.path.join(trans_path, transfile), "r") as fileh:
                texto = fileh.read()

                # clean transcription
                texto = clean_gabc(texto)

                num = int(re.search(r'\d+', transfile).group())
                trans_folio[num-1] = texto

        omrdata.append({'name': base_name, 'table': table, 'trans': trans_folio})
        trans_folios.append(' '.join(trans_folio))

    return omrdata, trans_folios



def gabc_diff_encoding(text):
    """Differencial encoding for gabc.
    """

    letters = "abcdefghijklmnopqrstuvwxyz"
    result = []
    prev_char = None
    for char in text:
        if char == ' ':
            result.append(' ')
        else:
            if prev_char is None: # first letter
                result.append('')
            else:
                diff = ord(char) - ord(prev_char)
                if diff >= 0:
                    encoded_char = letters[diff % 26]  # Positivos en minúsculas
                else:
                    encoded_char = letters[(-diff) % 26].upper()  # Negativos en mayúsculas
                result.append(encoded_char)
            prev_char = char

    if len(result) > 1 and result[1] == ' ':
        result = result[2:]
    return ''.join(result)



def clean_volpiano(text):
    """Clean volpiano melody for chant detection and use differencial encoding.
    """

    text = text.lower()
    
    # erase i, I, y, Y, z and Z
    text = "".join(char for char in text if char not in "iIyYzZ")
    
    # replace 9 and ) by chr(ord('a')-1) (character before a)
    text = text.replace('9', '`').replace(')', '`')
    
    text = re.sub(r'-+', ' ', text)
    text = re.sub(r'[^a-z `]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() # spaces to one space

    # diff coding
    valid_chars = "`abcdefghjklmnopqrstuvwxyz"
    char_to_index = {char: idx for idx, char in enumerate(valid_chars)}

    letters = "abcdefghijklmnopqrstuvwxyz"
    result = []
    prev_char = None
    for char in text:
        if char == ' ':
            result.append(' ')
        else:
            if prev_char is None: # first letter
                result.append('')
            else:
                # diff = ord(char) - ord(prev_char)
                diff = char_to_index[char] - char_to_index[prev_char]
                if diff >= 0:
                    encoded_char = letters[diff % 26]  # Positivos en minúsculas
                else:
                    encoded_char = letters[(-diff) % 26].upper()  # Negativos en mayúsculas
                result.append(encoded_char)
            prev_char = char

    if len(result) > 1 and result[1] == ' ':
        result = result[2:]
    return ''.join(result)



def get_melodies_cantusdatabase():
    """Retrieve all melodies in local DB
    """
    # load locally stored dictionary
    if os.path.exists(DICTIONARY_MELODY):
        with open(DICTIONARY_MELODY, 'rb') as archivo:
            id, cantusid, volpiano = pickle.load(archivo)
        return id, cantusid, volpiano

    # connect to database
    conection = sqlite3.connect(CANTUS_DB_FILE)
    cursor = conection.cursor()

    # get full text
    cursor.execute('SELECT id, cantusid, volpiano FROM chant_cantusdb WHERE volpiano != ""')
    result = cursor.fetchall()
    conection.close()

    # convert to list
    id = list(item[0] for item in result)
    cantusid = list(item[1] for item in result)
    volpiano = list(item[2] for item in result)
    lenvolpiano = [0] * len(volpiano)

    # clean volpiano melodies
    for i, volp in enumerate(volpiano):
        volpiano[i] = clean_volpiano(volp)
        lenvolpiano[i] = len(volpiano[i].replace(' ', ''))

    # filter very short melodies
    MINLEN = 4
    id = list(item for item, lenv in zip(id, lenvolpiano) if lenv>=MINLEN)
    cantusid = list(item for item, lenv in zip(cantusid, lenvolpiano) if lenv>=MINLEN)
    volpiano = list(item for item, lenv in zip(volpiano, lenvolpiano) if lenv>=MINLEN)

    # store dictionary into file
    with open(DICTIONARY_MELODY, 'wb') as archivo:
        pickle.dump((id, cantusid, volpiano), archivo)

    return id, cantusid, volpiano



def get_chant_fulltext(clean=True):
    """Retrieve full text for all chants in local DB
    """
    # connect to database
    conection = sqlite3.connect(CANTUS_DB_FILE)
    cursor = conection.cursor()

    # get full text
    cursor.execute('SELECT text, id FROM chant')
    result = cursor.fetchall()
    conection.close()

    pattern = re.compile(r"///|\|\.\.\.|\(\.\.\.\)|\*|###|\.\.\.")
    pattern2 = re.compile(r"\(\.\.\.\)")
    incompleto = False
    dictionary = []
    cantusid = []

    # clean text
    for entry in result:
        canto = entry[0]
        id = entry[1]

        if not clean:
            dictionary.append(canto)
        else:
            if pattern2.search(canto):
                incompleto = True
            else:
                incompleto = False
            texto = re.sub(pattern, "", canto)
            texto = ' '.join(texto.split())
            if incompleto and len(texto.split()) == 1:
                continue
            dictionary.append(texto)
            cantusid.append(id)

    # remove elements with less than NMINWORDS words
    NMINWORDS = 3
    if clean:
        dictionary_clean = []
        cantusid_clean = []
        for frase, id_ in zip(dictionary, cantusid):
            if len(frase.split()) >= NMINWORDS:
                dictionary_clean.append(frase)
                cantusid_clean.append(id_)
        dictionary = dictionary_clean
        cantusid = cantusid_clean

    # remove cantus ids ending in :cs
    dictionary_clean = []
    cantusid_clean = []
    for frase, id_ in zip(dictionary, cantusid):
        if not id_.endswith(':cs'):
            dictionary_clean.append(frase)
            cantusid_clean.append(id_)
    dictionary = dictionary_clean
    cantusid = cantusid_clean

    # remove cantus with ids <name>:<number> if another id with just <name> exists
    nameset = set()
    for id_ in cantusid:
        if ':' not in id_:
            nameset.add(id_)
    dictionary_clean = []
    cantusid_clean = []
    for frase, id_ in zip(dictionary, cantusid):
        if ':' in id_:
            name = id_.split(':')[0]
            number = id_.split(':')[1]
            if name in nameset and number.isdigit():
                continue
        dictionary_clean.append(frase)
        cantusid_clean.append(id_)
    dictionary = dictionary_clean
    cantusid = cantusid_clean

    return dictionary, cantusid



def align_chant(canto):
    """Compute alignment between chant and sequence (SLOW)
    """
    score = []
    start = []
    end = []

    rangos = []

    if len(canto) == 0:
        return (score, start, end)
    sequence = sequence_foralign
    
    for i in range(0, max_alignments):
        alignments = aligner.align(sequence, canto)
        try:
            if len(alignments) == 0:
                return (score, start, end)
        except:
            return (score, start, end)
        alignment = alignments[0]
        
        # extract score and range
        scorei = alignment.score / len(canto)
        starti = alignment.aligned[0][0][0]
        endi = alignment.aligned[0][-1][-1]

        # check possible overlap with other alignments
        enrango = False
        for rango in rangos:
            if starti in rango or endi-1 in rango:
                enrango = True
                break;
        if not enrango:
            rangos.append(range(starti, endi))
            score.append(scorei)
            start.append(starti)
            end.append(endi)

        sequence = sequence[:starti] + '-'*(endi - starti) + sequence[endi:]

    return (score, start, end) 



def align_chant_windows(canto):
    """Compute alignment between chant and sequence (windowing version)
    """
    score = []
    start = []
    end = []

    # max_alignments = 1

    if len(canto) == 0:
        return (score, start, end)

    rangos = []

    # window overlap and hop
    overlap = len(canto)
    winlen = min(overlap + 500 + overlap, len(sequence_foralign))
    
    # process sequence in windows
    inicio = 0
    final = 0
    while final < len(sequence_foralign):
        final = inicio + winlen
        ventana = sequence_foralign[inicio:final]
        alignments = aligner.align(ventana, canto)

        for i, alignment in enumerate(alignments):
            if i >= max_alignments:
                break;
            # determine alignment score and range
            scorei = alignment.score / len(canto)
            starti = alignment.aligned[0][0][0] + inicio
            endi = alignment.aligned[0][-1][-1] + inicio

            # check possible overlap with other alignments
            enrango = False
            for i, rango in enumerate(rangos):
                if starti in rango or endi-1 in rango:
                    if scorei > score[i]:
                        # update score and range
                        rangos[i] = range(starti, endi)
                        score[i] = scorei
                        start[i] = starti
                        end[i] = endi
                    enrango = True
                    break;
            if not enrango:
                rangos.append(range(starti, endi))
                score.append(scorei)
                start.append(starti)
                end.append(endi)

        inicio += (winlen - overlap)

    return (score, start, end) 



# def align_chant_windows2(sequence_foralign, canto):
#     """Compute alignment between chant and sequence (windowing version)
#     """
#     score = []
#     start = []
#     end = []

#     max_alignments = 1

#     if len(canto) == 0:
#         return (score, start, end)

#     rangos = []

#     # window overlap and hop
#     overlap = len(canto)
#     winlen = min(overlap + 500 + overlap, len(sequence_foralign))
    
#     # process sequence in windows
#     inicio = 0
#     final = 0
#     while final < len(sequence_foralign):
#         final = inicio + winlen
#         ventana = sequence_foralign[inicio:final]
#         alignments = aligner.align(ventana, canto)

#         for i, alignment in enumerate(alignments):
#             if i >= max_alignments:
#                 break;
#             # determine alignment score and range
#             scorei = alignment.score / len(canto)
#             starti = alignment.aligned[0][0][0] + inicio
#             endi = alignment.aligned[0][-1][-1] + inicio

#             # check possible overlap with other alignments
#             enrango = False
#             for i, rango in enumerate(rangos):
#                 if starti in rango or endi-1 in rango:
#                     if scorei > score[i]:
#                         # update score and range
#                         rangos[i] = range(starti, endi)
#                         score[i] = scorei
#                         start[i] = starti
#                         end[i] = endi
#                     enrango = True
#                     break;
#             if not enrango:
#                 rangos.append(range(starti, endi))
#                 score.append(scorei)
#                 start.append(starti)
#                 end.append(endi)

#         inicio += (winlen - overlap)

#     return (score, start, end) 




def dpcore_chant(M):
    """
    Parameters:
    M (numpy.ndarray): Input score matrix .

    Returns:
    q (numpy.ndarray): Optimal path.
    D (numpy.ndarray): Dynamic programming matrix.
    phi (numpy.ndarray): Backtracking matrix.
    """

    # Transition cost
    C = 1.0

    # Convert to local cost
    M = 1 - M

    # Initialize variables
    rows, cols = M.shape
    D = np.zeros_like(M, dtype=float)
    phi = np.zeros_like(M, dtype=int)

    # Cell classes
    VACIO = 0
    BEGIN = 2
    FINAL = 3
    MIDDL = 1

    diffmatrix = np.diff(np.hstack([M, np.zeros((rows, 1))]), axis=1)
    MM = np.full_like(M, MIDDL, dtype=int)
    MM[diffmatrix > 0] = FINAL
    temp_matrix = np.hstack([np.zeros((rows, 1)), diffmatrix[:, :-1]])
    MM[temp_matrix < 0] = BEGIN
    MM[M == 1] = VACIO

    del diffmatrix

    # DP algorithm
    for t in tqdm(range(1, cols)):
        idxt = MM[:, t] != BEGIN
        D[idxt, t] = M[idxt, t] * C + D[idxt, t - 1]
        idxtt = np.where((MM[:, t - 1] == FINAL) | (MM[:, t - 1] == VACIO))[0]

        for j in np.where(~idxt)[0]:
            d = np.finfo(float).max
            tb = 0

            for jj in np.hstack([[j], idxtt]):
                d2 = M[j, t] * C + D[jj, t - 1]

                if d2 < d:
                    d = d2
                    tb = j - jj

            # Store result for this cell
            D[j, t] = d
            phi[j, t] = tb

    # Backtrack
    q = np.zeros(cols, dtype=int)
    idx = np.argmin(D[:, -1])
    q[-1] = idx
    for t in range(cols - 1, 0, -1):
        q[t - 1] = idx - phi[idx, t]
        idx = q[t - 1]

    return q, D, phi


def init_pool(seq, maxal):
    global sequence_foralign
    global max_alignments
    sequence_foralign = seq
    max_alignments = maxal


def create_notation_comments(sentence, notation, trie, session):
    notationbox = '% ' + sentence + '\n\n'

    # check sentence has more than 2 words
    if len(sentence.strip().split()) > 2 and trie is not None:
        # search for related chants
        results = trie.search_prefix(sentence + ' ')
        if len(results) > 0:
            notationbox = notationbox + '% RELATED CHANTS:\n'
            for result in results[:10]:
                chantinfo = session.get_chant_info(result[1])
                if not chantinfo:
                    genre = ''
                    feast = ''
                else:
                    genre = get_genre_desc(chantinfo['info']['field_genre'])
                    feast = chantinfo['info']['field_feast']
                    if feast is None:
                        feast = ''
                notationbox = notationbox + '% - (' + result[1] + ')(' + genre + ')(' + feast + ')' + result[0] + '\n'
            notationbox = notationbox + '\n'
    
    notationbox = notationbox + notation
    return notationbox


def clean_dictionary_for_submit(dataorig):
    for i in range(len(dataorig['annotationGroups'])):
        dataorig['annotationGroups'][i].pop('imageId', None)
        dataorig['annotationGroups'][i].pop('image', None)
        for j in range(len(dataorig['annotationGroups'][i]['annotations'])):
            dataorig['annotationGroups'][i]['annotations'][j].pop('annotationGroupId', None)
    return dataorig



def build_enclosing_polygon(data, f, group, grouplast, xini, xmax):
    """
    Construye un polígono de 8 puntos (sentido horario) que engloba
    el texto desde (f,group) hasta (f,grouplast) en el mismo folio f.
    Los boxes tienen formato [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]].
    xini recorta la izquierda del primer grupo; xmax recorta la derecha del último.
    Devuelve una lista de dicts: [{"x":..., "y":...}, ...] (8 puntos).
    """
    # parse boxes
    box_first = ast.literal_eval(data[f]['annotationGroups'][group]['boundingBox'])
    box_last  = ast.literal_eval(data[f]['annotationGroups'][grouplast]['boundingBox'])

    xmin_first, ymin_first = box_first[0]
    xmax_first, _          = box_first[1]
    _, ymax_first          = box_first[2]

    xmin_last, ymin_last = box_last[0]
    xmax_last, _         = box_last[1]
    _, ymax_last         = box_last[2]

    # Construcción de los 8 puntos en sentido horario empezando en la parte superior izquierda recortada
    # Caso de un grupo único
    if group == grouplast:
        pts = [
            {"x": xini,       "y": ymin_first},  # 1: sup-izq recortada
            {"x": xmax,       "y": ymin_first},  # 2: sup-derecha recortada
            {"x": xmax,       "y": ymax_first},  # 3: inf-derecha recortada
            {"x": xini,       "y": ymax_first},  # 4: inf-izq recortada
        ]
    # Caso de varios grupos
    else:
        pts = [
            {"x": xini,       "y": ymin_first},  # 1: sup-izq recortada (primer grupo)
            {"x": xmax_first, "y": ymin_first},  # 2: sup-derecha del primer grupo
            {"x": xmax_first, "y": ymin_last},   # 3: vertical hacia la parte superior del último grupo
            {"x": xmax,       "y": ymin_last},   # 4: sup-derecha recortada (último grupo)
            {"x": xmax,       "y": ymax_last},   # 5: inf-derecha recortada (último grupo)
            {"x": xmin_last,  "y": ymax_last},   # 6: inf-izq del último grupo
            {"x": xmin_last,  "y": ymax_first},  # 7: inf-izq del primer grupo
            {"x": xini,       "y": ymax_first},  # 8: inf-izq recortada (primer grupo)
        ]

    return pts



def create_notation_omr(data, i,
                        decision_begin, decision_last,
                        seq_folio, seq_group, seq_time):
    begin = decision_begin[i]
    last = decision_last[i]
    beginnext = decision_begin[i+1:i+2]

    folio = seq_folio[begin]
    group = seq_group[begin]
    foliolast = seq_folio[last-1]
    grouplast = seq_group[last-1]

    # Determine if chant begin at a staff edge
    time = 0.0
    if begin > 0 and seq_group[begin-1]==group and seq_folio[begin-1]==folio:
        time = seq_time[begin]
    idx = nearest_index(data[folio]['annotationGroups'][group]['mir_timesmu'], time)

    # Determine beginning of next chant
    if len(beginnext) == 0:
        timelast = 1.0
        idxlast = len(data[foliolast]['annotationGroups'][grouplast]['mir_timesmu'])
    else:
        beginnext = beginnext[0]
        if seq_group[beginnext-1]==seq_group[beginnext] and seq_folio[beginnext-1]==seq_folio[beginnext]:
            grouplast = seq_group[beginnext]
            foliolast = seq_folio[beginnext]
            timelast = seq_time[beginnext]
            # compruebo si el canto siguiente tiene anotacion manual
            if 'mir_timesmu' not in data[seq_folio[beginnext]]['annotationGroups'][seq_group[beginnext]]:
                idxlast = len(data[seq_folio[beginnext]]['annotationGroups'][seq_group[beginnext]]['notation'])
            else:
                idxlast = nearest_index(data[seq_folio[beginnext]]['annotationGroups'][seq_group[beginnext]]['mir_timesmu'],
                                        timelast)
        else:
            grouplast = seq_group[beginnext-1]
            foliolast = seq_folio[beginnext-1] 
            timelast = 1.0
            idxlast = len(data[foliolast]['annotationGroups'][grouplast]['mir_timesmu'])
    
    cur_folio = folio
    cur_group = group
    roman_idx = 1
    notation_str = ""
    while True:
        if len(data[cur_folio]['annotationGroups']) == 0:
            notes = ""
        elif 'mir_notationsmu' not in data[cur_folio]['annotationGroups'][cur_group]:
            notes = data[cur_folio]['annotationGroups'][cur_group]['notation']
        else:
            notes = data[cur_folio]['annotationGroups'][cur_group]['mir_notationsmu']
            
        # Caso especial: sólo un grupo en todo el rango
        if folio == foliolast and group == grouplast:
            part = notes[idx:idxlast]
        # Grupo inicial
        elif cur_folio == folio and cur_group == group:
            part = notes[idx:]
        # Grupo final
        elif cur_folio == foliolast and cur_group == grouplast:
            part = notes[:idxlast]
        # Grupos intermedios
        else:
            part = notes

        prefix = " " + roman.toRoman(roman_idx) + " "
        notation_str += prefix + part + " (z) "
        roman_idx += 1

        if cur_folio == foliolast and cur_group == grouplast:
            break
        cur_group += 1
        if cur_group >= len(data[cur_folio]['annotationGroups']):
            cur_folio += 1
            cur_group = 0

    notation_str = re.sub(r"\s+", " ", notation_str).strip()

    # Build polygon
    if foliolast > folio:
        grouplast = len(data[folio]['annotationGroups']) - 1
        foliolast = folio
        timelast = 1.0
    else:
        timelast = data[foliolast]['annotationGroups'][grouplast]['mir_timesmu'][idxlast-1]
    box = ast.literal_eval(data[folio]['annotationGroups'][group]['boundingBox'])
    xini = int(box[0][0] + time*(box[1][0]-box[0][0]))
    box = ast.literal_eval(data[folio]['annotationGroups'][grouplast]['boundingBox'])
    xmax = int(box[0][0] + timelast*(box[1][0]-box[0][0]))
    polygon = build_enclosing_polygon(data, folio, group, grouplast, xini, xmax)

    return notation_str, polygon
    


def submit_data(sequence,
                decision, decision_id, decision_begin, decision_last, decision_cost, decision_orphan,
                seq_folio, seq_group, seq_char, seq_time,
                data, dataorig,
                session, trie
                ):
    """Submit data to the server.
    """

    dataout = {}
    folio = len(data) // 2

    for i in range(len(decision)):
        if seq_folio[decision_begin[i]] != folio:
            continue

        # detect beginning and last group of the annotation (for UI drawing)
        group = seq_group[decision_begin[i]]
        if seq_folio[decision_last[i]] != folio:
            grouplast = len(data[folio]['annotationGroups']) - 1
        else:
            grouplast = seq_group[decision_last[i]]

        # no OMR
        if seq_time[decision_begin[i]] is None:
            notation = sequence[seq_char[decision_begin[i]]:seq_char[decision_last[i]]]
            polygon = []
        # OMR
        else:
            notation, polygon = create_notation_omr(data, i, decision_begin, decision_last,
                                           seq_folio, seq_group, seq_time)

        # create chant
        chantRecordId = session.add_new_chant_record(decision_id[i])

        # add annotation group with this chant
        if len(dataorig['annotationGroups']) == 0:
            maxorder_group = 0
            maxorder_annot = 0
        else:
            maxorder_group = max(group['order'] for group in dataorig['annotationGroups'])
            maxorder_group = maxorder_group + 1
            maxorder_annot = max(
                annotation['order']
                for group in dataorig['annotationGroups']
                for annotation in group.get('annotations', [])
            )
            maxorder_annot = maxorder_annot + 1
        
        random_seq = ''.join(random.choices(string.ascii_letters, k=8))
        bbox = ast.literal_eval(data[folio]['annotationGroups'][group]['boundingBox'])
        bboxlast = ast.literal_eval(data[folio]['annotationGroups'][grouplast]['boundingBox'])
        bboxsubmit = [[min(bbox[0][0],bboxlast[0][0]), bbox[0][1]],
                      [max(bbox[1][0],bboxlast[1][0]), bbox[1][1]],
                      [max(bbox[1][0],bboxlast[1][0]), bboxlast[2][1]],
                      [min(bbox[0][0],bboxlast[0][0]), bboxlast[3][1]]
                     ]
        # No OMR
        if len(polygon) == 0:
            polygon = [
                {
                "x": bboxsubmit[0][0],
                "y": bboxsubmit[0][1]
                },
                {
                "x": bboxsubmit[1][0],
                "y": bboxsubmit[1][1]
                },
                {
                "x": bboxsubmit[2][0],
                "y": bboxsubmit[2][1]
                },
                {
                "x": bboxsubmit[3][0],
                "y": bboxsubmit[3][1]
                }
            ]
        newgroup = {
            "id": random_seq,
            "order": maxorder_group,
            "userId": 21,
            "annotations": [
                {
                "id": None,
                "order": maxorder_annot,
                "polygon": polygon,
                # "polygon": [
                #     {
                #     "x": bboxsubmit[0][0],
                #     "y": bboxsubmit[0][1]
                #     },
                #     {
                #     "x": bboxsubmit[1][0],
                #     "y": bboxsubmit[1][1]
                #     },
                #     {
                #     "x": bboxsubmit[2][0],
                #     "y": bboxsubmit[2][1]
                #     },
                #     {
                #     "x": bboxsubmit[3][0],
                #     "y": bboxsubmit[3][1]
                #     }
                # ],
                "type": "CHANT"
                }
            ],
            "notation": create_notation_comments(decision[i], notation, trie, session),
            # "notation": '% ' + decision[i] + '\n\n' + notation,
            "chantRecordVersionId": chantRecordId,
            "boundingPolygon": str(bboxsubmit).replace(' ', ''),
            "boundingBox": str(bboxsubmit).replace(' ', ''),
            "notes": str(decision_cost[i]) + ',' + str(decision_orphan[i]) 
            # "boundingPolygon": data[1]['annotationGroups'][group]['boundingBox'],
            # "boundingBox": data[1]['annotationGroups'][group]['boundingBox']
        }
        dataorig['annotationGroups'].extend([newgroup])

        # Submission
        response = session.add_new_chant_annotation(dataorig)
        
        # Retrieve response and clean it
        dataorig = response.json()['image']
        dataorig = clean_dictionary_for_submit(dataorig)
        dataout = copy.deepcopy(dataorig)
   
    return dataout


def compute_similarity_hybrid(data, dataorig, session: SporkWorkspaceSession,
                              dictionary = None, cantusid = None, trie = None):
    """Decide sequence of chants from a OMR transcribed source.
    """

    gc.collect()

    # Read OMR transcription (lyrics)
    sequence, sequenceclean, seq_folio, seq_group, seq_box, seq_char, seq_time = create_omr_sequence(data)
    if len(sequenceclean) == 0:
        print('No lyrics found in OMR data.')
        return [{} for _ in range(len(dataorig))]

    # Lyrics dictionary
    if dictionary is None or cantusid is None:
        dictionary, cantusid = get_chant_fulltext()

    #resultados = align_chant_windows2(sequenceclean, dictionary[6650])

    print('ALIGNING WITH MELODY/LYRICS DICTIONARIES ...')

    # Local alignment (lyrics)
    sequence_foralign = sequenceclean
    max_alignments = 1
    with Pool(cpu_count(), initializer=init_pool, initargs=(sequence_foralign,max_alignments,)) as pool:
        results = list(tqdm(pool.imap(align_chant_windows, dictionary),
                            total=len(dictionary)))

    # Alignment score matrix (lyrics)
    rows = len(dictionary)
    cols = len(sequenceclean)
    matrix = np.zeros((rows, cols))

    # Fill matrix (lyrics)
    cantusidfil = []
    dictionaryfil = []
    row = 0
    for c, (score, start, end) in tqdm(enumerate(results)): # chant
        chantadded = False
        for i, _ in enumerate(score): # alignment
            if score[i] <= LYRICS_MIN_SCORE:
                continue
            matrix[row, start[i]:end[i]] = score[i]
            if not chantadded:
                cantusidfil.append(cantusid[c])
                dictionaryfil.append(dictionary[c])
                chantadded = True
        if chantadded:
            row = row + 1
    matrix = matrix[:row,:]

    print('SEQUENCE DECISION (sorry, this is a bit slow right now, wait for patches) ...')

    # Decision
    q, _, _ = dpcore_chant(matrix)

    # Retrieve the indices corresponding to chant beginnings
    idx_p = np.where(np.diff(q) != 0)[0] + 1
    idx_p = np.insert(idx_p, 0, 0)
    idx_q = q[idx_p]

    firstchar = idx_p
    decision = [dictionaryfil[i] for i in idx_q]
    decision_id = [cantusidfil[i] for i in idx_q]

    def get_decision_info(q, p):
        """Get chant alignment info (cost, begin, end) from alignment path.
        """
        #global filtered_matrix

        len = matrix.shape[1]

        decision_cost = [None] * q.size
        decision_begin = [None] * q.size
        decision_last = [None] * q.size # this is actually last+1

        for i in range(0, p.size):
            qi = q[i]
            pi = p[i]
            c = pi
            while matrix[qi,c] == 0:
                c = c+1
            decision_begin[i] = c
            decision_cost[i] = matrix[qi,c]
            c = c+1
            while c < len and matrix[qi,c] > 0:
                c = c+1
            decision_last[i] = c

        return decision_begin, decision_last, decision_cost

    decision_begin, decision_last, decision_cost = get_decision_info(idx_q, idx_p)


    def filter_decisions(decision, decision_id,
                         decision_begin, decision_last, decision_cost):
        """Filter decisions based on cost and length thresholds.
        """
        DECISION_MIN_COST = 0.5
        DECISION_MIN_LENGTH = 50
        decision_f, decision_id_f, decision_begin_f, decision_last_f, decision_cost_f = [], [], [], [], []
        for i in range(0, len(decision)):
            length = decision_last[i] - decision_begin[i]
            if decision_cost[i] >= DECISION_MIN_COST or length >= DECISION_MIN_LENGTH:
                decision_f.append(decision[i])
                decision_id_f.append(decision_id[i])
                decision_begin_f.append(decision_begin[i])
                decision_last_f.append(decision_last[i])
                decision_cost_f.append(decision_cost[i])
        decision = decision_f
        decision_id = decision_id_f
        decision_begin = decision_begin_f
        decision_last = decision_last_f
        decision_cost = decision_cost_f
        return decision, decision_id, decision_begin, decision_last, decision_cost
    
    decision, decision_id, decision_begin, decision_last, decision_cost = filter_decisions(
        decision, decision_id,
        decision_begin, decision_last, decision_cost
    )

    # Para cada decision, determinar si hay grupos huerfanos entre la decision actual y la anterior.
    # En caso afirmativo, marcar el id del folio del primer grupo huerfano en una lista auxiliar.
    decision_orphan = [None] * len(decision)
    for i in range(0, len(decision)):
        if i == 0:
            begin_prev = 0
        else:
            begin_prev = decision_last[i-1] - 1
        begin_cur = decision_begin[i]
        folio_prev = seq_folio[begin_prev]
        group_prev = seq_group[begin_prev]
        folio_cur = seq_folio[begin_cur]
        group_cur = seq_group[begin_cur]

        # Mismo folio
        if folio_prev == folio_cur:
            if group_cur > group_prev + 1:
                decision_orphan[i] = data[folio_cur]['id']
        # Folios diferentes
        else:
            # Grupos restantes en folio anterior
            if group_prev < len(data[folio_prev]['annotationGroups']) - 1:
                decision_orphan[i] = data[folio_prev]['id']
            # Folios intermedios completos
            for f in range(folio_prev + 1, folio_cur):
                if len(data[f]['annotationGroups']) > 0:
                    decision_orphan[i] = data[folio_prev+1]['id'] if decision_orphan[i] is None else decision_orphan[i]
            # Grupos iniciales en folio actual
            if group_cur > 0:
                decision_orphan[i] = data[folio_cur]['id'] if decision_orphan[i] is None else decision_orphan[i]

    # Prepare data for submission
    for i in range(len(dataorig)):
        dataorig[i] = clean_dictionary_for_submit(dataorig[i])

    # Submit decision
    CONTEXT = 5
    dataout_block = []
    for i in range(len(dataorig)):
        # Local context
        local_neighbors = data[i : i + 2 * CONTEXT + 1]
        assert len(local_neighbors) == 2 * CONTEXT + 1
        page_out = submit_data(
            sequence,
            decision, decision_id, decision_begin, decision_last,
            decision_cost, decision_orphan,
            [x-i for x in seq_folio], seq_group, seq_char, seq_time,
            local_neighbors, dataorig[i],
            session, trie
        )
        dataout_block.append(page_out)

    return dataout_block

    # # Submit decision
    # dataout = submit_data(sequence,
    #             decision, decision_id, decision_begin, decision_last, decision_cost, decision_orphan,
    #             seq_folio, seq_group, seq_char, seq_time,
    #             data, dataorig,
    #             session, trie
    #             )

    # return dataout
