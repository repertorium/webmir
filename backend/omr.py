# imports
from io import BytesIO
import json
import numpy as np
import os
from PIL import Image
import random
import requests
import string
import time
from ultralytics import YOLO

from utils import sort_boxes
from repertorium_omr.preprocessing import preprocess_image_from_object2, IMG_HEIGHT
from repertorium_omr.model import CTCTrainedCRNN
from repertorium_omr.model2 import CTCTrainedCRNN2

# constants
#CKPT_YOLO = './backend/weights/best.pt'
#CKPT_LYRICS = './backend/weights/repertorium_lyrics_char_crnn_greedy.ckpt'
#CKPT_GABC = './backend/weights/gabc-music.ckpt'
#W2I_PATH = './backend/repertorium_omr/data/vocab/lyrics/w2i_char.json'

CKPT_YOLO = '/app/weights/best.pt'
CKPT_LYRICS = '/app/weights/repertorium_lyrics_char_crnn_greedy.ckpt'
CKPT_GABC = '/app/weights/gabc-music.ckpt'
W2I_PATH = '/app/repertorium_omr/data/vocab/lyrics/w2i_char.json'

USER_AGENT = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) ' \
             'Gecko/20100101 ' \
             'Firefox/126.0'

class OMRprocessor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT
        
        self.modelyolo = YOLO(CKPT_YOLO)

        _, i2w = self._retrieve_vocabulary()

        self.modelocr = CTCTrainedCRNN.load_from_checkpoint(CKPT_LYRICS,
            ctc='greedy',
            ytest_i2w=i2w
        )
        self.modelocr.to('cpu')
        self.modelocr.freeze()

        self.modelomr = CTCTrainedCRNN2.load_from_checkpoint(CKPT_GABC)
        self.modelomr.to('cpu')
        self.modelomr.freeze()

    def _retrieve_vocabulary(self):
        w2i, i2w = {}, {}
        with open(W2I_PATH, "r") as file:
            w2i = json.load(file)
        i2w = {v: k for k, v in w2i.items()}
        return w2i, i2w

    def _download_image(self, image_url):
        # create iiif link
        image_url = os.path.dirname(image_url)
        image_url = image_url + '/full/1000,/0/default.jpg'
        # download image
        try:
            response = self.session.get(image_url)
            imagen = Image.open(BytesIO(response.content))
            return imagen
        except Exception as e:
            raise e

    def _layout_detection(self, imagen):
        kwargs = {
            'save': False,
            'device': 'cpu',
            'imgsz': 512,
            'verbose': False
        }
        results = self.modelyolo.predict(imagen, **kwargs)
        if len(results) == 0:
            return []
        results = results[0].summary()
        boxes = [[round(result['box']['x1']), 
                  round(result['box']['y1']),
                  round(result['box']['x2']),
                  round(result['box']['y2'])]
                  for result in results if result['name'] == 'staff'
        ]
        return boxes

    def _estimate_lyrics_boxes(self, boxes):
        boxes = [[box[0], (box[1]+box[3])//2, box[2], (3*box[3]-box[1])//2]
                 for box in boxes]
        return boxes

    def _estimate_lyrics_and_music_boxes(self, boxes):
        boxes = [[box[0], box[1], box[2], (3*box[3]-box[1])//2] 
                 for box in boxes]
        return boxes

    def _transcript_lyrics(self, imagen, boxes):
        notations = []
        times = []
        for box in boxes:
            x = imagen.crop(tuple(box))
            x = preprocess_image_from_object2(x, image_height=IMG_HEIGHT)
            notation, time = self.modelocr.transcribe(x.unsqueeze(0))
            notations.append(notation)
            times.append(time)
        return notations, times

    def _transcript_music(self, imagen, boxes):
        notations = []
        times = []
        for box in boxes:
            x = imagen.crop(tuple(box))
            x = preprocess_image_from_object2(x, image_height=2*IMG_HEIGHT)
            notation, time = self.modelomr.transcribe(x.unsqueeze(0))
            notations.append(notation)
            times.append(time)
        return notations, times
    
    def _clean_transcript_music(self, notationsmu, timesmu):
        # change uppercase to lowercase
        notationsmu = [notation.lower() for notation in notationsmu]
        # remove virgas (character v)
        notationsmu_clean = []
        timesmu_clean = []
        for notation, time in zip(notationsmu, timesmu):
            notation_clean = ''
            time_clean = []
            for char, t in zip(notation, time):
                if char != 'v':
                    notation_clean += char
                    time_clean.append(t)
            notationsmu_clean.append(notation_clean)
            timesmu_clean.append(np.array(time_clean))
        return notationsmu_clean, timesmu_clean

    def _create_annotation_groups(self,
                                  notations, timesly, 
                                  notationsmu, timesmu,
                                  boxes):
        groups = []
        for i, notation, box in zip(range(len(notations)), notations, boxes):
            bbox = [[box[0],box[1]], [box[2],box[1]],
                    [box[2],box[3]], [box[0],box[3]]]
            random_seq = ''.join(random.choices(string.ascii_letters, k=8))
            newgroup = {
                "id": random_seq,
                "order": i,
                "userId": 21,
                "annotations": [{
                    "id": None,
                    "order": 0,
                    "polygon": [{"x": x, "y": y} for x, y in 
                                [(box[0],box[1]), (box[2],box[1]), 
                                 (box[2],box[3]), (box[0],box[3])]],
                    "type": "NOTATION",
                }],
                "notation": notation + " (-)",
                "notationtime": (
                    np.pad(timesly[i], (0,4), mode='edge')
                    if timesly[i].size > 0
                    else np.full(4, 1.0)
                ),
                "boundingPolygon": str(bbox).replace(" ", ""),
                "boundingBox": str(bbox).replace(" ", ""),
                "mir_notationsly": notations[i],
                "mir_timesly": timesly[i],
                "mir_notationsmu": notationsmu[i],
                "mir_timesmu": timesmu[i],
            }
            groups.append(newgroup)
        return groups

    def process_image(self, image_url, width=None, height=None):
        # Try and retry downloading image
        imagen = None
        while imagen is None:
            try:
                imagen = self._download_image(image_url)
            except Exception as e:
                print(f"Error downloading image: {e}")
                print(f"Retrying...")
                # wait a bit before retrying
                time.sleep(2)
        # Layout detection
        boxesmu = self._layout_detection(imagen)
        boxesmu = sort_boxes(boxesmu)
        # Lyrics
        boxesly = self._estimate_lyrics_boxes(boxesmu)
        notationsly, timesly = self._transcript_lyrics(imagen, boxesly)
        # Music
        notationsmu, timesmu = self._transcript_music(imagen, boxesmu)
        notationsmu, timesmu = self._clean_transcript_music(notationsmu, timesmu)
        # Create annotation groups
        if width is not None and height is not None:
            boxesmu = [[int(box[0] * width / imagen.width),
                        int(box[1] * height / imagen.height),
                        int(box[2] * width / imagen.width),
                        int(box[3] * height / imagen.height)] for box in boxesmu]
        groups = self._create_annotation_groups(notationsly, timesly,
                                                notationsmu, timesmu, boxesmu)
        return groups
