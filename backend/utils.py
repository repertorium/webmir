from collections import defaultdict
from functools import cmp_to_key
import numpy as np


def extract_balanced_json(text):
    """
    Extracts a balanced JSON string from the given text.
    (looks for the first opening brace '{' and the corresponding
    closing brace '}').
    
    It returns the substring from the start of the text
    to the closing brace. If no balanced JSON is found, it returns None.
    """
    start = 1
    while start < len(text) and text[start] != '{':
        start += 1
    if start == len(text):
        return None

    # find correct closing brace
    brace_count = 0
    end = start
    while end < len(text):
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
            if brace_count == 0:
                break
        end += 1

    if brace_count != 0:
        return None

    return text[:end + 1]


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.entries = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, phrase: str, phrase_id: int):
        node = self.root
        for char in phrase:
            node = node.children[char]
            node.entries.append((phrase, phrase_id))

    def search_prefix(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.entries  # (sentence, id)


def horizontal_overlap(box1, box2):
    return not (box1[2] < box2[0] or box2[2] < box1[0])


def vertical_overlap(box1, box2):
    # return not (box1[3] < box2[1] or box2[3] < box1[1])
    y_top = max(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])
    overlap = max(0, y_bottom - y_top)
    height1 = box1[3] - box1[1]
    height2 = box2[3] - box2[1]
    return (overlap >= 0.5 * height1) and (overlap >= 0.5 * height2)


def group_boxes_into_columns(boxes):
    columns = []
    for box in boxes:
        added = False
        for column in columns:
            if any(horizontal_overlap(box, other) for other in column):
                column.append(box)
                added = True
                break
        if not added:
            columns.append([box])
    return columns


def sort_column(column):
    def compare_boxes(b1, b2):
        if vertical_overlap(b1, b2):
            return -1 if b1[0] < b2[0] else 1
        else:
            return -1 if b1[1] < b2[1] else 1

    return sorted(column, key=cmp_to_key(compare_boxes))


def sort_boxes(boxes):
    columns = group_boxes_into_columns(boxes)
    columns.sort(key=lambda col: min(b[0] for b in col))  # sort columns by xmin
    sorted_boxes = []
    for column in columns:
        sorted_boxes.extend(sort_column(column))
    sorted_idx = [boxes.index(x) for x in sorted_boxes]
    sorted_boxes = [boxes[i] for i in sorted_idx]
    return sorted_boxes


def nearest_index(arr, x):
    # Posición donde x se insertaría para mantener el orden
    idx = np.searchsorted(arr, x)

    # Casos borde
    if idx == 0:
        return 0
    if idx == len(arr):
        return len(arr) - 1

    # Candidatos
    before = arr[idx - 1]
    after = arr[idx]

    # En caso de empate, elegimos el índice menor (before)
    if abs(before - x) <= abs(after - x):
        return idx - 1
    else:
        return idx
    