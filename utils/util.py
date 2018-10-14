# -*- coding: utf-8 -*-

import json

def save_to_json(model, filename='model.json'):
    with open(filename, mode='w') as f:
        f.write(model.to_json(indent=4))


