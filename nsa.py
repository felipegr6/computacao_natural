import csv
import random as r
import math as m
import numpy as np
import sys
import time as t

from multiprocessing import Pool
import requests, json

def set_status(status, title, description, id_process = '', result = ''):
    url = 'https://us-central1-myjobs-e88f6.cloudfunctions.net/jobs'

    if id_process == '':
        payload = {'status': status, 'title': title, 'description': description}
    else:
        payload = {'status': status, 'title': title, 'description': description, 'id': id_process, 'result': result}

    r = requests.post(url, json = payload)
    print(r.text)
    result = json.loads(r.text)
    return result['id']

def get_self_data():
    with open('creditcard.csv') as csv_file:
        data = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                d = []
                for i in range(1, 30):
                    d.append(float(row[i]))
                d.append(int(row[30]))
                data.append(d)
            line_count += 1

        aux_array = np.array(data)
        aux_array = aux_array.transpose()
        new_data = []
        
        for i in range(len(aux_array) - 2):
            row = []
            aux = aux_array[i]
            std = aux.std()
            mean = aux.mean()

            minimum = np.amin(aux)
            maximum = np.amax(aux)

            # print('{} {}'.format(minimum, maximum))

            for a in aux:
                # row.append((a - mean)/std)
                row.append((a - minimum)/(maximum - minimum))
            new_data.append(row)
        new_data.append(aux_array[-1].tolist())
        new_data = np.array(new_data).transpose().tolist()
        
        class_self     = [item for item in new_data if int(item[-1]) == 0]
        class_non_self = [item for item in new_data if int(item[-1]) == 1]

        return (new_data, class_self, class_non_self)

def xyz(self_class, non_self_class):
    distances = []

    for sc in self_class:
        for nsc in non_self_class:
            distances.append(eucledian_distance(sc, nsc))

    return (np.mean(distances), np.std(distances))

def generate_row(quantity, size):
    data = []
    for _ in range(quantity):
        d = []
        for _ in range(size):
            # k = r.randint(0,1)
            # if k == 0: k = -1
            # d.append(k * r.random())
            d.append(r.random())
        data.append(d)
    return data

def generate_random_detectors(num_detectors, size):
    detectors = generate_row(num_detectors,size)

    return detectors

def eucledian_distance(detector, data):
    l = len(detector)
    s = 0.
    for i in range(l):
        s += m.pow(detector[i] - data[i], 2)
    # print(m.sqrt(s))
    return m.sqrt(s)

def manhatan_distance(detector, data):
    l = len(detector)
    s = 0.
    for i in range(l):
        aux = detector[i] - data[i]
        if aux < 0.0: aux *= -1
        s += aux
    return s

def matches(detector, self_data, diff):
    for s in self_data:
        if eucledian_distance(detector, s) <= diff:
            return True
    return False

def match(detector, self_data, diff):
    if eucledian_distance(detector, self_data) <= diff:
            return True
    return False

def detector_generation(num_detectors, self_data, diff):
    repertorie = []
    
    while len(repertorie) < num_detectors:
        detectors = generate_random_detectors(num_detectors, len(self_data[0]) - 1)
        for d in detectors:
            if not matches(d, self_data, diff):
                repertorie.append(d)
    return repertorie

def secs(s):
    if s >= 60.0:
        return '{:2.3f} mins'.format(s/60.)
    return '{:2.3f} secs'.format(s)

def application(num_detectors, diff, training_percent):
    dt1 = t.time()
    self_data = get_self_data()
    dt2 = t.time()
    # print(xyz(self_data[1],self_data[2]))
    quantity = int(len(self_data[1]) * training_percent)
    detectors = detector_generation(num_detectors, self_data[1][0:quantity], diff)
    dt3 = t.time()
    class_self = []
    class_non_self = []

    for s in self_data[0]:
        is_self = True
        for d in detectors:
            if match(d, s, diff):
                is_self = False
                break
        if is_self:
            class_self.append(s)
        else:
            class_non_self.append(s)
    
    dt4 = t.time()

    # Fraud
    true_positive = 0
    false_positive = 0

    # Legitim
    true_negative = 0
    false_negative = 0

    for s in class_self:
        if int(s[-1]) == 0:
            true_negative += 1
        else:
            false_negative += 1

    for ns in class_non_self:
        if int(ns[-1]) == 1:
            true_positive += 1
        else:
            false_positive += 1

    # Sensitivity
    detection_rate = true_positive / (true_positive + false_negative)
    
    # Miss rate
    false_negative_rate = false_negative / (false_negative + true_positive)
    
    # Fall-out
    false_positive_rate = false_positive / (false_positive + true_negative)

    # Accuracy
    acc = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)

    total_time = dt4 - dt1

    # print('{} {} {}'.format(len(self_data[0]), len(self_data[1]), len(self_data[2])))
    # print('{} {}'.format(len(self_data[1]), len(class_self)))
    # print('{} {}'.format(len(self_data[2]), len(class_non_self)))
    # print('Read data: {}\nDetector generation: {}\nClassification: {}\nTotal: {}'.format(secs(dt2-dt1), secs(dt3-dt2), secs(dt4-dt3), secs(dt4-dt1)))
    # print('Detection: {:.3%}, False Negative: {:.3%}, False Positive: {:.3%}, Accuracy: {:.3%}'.format(detection_rate, false_negative_rate, false_positive_rate, acc))
    # print('tp: {}, fp: {}, tn: {}, fn: {}'.format(true_positive, false_positive, true_negative, false_negative))

    return (detection_rate, false_negative_rate, false_positive_rate, total_time, acc)

def app(a):
    return application(a[0], a[1], a[2])

if __name__ == '__main__':
    name_program = sys.argv[0]
    number_detectors = int(sys.argv[1])
    threshold = float(sys.argv[2])
    training_percent = float(sys.argv[3])

    description = 'num detectors: {}, threshold: {}, training_percent: {:.3%}'.format(number_detectors, threshold,training_percent)

    id_process = set_status(0, name_program, description)

    print('Starting with id {}'.format(id_process))
    
    data = [(number_detectors, threshold, training_percent)] * 16
    
    dr = 0.0
    fnr = 0.0
    fpr = 0.0
    tt = 0.0
    a = 0.0
    with Pool(4) as p:
        results = p.map(app, data)
        for r in results:
           dr += r[0]
           fnr += r[1]
           fpr += r[2]
           tt += r[3]
           a += r[4]
        n = len(results)
        print('==================')
        result = 'Detection: {:.3%}, False Negative: {:.3%}, False Positive: {:.3%}, Total Time = {}, Accuracy: {:.3%}'.format(dr/n, fnr/n, fpr/n, secs(tt/n), a/n)
        print(result)
        set_status(1, name_program, description, id_process, result)