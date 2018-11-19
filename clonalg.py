import csv
import random as r
import math as m
import numpy as np
import sys
import time as t

def get_self_data():
    with open('/home/fgrocha/Downloads/creditcard.csv') as csv_file:
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

            minimum = np.amin(aux)
            maximum = np.amax(aux)

            for a in aux:
                row.append((a - minimum)/(maximum - minimum))
            new_data.append(row)
        new_data.append(aux_array[-1].tolist())
        new_data = np.array(new_data).transpose().tolist()
        
        class_self     = [item for item in new_data if int(item[-1]) == 0]
        class_non_self = [item for item in new_data if int(item[-1]) == 1]

        return (new_data, class_self, class_non_self)

def secs(s):
    if s >= 60.0:
        return '{:2.3f} mins'.format(s/60.)
    return '{:2.3f} secs'.format(s)

def eucledian_distance(x, y):
    l = len(x)
    s = 0.
    for i in range(l):
        s += m.pow(x[i] - y[i], 2)
    return m.sqrt(s)

def antibody_generate(quantity, size):
    data = []
    for _ in range(quantity):
        d = []
        for _ in range(size):
            d.append(r.random())
        data.append(d)
    
    # Affinity column
    for d in data:
        d.append(0)

    return data

def affinity_calculation(antigen, antibodies):
    affinities = []
    for ab in antibodies:
        affinities.append(eucledian_distance(antigen, ab))

    for i in range(len(antibodies)):
        antibodies[i][-1] = affinities[i]

    antibodies.sort(key=lambda antibody: antibody[-1]) # Sorts by affinity
    return antibodies

def calculate_mutation_rate(antibody, mutate_factor=-2.5):
  return m.exp(mutate_factor * antibody[-1])

def get_subset(data, training_percent):
    l = int(len(data) * training_percent)
    subset = []

    for i in range(l):
        subset.append(data[i])

    return subset

def point_mutation(bitstring, rate):
    child = []
    
    for i in range(len(bitstring) - 1):
        number = bitstring[i]
        value = 0.0
        if r.random() < rate:
            value = 1.0 - number
        else:
            value = number
        child.append(value)
    
    child.append(0.0)
    
    return child

def clone_and_hypermutate(antibodies, clone_factor):
    clones = []
    num_clones = int(len(antibodies) * clone_factor)
    
    for antibody in antibodies:
        m_rate = calculate_mutation_rate(antibody)
        for _ in range(num_clones):
            clone = point_mutation(antibody, m_rate)
            clones.append(clone)
    
    return clones

def match(antibodies, self_data, diff):
    for ab in antibodies:
        if eucledian_distance(ab, self_data) <= diff:
            return True
    return False

def application(num_generations, num_antibody, training_percent, clone_factor, diff):
    (data, sf, nsf) = get_self_data()
    
    antibodies = antibody_generate(num_antibody, len(data[0]) - 1)

    training_subset = get_subset(nsf, training_percent)

    for _ in range(num_generations):
        for t in training_subset:
            antibodies = affinity_calculation(t, antibodies)
            clones = affinity_calculation(t, clone_and_hypermutate(antibodies, clone_factor))
            antibodies.extend(clones)
            antibodies = affinity_calculation(t, antibodies)[0:num_antibody]

    class_self = []
    class_non_self = []

    for d in data:
        if match(antibodies, d, diff):
            class_non_self.append(d)
        else:
            class_self.append(d)

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

    detection_rate = true_positive / (true_positive + false_negative)
    false_negative_rate = false_negative / (false_negative + true_positive)
    false_positive_rate = false_positive / (false_positive + true_negative)

    print('{} {} {}'.format(len(data), len(sf), len(nsf)))
    print('{} {}'.format(len(sf), len(class_self)))
    print('{} {}'.format(len(nsf), len(class_non_self)))
    
    print('Detection: {:.3%}, False Negative: {:.3%}, False Positive: {:.3%}'.format(detection_rate, false_negative_rate, false_positive_rate))

application(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
