#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:38:02 2017

@author: tristan
"""

"""----------------------------------------------------------------------
-- Tristan Metadata and conv
----------------------------------------------------------------------"""

#%%
dictBass = {
        '1'     : 0,
        '#1'    : 1,
        'b2'    : 1,
        '2'     : 2,
        '#2'    : 3,
        'b3'    : 3,
        '3'     : 4,
        '4'     : 5,
        '#4'    : 6,
        'b5'    : 6,
        '5'     : 7,
        '#5'    : 8,
        'b6'    : 8,
        '6'     : 9,
        '#6'    : 10,
        'b7'    : 10,
        '7'     : 11,
        'b9'    : 3,
        '9'     : 4,
        'N'     : 12
        }

QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'maj9':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '9':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'min11':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '11':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#11':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj13':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min13':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b13':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '1':       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

a0 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'N',
    'dim':     'N',
    'sus4':    'N',
    'sus2':    'N',
    '7':       'maj',
    'maj7':    'maj',
    'min7':    'min',
    'minmaj7': 'min',
    'maj6':    'maj',
    '6':    'maj', #add
    'min6':    'min',
    'dim7':    'N',
    'hdim7':   'N',
    'hdim':    'N',
    'maj9':    'maj',
    'min9':    'min',
    '9':       'maj',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'} #alphabet = maj/min

a1 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'N',
    'dim':     'dim',
    'sus4':    'N',
    'sus2':    'N',
    '7':       'maj',
    'maj7':    'maj',
    'min7':    'min',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'dim',
    'hdim7':   'dim',
    'hdim':    'dim',
    'maj9':    'maj',
    'min9':    'min',
    '9':       'maj',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'} #alphabet = maj/min/dim -> harmonisation gamme majeur en accords de 3 sons
      
a2 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'N',
    'dim':     'dim',
    'sus4':    'N',
    'sus2':    'N',
    '7':       '7',
    '6':    'maj', #add
    'maj7':    'maj7',
    'min7':    'min7',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'dim7',
    'hdim7':   'dim',
    'hdim':    'dim',
    'maj9':    'maj7',
    'min9':    'min7',
    '9':       '7',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'} #alphabet = maj/min/maj7/min7/7/dim/dim7 -> harmonisation gamme majeur en accords de 4 sons

a3 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'aug',
    'dim':     'dim',
    'sus4':    'sus',
    'sus2':    'sus',
    '7':       '7',
    'maj7':    'maj7',
    'min7':    'min7',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'dim7',
    'hdim7':   'dim',
    'hdim':    'dim',
    'maj9':    'maj7',
    'min9':    'min7',
    '9':       '7',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'} #alphabet = maj/min/maj7/min7/7/dim/dim7/aug/sus

a5 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'aug',
    'dim':     'dim',
    'sus4':    'sus4',
    'sus2':    'sus2',
    '7':       '7',
    '6':    'maj', #add
    'maj7':    'maj7',
    'min7':    'min7',
    'minmaj7': 'minmaj7',
    'maj6':    'maj6',
    'min6':    'min6',
    'dim7':    'dim7',
    'hdim7':   'hdim7',
    'hdim':    'hdim7', 
    'maj9':    'maj7',
    'min9':    'min7',
    '9':       '7',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj', #maj7??
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N', #X
    '5':       'N', #X
    '': 'N'} #de l'article STRUCTURED TRAINING FOR LARGE-VOCABULARY CHORD RECOGNITION

gamme = {
    'Ab':   'G#',
    'A':    'A',
    'A#':   'A#',
    'Bb':   'A#',
    'B':    'B',
    'Cb':   'B',
    'C':    'C',
    'C#':   'C#',
    'Db':   'C#',
    'D':    'D',
    'D#':   'D#',
    'Eb':   'D#',
    'E':    'E',
    'Fb':   'E',
    'F':    'F',
    'F#':   'F#',
    'Gb':   'F#',
    'G':    'G',
    'G#':   'G#',
    'N' :   'N',
    'X' :   'N',
    '' :    'N'}

gammeKey = {
    'Ab':   'G#',
    'A':    'A',
    'Am':    'A:minor',
    'A#':   'A#',
    'A#':    'A#:minor',
    'Bb':   'A#',
    'Bbm':    'A#:minor',
    'B':    'B',
    'Bm':    'B:minor',
    'Cb':   'B',
    'Cbm':   'B:minor',
    'C':    'C',
    'Cm':    'C:minor',   
    'C#':   'C#',
    'C#m':    'C#:minor',
    'Db':   'C#',
    'Dbm':    'C#:minor',  
    'D':    'D',
    'Dm':    'D:minor',
    'D#':   'D#',
    'D#m':    'D#:minor',   
    'Eb':   'D#',
    'Ebm':    'D#:minor',
    'E':    'E',
    'Em':    'E:minor',
    'F':    'F',
    'Fm':    'F:minor',
    'F#':   'F#',
    'F#m':    'F#:minor',
    'Gb':   'F#',
    'Gbm':    'F#:minor',
    'G':    'G',
    'Gm':    'G:minor',
    'G#':   'G#',
    'G#m':    'G#:minor',
    'N' :   'N',
    '' :    'N'}

tr = {
    'G':    'G#',
    'G#':   'A',
    'A':    'A#',
    'A#':   'B',
    'B':    'C',
    'C':    'C#',
    'C#':   'D',
    'D':    'D#',
    'D#':   'E',
    'E':    'F',
    'F':    'F#',
    'F#':   'G',
    'N' :   'N',
    '' :    'N'}
