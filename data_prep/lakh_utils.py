"""
The Lakh MIDI Dataset utilities
"""

import json
import os

from typing import Dict
import matplotlib.pyplot as plt
from bokeh.colors.groups import purple as colors

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def msd_id_to_dirs(msd_id: str) -> str:
  """
  Given an MSD ID, generate the path prefix.
  E.g. TRABCD12345678 -> A/B/C/TRABCD12345678

  :param msd_id: the MSD id
  :return: the directory path
  """
  return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def get_midi_path(msd_id: str,
                  midi_md5: str,
                  dataset_path: str) -> str:
  """
  Given an MSD ID and MIDI MD5, return path to a MIDI file.

  :param msd_id: the MSD id
  :param midi_md5: the MD5 of the MIDI, use get_matched_midi_md5
  :param dataset_path: the dataset path
  :return: the MIDI path
  """
  return os.path.join(dataset_path,
                      "lmd_matched",
                      msd_id_to_dirs(msd_id),
                      midi_md5 + ".mid")


def msd_id_to_h5(msd_id: str,
                 dataset_path: str) -> str:
  """
  Given an MSD ID, return the path to the corresponding h5.

  :param msd_id: the MSD id
  :param dataset_path: the dataset path
  :return:
  """
  return os.path.join(dataset_path,
                      "lmd_matched_h5",
                      msd_id_to_dirs(msd_id) + ".h5")


def get_msd_score_matches(match_scores_path: str) -> Dict:
  """
  Returns the dictionary of scores from the match scores file.

  :param match_scores_path: the match scores path
  :return: the dictionary of scores
  """
  with open(match_scores_path) as f:
    return json.load(f)


def get_matched_midi_md5(msd_id: str, msd_score_matches: dict):
  """
  Returns the MD5 of the matched MIDI from its MSD id.

  :param msd_id: the MSD id
  :param msd_score_matches: the MSD score dict, use get_msd_score_matches
  :return: the matched MIDI MD5
  """
  max_score = 0
  matched_midi_md5 = None
  for midi_md5, score in msd_score_matches[msd_id].items():
    if score > max_score:
      max_score = score
      matched_midi_md5 = midi_md5
  if not matched_midi_md5:
    raise Exception(f"Not matched {msd_id}: {msd_score_matches[msd_id]}")
  return matched_midi_md5

def plot_bars(most_common_tags, title):
    #plt.figure(num=None, figsize=(10, 8), dpi=500)
    plt.bar([tag for tag, _ in most_common_tags],
            [count for _, count in most_common_tags],
            color=[color.name for color in colors
                    if color.name != "lavender"])
    plt.title(title)
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.ylabel("count")
    plt.show()