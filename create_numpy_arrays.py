# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates dataset files to be used by `deep_edge_trainer.py`.

The input should be edge-list text file, with lines like "node1 node2", where
the nodes can be strings or ints. The line depicts a relationship node1-node2
(undirected) or node1->node2 (directed). The node IDs (e.g. "node1") will be
mapped to integers in [0, |V| - 1], where |V| is the number of graph nodes. The
mapping will be saved in `index.pkl`.

If you use with --n2v_partition, the input graph (edge-list) will be partitioned
into train and test, both of equal number of edges, where the train partition is
connected, to produce graphs compatible with node2vec's setup. 

The output directory will be populated with files:
  train.txt.npy: int32.numpy array (|E|/2, 2) containing training edges.
  test.txt.npy: int32.numpy array (|E|/2, 2) containing test edges.
  train.neg.txt.npy: int32.numpy array (|E|/2, 2) containing negative trai
    edges, sampled from compliment of (train.txt.npy).
  test.neg.txt.npy: int32.numpy array (|E|/2, 2) containing negative test edges,
    sampled from compliment of (test.txt.npy union train.txt.npy)

See doc of `CreateDatasetFiles()` for complete list of files and description.
"""

import pickle
import copy
import random
import networkx as nx
import numpy
import os
import sys

import tensorflow as tf

from tensorflow import flags


flags.DEFINE_string('input', '',
                    '[Required] Path to edge-list textfile.')
flags.DEFINE_string('output_dir', '',
                    'Directory where training files will be written.')
flags.DEFINE_boolean('directed', False, 'Must be set if graph is directed.')
flags.DEFINE_boolean('n2v_partition', False,
                     'If set, prunes the graph to the largest connected-'
                     'component, and splits half of train into test. Train '
                     'graph is then guaranteed to be connected [but test is not, '
                     "following node2vec's setup]. If leave unset to, test "
                     'file will be empty, but all connected-components will be '
                     'kept.')
FLAGS = flags.FLAGS


def LargestSubgraph(graph):
  """Returns the Largest connected-component of `graph`."""
  if graph.__class__ == nx.Graph:
    return LargestUndirectedSubgraph(graph)
  elif graph.__class__ == nx.DiGraph:
    largest_undirected_cc = LargestUndirectedSubgraph(nx.Graph(graph))
    directed_subgraph = nx.DiGraph()
    for (n1, n2) in graph.edges():
      if n2 in largest_undirected_cc and n1 in largest_undirected_cc[n2]:
        directed_subgraph.add_edge(n1, n2)

    return directed_subgraph

def LargestUndirectedSubgraph(graph):
  """Returns the largest connected-component of undirected `graph`."""
  if nx.is_connected(graph):
    return graph

  cc = nx.connected_component_subgraphs(graph)
  sizes = map(len, cc)
  sizes_and_cc = sorted(zip(sizes, cc))
  sizes_and_cc = list(sizes_and_cc)  # Is necessary in python3 to index (below)?
  return sizes_and_cc[-1][1]

def SampleTestEdgesAndPruneGraph(graph, remove_percent=0.5, check_every=5):
  """Removes and returns `remove_percent` of edges from graph.

  Removal is random but makes sure graph stays connected."""
  graph = copy.deepcopy(graph)
  undirected_graph = graph.to_undirected()

  edges = copy.deepcopy(graph.edges())
  random.shuffle(edges)
  remove_edges = int(len(edges) * remove_percent)
  num_edges_removed = 0
  currently_removing_edges = []
  removed_edges = []
  last_printed_prune_percentage = -1
  for j in range(len(edges)):
    n1, n2 = edges[j]
    graph.remove_edge(n1, n2)
    if n1 not in graph[n2]:
      undirected_graph.remove_edge(*(edges[j]))
    currently_removing_edges.append(edges[j])
    if j % check_every == 0:
      if nx.is_connected(undirected_graph):
        num_edges_removed += check_every
        removed_edges += currently_removing_edges
        currently_removing_edges = []
      else:
        for i in range(check_every):
          graph.add_edge(*(edges[j - i]))
          undirected_graph.add_edge(*(edges[j - i]))
        currently_removing_edges = []
        if not nx.is_connected(undirected_graph):
          print('  DID NOT RECOVER :(')
          return None
    prunned_percentage = int(100 * len(removed_edges) / remove_edges)
    rounded = (prunned_percentage / 10) * 10
    if rounded != last_printed_prune_percentage:
      last_printed_prune_percentage = rounded
      print('Partitioning into train/test. Progress=%i%%' % rounded)

    if len(removed_edges) >= remove_edges:
      break

  return graph, removed_edges


def SampleNegativeEdges(graph, num_edges):
  """Samples `num_edges` edges from compliment of `graph`."""
  random_negatives = set()
  nodes = list(graph.nodes())
  while len(random_negatives) < num_edges:
    i1 = random.randint(0, len(nodes) - 1)
    i2 = random.randint(0, len(nodes) - 1)
    if i1 == i2:
      continue
    if i1 > i2:
      i1, i2 = i2, i1
    n1 = nodes[i1]
    n2 = nodes[i2]
    if graph.has_edge(n1, n2):
      continue
    random_negatives.add((n1, n2))

  return random_negatives


def RandomNegativesPerNode(graph, negatives_per_node=400):
  """For every node u in graph, samples 20 (u, v) where v is not in graph[u]."""
  negatives = []
  node_list = list(graph.nodes())
  num_nodes = len(node_list)
  print_every = num_nodes / 10
  for i, n in enumerate(node_list):
    found_negatives = 0
    if i % print_every == 0:
      print('Finished sampling negatives for %i / %i nodes' % (i, num_nodes))
    while found_negatives < negatives_per_node:
      n2 = node_list[random.randint(0, num_nodes - 1)]
      if n == n2 or n2 in graph[n]:
        continue
      negatives.append((n, n2))
      found_negatives += 1
  return negatives


def NumberNodes(graph):
  """Returns a copy of `graph` where nodes are replaced by incremental ints."""
  node_list = sorted(graph.nodes())
  index = {n: i for (i, n) in enumerate(node_list)}

  newgraph = graph.__class__()
  for (n1, n2) in graph.edges():
    newgraph.add_edge(index[n1], index[n2])

  return newgraph, index


def MakeDirectedNegatives(positive_edges):
  positive_set = set([(u, v) for (u, v) in list(positive_edges)])
  directed_negatives = []
  for (u, v) in positive_set:
    if (v, u) not in positive_set:
      directed_negatives.append((v, u))
  return numpy.array(directed_negatives, dtype='int32')


def CreateDatasetFiles(graph, output_dir, partition=True):
  """Writes a number of dataset files to `output_dir`.

  Args:
    graph: nx.Graph or nx.DiGraph to simulate walks on and extract negatives.
    output_dir: files will be written in this directory, including:
      {train, train.neg, test, test.neg}.txt.npy, index.pkl, and
      if flag --directed is set, test.directed.neg.txt.npy.
      The files {train, train.neg}.txt.npy are used for model selection;
      {test, test.neg, test.directed.neg}.txt.npy will be used for calculating
      eval metrics; index.pkl contains information about the graph (# of nodes,
      mapping from original graph IDs to new assigned integer ones in
      [0, largest_cc_size-1].
    partition: If set largest connected component will be used and data will 
      separated into train/test splits.

  Returns:
    The training graph, after node renumbering.
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  original_size = len(graph)
  if partition:
    print('Taking largest subgraph (connected component)')
    graph = LargestSubgraph(graph)
    size_largest_cc = len(graph)
  else:
    size_largest_cc = -1
  graph, index = NumberNodes(graph)

  if partition:
    print('Partitioning into train / test')
    train_graph, test_edges = SampleTestEdgesAndPruneGraph(graph)
  else:
    train_graph, test_edges = graph, []

  # Sample negatives, to be equal to number of `test_edges` * 2.
  random_negatives = list(
      SampleNegativeEdges(graph, len(test_edges) + len(train_graph.edges())))
  random.shuffle(random_negatives)
  test_negatives = random_negatives[:len(test_edges)]
  # These are only used for evaluation, never training.
  train_eval_negatives = random_negatives[len(test_edges):]

  test_negatives = numpy.array(test_negatives, dtype='int32')
  test_edges = numpy.array(test_edges, dtype='int32')
  train_edges = numpy.array(train_graph.edges(), dtype='int32')
  train_eval_negatives = numpy.array(train_eval_negatives, dtype='int32')

  numpy.save(os.path.join(output_dir, 'train.txt'), train_edges)
  numpy.save(os.path.join(output_dir, 'train.neg.txt'), train_eval_negatives)
  numpy.save(os.path.join(output_dir, 'test.txt'), test_edges)
  numpy.save(os.path.join(output_dir, 'test.neg.txt'), test_negatives)
  if FLAGS.directed:
    directed_negatives = MakeDirectedNegatives(
        numpy.concatenate([train_edges, test_edges], axis=0))
    directed_negatives = numpy.concatenate([directed_negatives, test_negatives],
                                           axis=0)
    numpy.save(
        os.path.join(output_dir, 'test.directed.neg.txt'), directed_negatives)

  pickle.dump({
      'index': index,
      'original_num_nodes': original_size,
      'largest_cc_num_nodes': size_largest_cc,
      'num_pos_test_edges': len(test_edges),
      'num_neg_test_edges': len(test_negatives),
      'num_pos_train_edges': len(train_edges),
      'num_neg_train_edges': len(train_eval_negatives),
  }, open(os.path.join(output_dir, 'index.pkl'), 'wb'))

  print('Wrote all graph edges (positive & negative splits)')

  return train_graph



def main(unused_argv):
  if FLAGS.directed:
    graph = nx.DiGraph()
  else:
    graph = nx.Graph()

  # Read graph
  graph = nx.read_edgelist(FLAGS.input, create_using=graph)

  # Create dataset files.
  CreateDatasetFiles(graph, FLAGS.output_dir, partition=FLAGS.n2v_partition)

if __name__ == '__main__':
  tf.compat.v1.app.run(main)
