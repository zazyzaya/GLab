import networkx as nx
from edgecentric_class import EdgeCentricInterface

FLOWS = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/flows.txt'
EDGEMAP = { 
	'timestamp' : 	0, 
	'duration' : 	1,
	'src_computer': 2,
	'src_port': 	3,
	'dst_computer': 4,
	'dst_port': 	5,
	'protocol': 	6,
	'packet_count': 7,
	'byte_count': 	8
}

class EdgeCentricLANL(EdgeCentricInterface):
	def __init__(self):
		super().__init__(nodetype='placeholder', edgetype='protocol')

	
	def add_edge(self, G, streamer, nb, et):
		edge = next(streamer)

		
