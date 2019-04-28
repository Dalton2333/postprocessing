import geometry.boundary_extractor

def plt_cross_section(ap,part):
    node_coords = []
    cross_section = geometry.boundary_extractor.get_cross_section(ap, part)
    for elmt_label in cross_section.data:
        element = part.elements[elmt_label]
        for node_label in element.nodes:
            node = part.nodes[node_label]
            node_coords.append(node.coordinates)
    import matplotlib.pyplot as plt
    import numpy
    tr_coords = list(numpy.transpose(node_coords))
    plt.scatter(tr_coords[0],tr_coords[1],tr_coords[2],c='r', marker='.',s=1)
    plt.title("cross section")
    plt.show()

