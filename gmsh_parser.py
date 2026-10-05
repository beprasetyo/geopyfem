from pathlib import Path
import sys

# Common Gmsh element types used by a 2-D FEM code.
# Format: gmsh_type: (name, number_of_nodes)
ELEMENT_TYPES = {
    1: ("Line2", 2),
    2: ("Tri3", 3),
    3: ("Quad4", 4),
    8: ("Line3", 3),
    9: ("Tri6", 6),
    10: ("Quad9", 9),
    15: ("Point1", 1),
    16: ("Quad8", 8),
}


def get_section(lines, section_name):
    """Return all lines inside a Gmsh $Section ... $EndSection."""
    start_marker = f"${section_name}"
    end_marker = f"$End{section_name}"

    try:
        start = lines.index(start_marker) + 1
        end = lines.index(end_marker)
    except ValueError as exc:
        raise ValueError(f"Section {start_marker} not found in the .msh file") from exc

    return lines[start:end]


def parse_physical_names(lines):
    """Read Physical Names and return {(dimension, tag): name}."""
    section = get_section(lines, "PhysicalNames")
    nphys = int(section[0])
    physical_names = {}

    for line in section[1:1 + nphys]:
        parts = line.split(maxsplit=2)
        dim = int(parts[0])
        tag = int(parts[1])
        name = parts[2].strip().strip('"')
        physical_names[(dim, tag)] = name

    return physical_names


def parse_entities(lines):
    """Return {(entity_dimension, entity_tag): [physical_tags, ...]}."""
    section = get_section(lines, "Entities")
    tokens = " ".join(section).split()
    p = 0

    npoints = int(tokens[p]); p += 1
    ncurves = int(tokens[p]); p += 1
    nsurfaces = int(tokens[p]); p += 1
    nvolumes = int(tokens[p]); p += 1

    entity_phys = {}

    # Points: tag x y z numPhysicalTags physicalTags...
    for _ in range(npoints):
        tag = int(tokens[p]); p += 1
        p += 3  # x, y, z
        nphys = int(tokens[p]); p += 1
        phys = [int(tokens[p + i]) for i in range(nphys)]
        p += nphys
        entity_phys[(0, tag)] = phys

    # Curves, surfaces, volumes:
    # tag minX minY minZ maxX maxY maxZ numPhysicalTags ...
    # numBoundingEntities boundingEntityTags...
    for dim, count in ((1, ncurves), (2, nsurfaces), (3, nvolumes)):
        for _ in range(count):
            tag = int(tokens[p]); p += 1
            p += 6  # bounding box
            nphys = int(tokens[p]); p += 1
            phys = [int(tokens[p + i]) for i in range(nphys)]
            p += nphys

            nbound = int(tokens[p]); p += 1
            p += nbound

            entity_phys[(dim, tag)] = phys

    return entity_phys


def parse_nodes(lines):
    """Return {gmsh_node_tag: (x, y, z)} from a Gmsh 4.1 ASCII file."""
    section = get_section(lines, "Nodes")
    tokens = " ".join(section).split()
    p = 0

    nblocks = int(tokens[p]); p += 1
    total_nodes = int(tokens[p]); p += 1
    p += 2  # minNodeTag, maxNodeTag

    nodes = {}

    for _ in range(nblocks):
        entity_dim = int(tokens[p]); p += 1
        _entity_tag = int(tokens[p]); p += 1
        parametric = int(tokens[p]); p += 1
        nblock_nodes = int(tokens[p]); p += 1

        node_tags = [int(tokens[p + i]) for i in range(nblock_nodes)]
        p += nblock_nodes

        ncoord = 3 + (entity_dim if parametric else 0)
        for tag in node_tags:
            values = [float(tokens[p + i]) for i in range(ncoord)]
            p += ncoord
            nodes[tag] = (values[0], values[1], values[2])

    if len(nodes) != total_nodes:
        raise ValueError(
            f"Expected {total_nodes} nodes, but parsed {len(nodes)} nodes"
        )

    return nodes


def parse_elements(lines):
    """
    Read element blocks.

    Returns a list of dictionaries containing the Gmsh element tag,
    entity dimension/tag, element type, and Gmsh node tags.
    """
    section = get_section(lines, "Elements")
    tokens = " ".join(section).split()
    p = 0

    nblocks = int(tokens[p]); p += 1
    total_elements = int(tokens[p]); p += 1
    p += 2  # minElementTag, maxElementTag

    elements = []

    for _ in range(nblocks):
        entity_dim = int(tokens[p]); p += 1
        entity_tag = int(tokens[p]); p += 1
        gmsh_type = int(tokens[p]); p += 1
        nblock_elements = int(tokens[p]); p += 1

        if gmsh_type not in ELEMENT_TYPES:
            raise NotImplementedError(
                f"Gmsh element type {gmsh_type} is not yet supported by this parser"
            )

        type_name, nnode = ELEMENT_TYPES[gmsh_type]

        for _ in range(nblock_elements):
            element_tag = int(tokens[p]); p += 1
            connectivity = [int(tokens[p + i]) for i in range(nnode)]
            p += nnode

            elements.append({
                "gmsh_tag": element_tag,
                "entity_dim": entity_dim,
                "entity_tag": entity_tag,
                "gmsh_type": gmsh_type,
                "type": type_name,
                "nodes": connectivity,
            })

    if len(elements) != total_elements:
        raise ValueError(
            f"Expected {total_elements} elements, but parsed {len(elements)} elements"
        )

    return elements


def physical_group_names(entity_dim, entity_tag, entity_phys, physical_names):
    """Return all Physical Group names attached to an entity."""
    tags = entity_phys.get((entity_dim, entity_tag), [])
    return [
        physical_names[(entity_dim, tag)]
        for tag in tags
        if (entity_dim, tag) in physical_names
    ]


def parse_gmsh(filename, include_group_elements=False):
    """
    Parse a Gmsh 4.1 ASCII mesh and build FEM-friendly data.

    Returns by default
    ------------------
    nodes, domain_elements, physical_groups

    If include_group_elements=True, a fourth object is returned:
    physical_group_elements.  This preserves the actual Line2/Line3 boundary
    connectivity, which is required to integrate distributed traction loads.

    Boundary-condition meaning is NOT assigned here.  Gmsh tells us where a
    physical group is; the XML problem specification tells the solver what that
    group means mechanically.
    """
    path = Path(filename)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    mesh_format = get_section(lines, "MeshFormat")[0].split()
    if mesh_format[0] != "4.1":
        raise ValueError(f"This parser expects Gmsh 4.1; found version {mesh_format[0]}")
    if mesh_format[1] != "0":
        raise ValueError("Binary .msh is not supported. Export the mesh as ASCII.")

    physical_names = parse_physical_names(lines)
    entity_phys = parse_entities(lines)
    gmsh_nodes = parse_nodes(lines)
    all_elements = parse_elements(lines)

    # Reindex Gmsh node tags to 0, 1, 2, ... for Python arrays.
    # This is ID conversion only, not spatial reordering.
    sorted_node_tags = sorted(gmsh_nodes)
    node_index = {gmsh_tag: i for i, gmsh_tag in enumerate(sorted_node_tags)}

    nodes = []
    for gmsh_tag in sorted_node_tags:
        x, y, _z = gmsh_nodes[gmsh_tag]
        nodes.append({
            "id": node_index[gmsh_tag],
            "x": x,
            "y": y,
            # The parser no longer decides mechanical boundary conditions.
            # These remain free until main.py applies the XML specification.
            "ux": 0,
            "uy": 0,
        })

    physical_groups = {}
    physical_group_elements = {}
    domain_elements = []

    for element in all_elements:
        names = physical_group_names(
            element["entity_dim"],
            element["entity_tag"],
            entity_phys,
            physical_names,
        )

        zero_based_connectivity = [node_index[tag] for tag in element["nodes"]]

        # Preserve both node membership and actual element connectivity
        # for every Physical Group.
        for name in names:
            physical_groups.setdefault(name, set()).update(zero_based_connectivity)
            physical_group_elements.setdefault(name, []).append({
                "dimension": element["entity_dim"],
                "type": element["type"],
                "connectivity": zero_based_connectivity,
            })

        # Only 2-D elements are continuum/domain elements in this 2-D FEM code.
        if element["entity_dim"] != 2:
            continue

        if len(names) == 0:
            material = "UNASSIGNED"
        elif len(names) == 1:
            material = names[0]
        else:
            material = "+".join(names)

        domain_elements.append({
            "id": len(domain_elements),
            "type": element["type"],
            "material": material,
            "connectivity": zero_based_connectivity,
        })

    groups_zero_based = {
        name: sorted(node_ids)
        for name, node_ids in physical_groups.items()
    }

    if include_group_elements:
        return (
            nodes,
            domain_elements,
            groups_zero_based,
            physical_group_elements,
        )

    # Backward-compatible 3-value return for existing code.
    return nodes, domain_elements, groups_zero_based


def write_outputs(filename, nodes, elements):
    """Write nodes_<stem>.txt and elements_<stem>.txt."""
    path = Path(filename)
    stem = path.stem
    outdir = path.parent

    nodes_file = outdir / f"nodes_{stem}.txt"
    elements_file = outdir / f"elements_{stem}.txt"

    with nodes_file.open("w") as f:
        f.write("# node_id\tx\ty\tux\tuy\n")
        for node in nodes:
            f.write(
                f'{node["id"]}\t{node["x"]:.16g}\t{node["y"]:.16g}'
                f'\t{node["ux"]}\t{node["uy"]}\n'
            )

    with elements_file.open("w") as f:
        f.write("# element_id\telement_type\tmaterial\tconnectivity...\n")
        for element in elements:
            conn = "\t".join(str(n) for n in element["connectivity"])
            f.write(
                f'{element["id"]}\t{element["type"]}\t{element["material"]}\t{conn}\n'
            )

    return nodes_file, elements_file


def main():
    if len(sys.argv) != 2:
        print("Usage: python gmsh_parser.py meshfile.msh")
        raise SystemExit(1)

    meshfile = sys.argv[1]
    nodes, elements, groups = parse_gmsh(meshfile)
    nodes_file, elements_file = write_outputs(meshfile, nodes, elements)

    print(f"Nodes    : {len(nodes)}")
    print(f"Elements : {len(elements)}")
    print("Physical groups:")
    for name, node_ids in groups.items():
        print(f"  {name}: {len(node_ids)} nodes")
    print(f"Written  : {nodes_file.name}")
    print(f"Written  : {elements_file.name}")


if __name__ == "__main__":
    main()
