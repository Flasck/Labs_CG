def parse_obj(filename):
    with open(filename, 'r') as obj:
        vertices = []
        textures = []
        faces = [[], []]
        for line in obj:
            line = line.split()
            if line[0] == "v":
                vertices.append([float(l) for l in line[1:]])
            if line[0] == "vt":
                textures.append([float(l) for l in line[1:]])
            if line[0] == "f":
                faces[0].append([int(l.split("/")[0]) for l in line[1:]])
                faces[1].append([int(l.split("/")[1]) for l in line[1:]])
        return vertices, textures, faces
