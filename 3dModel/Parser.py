def parse_obj(filename):
    with open(filename, 'r') as obj:
        vertices = []
        faces = []
        for line in obj:
            line = line.split()
            if line[0] == "v":
                vertices.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == "f":
                faces.append([int(line[1].split("/")[0]), int(line[2].split("/")[0]), int(line[3].split("/")[0])])
        return vertices, faces
