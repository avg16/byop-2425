#ar script here
class OBJ():
    def __init__(self, filename, swap_y_z = False):
        "swap y z to match the software's config"
        self.faces = []
        self.texture_coords = []
        self.normals = []
        self.vertices = []
        material = None

        for line in open(filename, "r"):
            if line.startswith('#'): 
                continue #if any comment comes in b/w
            values = line.split()
            if not values:
                continue
            if values[0] == 'v': # vertex
                v = list(map(float, values[1:4]))
                if swap_y_z== True:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vt': # texture coordinate
                v = list(map(float, values[1:3]))
                self.texture_coords.append(v)
            elif values[0] == 'vn': # normal
                v = list(map(float, values[1:4]))
                if swap_y_z==True:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            # elif values[0] == 'usemtl': # new material
            #     material = values[1]
            elif values[0] == 'f': # face
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, texcoords, norms))