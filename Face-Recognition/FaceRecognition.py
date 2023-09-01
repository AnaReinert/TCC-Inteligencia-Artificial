import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos
    return False, []


def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    ana = reconhece_face("./img/ana.jpg")
    if (ana[0]):
        rostos_conhecidos.append(ana[1][0])
        nomes_dos_rostos.append("Ana")

    return rostos_conhecidos, nomes_dos_rostos