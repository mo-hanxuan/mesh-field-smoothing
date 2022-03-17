from xml.dom import ValidationErr
from elementsBody import *
import numpy as np


def decide_ratio_draw(obj):
    if not isinstance(obj, ElementsBody):
        raise ValidationErr("error, not isinstance(obj, ElementsBody) ")
    ### get the maximum x and minimum, and obj1.ratio_draw
    xMax = max(obj.nodes.values(), key=lambda x: x[0])[0]
    xMin = min(obj.nodes.values(), key=lambda x: x[0])[0]
    yMax = max(obj.nodes.values(), key=lambda x: x[1])[1]
    yMin = min(obj.nodes.values(), key=lambda x: x[1])[1]
    zMax = max(obj.nodes.values(), key=lambda x: x[2])[2]
    zMin = min(obj.nodes.values(), key=lambda x: x[2])[2]
    print(
        "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\n"
        "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\n"
        "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\033[0m".format(
            "xMax =", xMax, "xMin =", xMin,
            "yMax =", yMax, "yMin =", yMin,
            "zMax =", zMax, "zMin =", zMin,
        )
    )
    obj.maxLen = max(xMax - xMin, yMax - yMin, zMax - zMin)
    obj.ratio_draw = 1. if obj.maxLen < 10. else 5. / obj.maxLen
    print("obj.ratio_draw =", obj.ratio_draw)
    return obj.ratio_draw